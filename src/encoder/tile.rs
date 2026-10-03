//! The encoder's decisions, made inside the tile walker.
//!
//! The walker of `decoder::tile` codes the syntax; in encode mode it asks
//! these methods for each decision just before the syntax codes it, with
//! the reconstruction so far in `CurrFrame`: the partition, the modes of a
//! block (from predictions made with the normative predictors), its motion
//! vector (a search on the reference), and the quantised levels of each
//! transform block (from the prediction then in place). With
//! rate-distortion optimisation on (`Tools::rdo`), the decisions are
//! searched by trial coding (`encoder::rdo`); these methods then supply the
//! candidates.

use crate::Result;
use crate::consts::*;
use crate::decoder::predict::block_inter_prediction;
use crate::decoder::state::Mv;
use crate::decoder::tile::{Plan, TileDecoder};
use crate::encoder::Tools;
use crate::encoder::fwd::forward_2d;
use crate::encoder::rdo::{Decision, Rdo};
use crate::symbol::Coder;
use crate::tables::*;

/// The encoder's per-frame state inside a tile.
pub(crate) struct EncCtx {
    /// The source planes, padded like `CurrFrame` (edges replicated).
    pub(crate) src: Vec<Vec<u16>>,
    pub(crate) stride: Vec<usize>,
    /// The quantised levels chosen for the transform block being coded,
    /// in `Quant`'s layout.
    pub(crate) coefs: Box<[i32; 1024]>,
    /// Lagrange multiplier for the SATD-based preselection (SATD units per
    /// bit).
    pub(crate) lambda: f64,
    /// Lagrange multiplier of the rate-distortion search (squared error
    /// per bit).
    pub(crate) rd_lambda: f64,
    /// The references inter prediction may use (`LAST_FRAME`...); empty
    /// on intra frames.
    pub(crate) refs: Vec<i32>,
    /// Half-width of the full-pel motion search.
    pub(crate) search_range: i32,
    pub(crate) tools: Tools,
    pub(crate) rdo: Rdo,
    /// Scratch: the residual and the forward transform of a block.
    pub(crate) res: Vec<i32>,
    pub(crate) fc: Vec<f64>,
}

impl EncCtx {
    fn src(&self, plane: usize, x: usize, y: usize) -> i32 {
        self.src[plane][y * self.stride[plane] + x] as i32
    }
}

/// Sum of absolute 4x4 Hadamard-transformed differences.
fn satd(a: &[i32], b: &[i32], w: usize, h: usize) -> u64 {
    let mut total = 0u64;
    for by in (0..h).step_by(4) {
        for bx in (0..w).step_by(4) {
            let mut d = [0i32; 16];
            for i in 0..4 {
                for j in 0..4 {
                    d[i * 4 + j] = a[(by + i) * w + bx + j] - b[(by + i) * w + bx + j];
                }
            }
            for i in 0..4 {
                let (a0, a1, a2, a3) = (d[i * 4], d[i * 4 + 1], d[i * 4 + 2], d[i * 4 + 3]);
                let (s0, s1, d0, d1) = (a0 + a1, a2 + a3, a0 - a1, a2 - a3);
                d[i * 4] = s0 + s1;
                d[i * 4 + 1] = s0 - s1;
                d[i * 4 + 2] = d0 + d1;
                d[i * 4 + 3] = d0 - d1;
            }
            for j in 0..4 {
                let (a0, a1, a2, a3) = (d[j], d[4 + j], d[8 + j], d[12 + j]);
                let (s0, s1, d0, d1) = (a0 + a1, a2 + a3, a0 - a1, a2 - a3);
                total +=
                    ((s0 + s1).abs() + (s0 - s1).abs() + (d0 + d1).abs() + (d0 - d1).abs()) as u64;
            }
        }
    }
    total / 2
}

/// Approximate bits of a motion vector difference component.
fn mv_bits(d: i32) -> f64 {
    if d == 0 {
        1.0
    } else {
        3.0 + 2.0 * ((d.unsigned_abs() as f64 / 2.0) + 1.0).log2()
    }
}

/// The luma intra modes the encoder tries, with a rough cost in bits for
/// the greedy (non-searching) decision.
const Y_MODES: [(usize, f64); 13] = [
    (DC_PRED, 1.0),
    (V_PRED, 2.5),
    (H_PRED, 2.5),
    (SMOOTH_PRED, 2.5),
    (PAETH_PRED, 3.0),
    (D45_PRED, 4.0),
    (D135_PRED, 4.0),
    (SMOOTH_V_PRED, 4.0),
    (SMOOTH_H_PRED, 4.0),
    (D67_PRED, 4.5),
    (D113_PRED, 4.5),
    (D157_PRED, 4.5),
    (D203_PRED, 4.5),
];

impl TileDecoder<'_, '_> {
    fn enc(&self) -> &EncCtx {
        self.enc.as_ref().expect("encode mode")
    }

    /// The level the encoder chose at `pos` (0 when decoding).
    #[inline]
    pub(crate) fn enc_level(&self, pos: usize) -> u32 {
        self.enc.as_ref().map_or(0, |e| e.coefs[pos].unsigned_abs())
    }

    /// The signed level the encoder chose at `pos` (0 when decoding).
    #[inline]
    pub(crate) fn enc_coef(&self, pos: usize) -> i32 {
        self.enc.as_ref().map_or(0, |e| e.coefs[pos])
    }

    fn qstep(&self) -> f64 {
        let bdi = ((self.f.bit_depth - 8) >> 1) as usize;
        AC_QLOOKUP[bdi][self.f.hdr.base_q_idx as usize] as f64
            / (1 << (self.f.bit_depth - 8)) as f64
    }

    /// The partition of a block (encode mode), before the syntax codes it.
    pub(crate) fn enc_partition(
        &mut self,
        r: usize,
        c: usize,
        b_size: usize,
        has_rows: bool,
        has_cols: bool,
    ) -> Result<usize> {
        if self.enc().tools.rdo {
            return self.enc_partition_rd(r, c, b_size, has_rows, has_cols);
        }
        if !(has_rows && has_cols) {
            return Ok(PARTITION_SPLIT);
        }
        Ok(self.enc_partition_greedy(r, c, b_size))
    }

    /// The partition of a block whose four quadrants are all inside the
    /// frame, without search: none or split (split when the source has
    /// more detail than the quantiser keeps).
    fn enc_partition_greedy(&mut self, r: usize, c: usize, b_size: usize) -> usize {
        if b_size > BLOCK_32X32 {
            return PARTITION_SPLIT;
        }
        if b_size <= BLOCK_8X8 {
            return PARTITION_NONE;
        }
        let n = block_width(b_size);
        let (x0, y0) = (c * MI_SIZE, r * MI_SIZE);
        let e = self.enc();
        let var = |x: usize, y: usize, s: usize| -> f64 {
            let mut sum = 0f64;
            let mut sq = 0f64;
            for i in 0..s {
                for j in 0..s {
                    let v = e.src(0, x + j, y + i) as f64;
                    sum += v;
                    sq += v * v;
                }
            }
            let a = (s * s) as f64;
            sq / a - (sum / a) * (sum / a)
        };
        let whole = var(x0, y0, n);
        let h = n / 2;
        let quads =
            (var(x0, y0, h) + var(x0 + h, y0, h) + var(x0, y0 + h, h) + var(x0 + h, y0 + h, h))
                / 4.0;
        let q = self.qstep();
        if whole > q * q * 0.6 && (quads < whole * 0.75 || whole > q * q * 4.0) {
            PARTITION_SPLIT
        } else {
            PARTITION_NONE
        }
    }

    /// The modes of the block about to be coded (encode mode): sets `plan`.
    pub(crate) fn enc_decide_block(&mut self) -> Result<()> {
        if self.enc().tools.rdo {
            return self.enc_decide_block_rd();
        }
        self.enc_decide_block_greedy();
        Ok(())
    }

    /// The prediction arguments `transform_block()` will use for the first
    /// transform block of `plane` at the block's largest transform size:
    /// `(x, y, haveLeft, haveAbove, haveAboveRight, haveBelowLeft, log2W,
    /// log2H, txSz)`.
    #[allow(clippy::type_complexity)]
    fn intra_args(
        &self,
        plane: usize,
    ) -> (usize, usize, bool, bool, bool, bool, usize, usize, usize) {
        let (sub_x, sub_y) = self.f.plane_ss(plane);
        let tx_sz = if self.b.lossless {
            TX_4X4
        } else if plane == 0 {
            MAX_TX_SIZE_RECT[self.b.mi_size]
        } else {
            let uv = MAX_TX_SIZE_RECT[self.f.plane_residual_size(self.b.mi_size, plane)];
            match (TX_WIDTH[uv], TX_HEIGHT[uv]) {
                (64, _) | (_, 64) => TX_32X32,
                _ => uv,
            }
        };
        let x = (self.b.mi_col >> sub_x) * MI_SIZE;
        let y = (self.b.mi_row >> sub_y) * MI_SIZE;
        let sb_mask = if self.f.seq.use_128x128_superblock {
            31
        } else {
            15
        };
        let row = (y << sub_y) >> MI_SIZE_LOG2;
        let col = (x << sub_x) >> MI_SIZE_LOG2;
        let sbr = (row & sb_mask) >> sub_y;
        let sbc = (col & sb_mask) >> sub_x;
        let step_x = TX_WIDTH[tx_sz] >> 2;
        let step_y = TX_HEIGHT[tx_sz] >> 2;
        let (al, au) = if plane == 0 {
            (self.b.avail_l, self.b.avail_u)
        } else {
            (self.b.avail_l_chroma, self.b.avail_u_chroma)
        };
        let ar = self.block_decoded(plane, sbr as isize - 1, (sbc + step_x) as isize);
        let bl = self.block_decoded(plane, (sbr + step_y) as isize, sbc as isize - 1);
        (
            x,
            y,
            al,
            au,
            ar,
            bl,
            TX_WIDTH_LOG2[tx_sz],
            TX_HEIGHT_LOG2[tx_sz],
            tx_sz,
        )
    }

    /// SATD between the source and `CurrFrame` over a region.
    fn region_satd(&self, plane: usize, x: usize, y: usize, w: usize, h: usize) -> u64 {
        let e = self.enc();
        let cur = &self.f.cur.planes[plane];
        let mut a = vec![0i32; w * h];
        let mut b = vec![0i32; w * h];
        for i in 0..h {
            for j in 0..w {
                a[i * w + j] = e.src(plane, x + j, y + i);
                b[i * w + j] = cur.get(x + j, y + i) as i32;
            }
        }
        satd(&a, &b, w.max(4), h.max(4))
    }

    /// Predicts `plane` with intra `mode` as `transform_block()` would and
    /// returns the SATD against the source.
    fn try_intra(&mut self, plane: usize, mode: usize) -> u64 {
        let (x, y, al, au, ar, bl, lw, lh, _) = self.intra_args(plane);
        self.predict_intra(plane, x, y, al, au, ar, bl, mode, lw, lh);
        self.region_satd(plane, x, y, 1 << lw, 1 << lh)
    }

    /// The cost in bits of coding luma intra mode `m` for this block, from
    /// the CDFs in force.
    fn y_mode_bits(&self, m: usize) -> f64 {
        let cost = if self.f.hdr.frame_is_intra {
            let above = if self.b.avail_u {
                INTRA_MODE_CONTEXT[self.mi(self.b.mi_row - 1, self.b.mi_col).y_mode as usize]
            } else {
                0
            };
            let left = if self.b.avail_l {
                INTRA_MODE_CONTEXT[self.mi(self.b.mi_row, self.b.mi_col - 1).y_mode as usize]
            } else {
                0
            };
            Coder::cost(&self.intra_frame_y_mode_cdf[above][left], m)
        } else {
            Coder::cost(&self.cdf.y_mode[SIZE_GROUP[self.b.mi_size]], m)
        };
        cost as f64 / 256.0
    }

    /// The candidates the rate-distortion search trials for this block:
    /// the best intra modes by SATD (plus their rate), then the best inter
    /// modes and vectors per reference.
    pub(crate) fn block_candidates(&mut self) -> Vec<Plan> {
        let lambda = self.enc().lambda;
        let tools = self.enc().tools;
        let ms = self.b.mi_size;
        self.b.use_filter_intra = false;
        self.b.angle_delta_y = 0;
        self.b.angle_delta_uv = 0;
        self.b.is_inter = false;
        self.b.ref_frame = [INTRA_FRAME, NONE];
        let mut ys: Vec<(usize, f64)> = Vec::with_capacity(Y_MODES.len());
        for &(m, _) in Y_MODES.iter() {
            self.b.y_mode = m;
            let cost = self.try_intra(0, m) as f64 + lambda * self.y_mode_bits(m);
            ys.push((m, cost));
        }
        ys.sort_by(|a, b| a.1.total_cmp(&b.1));
        let mut best_uv = DC_PRED;
        if self.b.has_chroma {
            self.b.y_mode = ys[0].0;
            let mut best = f64::MAX;
            for &m in &[DC_PRED, V_PRED, H_PRED, SMOOTH_PRED, PAETH_PRED] {
                self.b.uv_mode = m;
                let cost = (self.try_intra(1, m) + self.try_intra(2, m)) as f64
                    + lambda * if m == DC_PRED { 1.0 } else { 3.0 };
                if cost < best {
                    best = cost;
                    best_uv = m;
                }
            }
        }
        let mut out = Vec::new();
        for &(m, _) in ys.iter().take(tools.intra_candidates.max(1) as usize) {
            out.push(Plan {
                y_mode: m,
                uv_mode: best_uv,
                ref_frame: INTRA_FRAME,
                ..Default::default()
            });
        }
        let refs = self.enc().refs.clone();
        if refs.is_empty() || ms < BLOCK_8X8 {
            return out;
        }
        let mut inter: Vec<(Plan, f64)> = Vec::new();
        for &rf in &refs {
            self.b.ref_frame = [rf, NONE];
            self.b.is_inter = true;
            self.b.use_intrabc = false;
            self.find_mv_stack(false);
            let n = self.b.num_mv_found;
            let mut cands: Vec<(usize, usize, Mv, f64)> = Vec::new();
            cands.push((NEARESTMV, 0, self.b.ref_stack_mv[0][0], 2.0));
            for idx in 1..n.min(3) {
                cands.push((NEARMV, idx, self.b.ref_stack_mv[idx][0], 3.0 + idx as f64));
            }
            cands.push((GLOBALMV, 0, self.b.global_mvs[0], 2.0));
            let pred_mv = self.b.ref_stack_mv[0][0];
            let searched = self.motion_search(rf, pred_mv);
            let bits = 3.0 + mv_bits(searched[0] - pred_mv[0]) + mv_bits(searched[1] - pred_mv[1]);
            cands.push((NEWMV, 0, searched, bits));
            for (mode, idx, mv, bits) in cands {
                if inter
                    .iter()
                    .any(|(p, _)| p.ref_frame == rf && p.mv[0] == mv && p.y_mode != NEWMV)
                    && mode != NEWMV
                {
                    continue;
                }
                let cost = self.inter_satd(rf, mv) as f64 + lambda * bits;
                inter.push((
                    Plan {
                        is_inter: true,
                        y_mode: mode,
                        ref_frame: rf,
                        ref_mv_idx: idx,
                        mv: [mv, [0, 0]],
                        ..Default::default()
                    },
                    cost,
                ));
            }
        }
        self.b.is_inter = false;
        self.b.ref_frame = [INTRA_FRAME, NONE];
        inter.sort_by(|a, b| a.1.total_cmp(&b.1));
        // Intra is worth a trial only when its SATD is near the best inter's.
        let best_inter = inter.first().map_or(f64::MAX, |x| x.1);
        let intra_keep = out
            .into_iter()
            .zip(ys.iter())
            .filter(|(_, y)| y.1 < best_inter * 1.5)
            .map(|(p, _)| p);
        let mut all: Vec<Plan> = inter
            .iter()
            .take(tools.inter_candidates.max(1) as usize)
            .map(|x| x.0)
            .collect();
        all.extend(intra_keep);
        all
    }

    /// Chooses the block's modes without search (`plan`), before
    /// `mode_info()` codes them.
    fn enc_decide_block_greedy(&mut self) {
        let lambda = self.enc().lambda;
        let ms = self.b.mi_size;
        self.plan = Default::default();
        self.b.use_filter_intra = false;
        self.b.angle_delta_y = 0;
        self.b.angle_delta_uv = 0;
        let mut best_y = (DC_PRED, f64::MAX);
        for &(m, bits) in Y_MODES.iter().take(10) {
            self.b.y_mode = m;
            let cost = self.try_intra(0, m) as f64 + lambda * bits;
            if cost < best_y.1 {
                best_y = (m, cost);
            }
        }
        self.b.y_mode = best_y.0;
        let mut best_uv = DC_PRED;
        if self.b.has_chroma {
            let mut best = f64::MAX;
            for &m in &[DC_PRED, V_PRED, H_PRED, SMOOTH_PRED, PAETH_PRED] {
                self.b.uv_mode = m;
                let cost = (self.try_intra(1, m) + self.try_intra(2, m)) as f64
                    + lambda * if m == DC_PRED { 1.0 } else { 3.0 };
                if cost < best {
                    best = cost;
                    best_uv = m;
                }
            }
        }
        self.plan.y_mode = best_y.0;
        self.plan.uv_mode = best_uv;
        self.plan.tx_type = DCT_DCT;
        // Inter: the best of NEWMV (searched), NEARESTMV and GLOBALMV on
        // LAST_FRAME, for blocks of 8x8 and up.
        let mut inter_cost = f64::MAX;
        if !self.enc().refs.is_empty() && ms >= BLOCK_8X8 {
            self.b.ref_frame = [LAST_FRAME, NONE];
            self.b.is_inter = true;
            self.b.use_intrabc = false;
            self.find_mv_stack(false);
            let pred_mv = self.b.ref_stack_mv[0][0];
            let nearest = self.b.ref_stack_mv[0][0];
            let global = self.b.global_mvs[0];
            let mut cands: Vec<(usize, Mv, f64)> =
                vec![(NEARESTMV, nearest, 2.0), (GLOBALMV, global, 2.0)];
            let searched = self.motion_search(LAST_FRAME, pred_mv);
            let bits = 3.0 + mv_bits(searched[0] - pred_mv[0]) + mv_bits(searched[1] - pred_mv[1]);
            cands.push((NEWMV, searched, bits));
            let mut best = (GLOBALMV, global, f64::MAX);
            for (mode, mv, bits) in cands {
                let cost = self.inter_satd(LAST_FRAME, mv) as f64 + lambda * bits;
                if cost < best.2 {
                    best = (mode, mv, cost);
                }
            }
            inter_cost = best.2;
            self.plan.y_mode = best.0;
            self.plan.mv[0] = best.1;
            self.plan.ref_frame = LAST_FRAME;
            self.plan.ref_mv_idx = 0;
            self.b.is_inter = false;
            self.b.ref_frame = [INTRA_FRAME, NONE];
        }
        if inter_cost < best_y.1 + lambda * 2.0 {
            self.plan.is_inter = true;
            self.plan.skip = self.inter_residual_is_zero(self.plan.mv[0]);
        } else {
            self.plan.is_inter = false;
            self.plan.y_mode = best_y.0;
            self.plan.skip = self.intra_residual_is_zero(best_y.0, best_uv);
        }
    }

    /// Whether every transform block of the block, predicted with these
    /// intra modes, quantises to zero.
    fn intra_residual_is_zero(&mut self, y_mode: usize, uv_mode: usize) -> bool {
        let planes = if self.b.has_chroma { 3 } else { 1 };
        for plane in 0..planes {
            let mode = if plane == 0 { y_mode } else { uv_mode };
            if plane == 0 {
                self.b.y_mode = y_mode;
            } else {
                self.b.uv_mode = uv_mode;
            }
            let (x, y, al, au, ar, bl, lw, lh, tx_sz) = self.intra_args(plane);
            self.predict_intra(plane, x, y, al, au, ar, bl, mode, lw, lh);
            let eob = if plane == 0 {
                self.choose_luma_tx_type(x, y, tx_sz)
            } else {
                let t = self.chroma_intra_tx_type(tx_sz, uv_mode);
                self.quantise_block(plane, x, y, tx_sz, t)
            };
            if eob > 0 {
                return false;
            }
        }
        true
    }

    /// Chooses the luma transform type of an intra block (its prediction in
    /// place) among the DCT and ADST combinations the reduced intra set
    /// allows, by distortion plus estimated rate on the quantised
    /// reconstruction; sets `plan.tx_type` and returns the end of block.
    fn choose_luma_tx_type(&mut self, x: usize, y: usize, tx_sz: usize) -> usize {
        let choice =
            TX_SIZE_SQR_UP[tx_sz] <= TX_16X16 && self.f.hdr.base_q_idx > 0 && !self.b.lossless;
        if !choice {
            self.plan.tx_type = DCT_DCT;
            return self.quantise_block(0, x, y, tx_sz, DCT_DCT);
        }
        let mut best = (DCT_DCT, f64::MAX);
        for t in [DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST] {
            let cost = self.trial_cost(x, y, tx_sz, t);
            if cost < best.1 {
                best = (t, cost);
            }
        }
        self.plan.tx_type = best.0;
        self.quantise_block(0, x, y, tx_sz, best.0)
    }

    /// Distortion plus lambda times estimated bits of coding the luma
    /// residual at (x, y) with transform type `t`.
    fn trial_cost(&mut self, x: usize, y: usize, tx_sz: usize, t: usize) -> f64 {
        let eob = self.quantise_block(0, x, y, tx_sz, t);
        let w = TX_WIDTH[tx_sz];
        let h = TX_HEIGHT[tx_sz];
        let tw = w.min(32);
        let th = h.min(32);
        let bd = self.f.bit_depth;
        let bdi = ((bd - 8) >> 1) as usize;
        let q = self.f.hdr.base_q_idx as i32;
        let dc_q = DC_QLOOKUP[bdi][clip3(0, 255, q + self.f.hdr.delta_q_y_dc) as usize] as i64;
        let ac_q = AC_QLOOKUP[bdi][q as usize] as i64;
        let denom: i64 = match tx_sz {
            TX_32X32 | TX_16X32 | TX_32X16 => 2,
            _ => 1,
        };
        let enc = self.enc.as_ref().expect("encode mode");
        let mut dq = vec![0i32; 64 * 64];
        let mut bits = 0f64;
        for i in 0..th {
            for j in 0..tw {
                let l = enc.coefs[i * tw + j] as i64;
                if l != 0 {
                    bits += 2.0 + 2.0 * ((l.unsigned_abs() + 1) as f64).log2();
                    let qq = if i == 0 && j == 0 { dc_q } else { ac_q };
                    dq[i * 64 + j] = (l * qq / denom) as i32;
                }
            }
        }
        let mut rec = vec![0i32; w * h];
        if eob > 0 {
            crate::dsp::itx::inverse_transform_2d(&dq, tx_sz, t, false, bd, &mut rec);
        }
        let cur = &self.f.cur.planes[0];
        let mut sse = 0f64;
        for i in 0..h {
            for j in 0..w {
                let res = enc.src(0, x + j, y + i) - cur.get(x + j, y + i) as i32;
                let d = (res - rec[i * w + j]) as f64;
                sse += d * d;
            }
        }
        let step = ac_q as f64 / (8 << (bd - 8)) as f64;
        sse + 0.3 * step * step * (1 << (2 * (bd - 8))) as f64 * bits
    }

    /// The transform type the decoder derives for an intra chroma block.
    fn chroma_intra_tx_type(&self, tx_sz: usize, uv_mode: usize) -> usize {
        if TX_SIZE_SQR_UP[tx_sz] > TX_32X32 || self.b.lossless {
            return DCT_DCT;
        }
        let t = MODE_TO_TXFM[uv_mode];
        let set = if TX_SIZE_SQR_UP[tx_sz] == TX_32X32 {
            0
        } else if self.f.hdr.reduced_tx_set || TX_SIZE_SQR[tx_sz] == TX_16X16 {
            2
        } else {
            1
        };
        if TX_TYPE_IN_SET_INTRA[set][t] != 0 {
            t
        } else {
            DCT_DCT
        }
    }

    /// Whether the block, predicted from LAST_FRAME with `mv`, has an
    /// all-zero residual after quantisation.
    fn inter_residual_is_zero(&mut self, mv: Mv) -> bool {
        let planes = if self.b.has_chroma { 3 } else { 1 };
        for plane in 0..planes {
            let (sub_x, sub_y) = self.f.plane_ss(plane);
            let x = (self.b.mi_col >> sub_x) * MI_SIZE;
            let y = (self.b.mi_row >> sub_y) * MI_SIZE;
            let w = block_width(self.b.mi_size) >> sub_x;
            let h = block_height(self.b.mi_size) >> sub_y;
            let pred = self.inter_pred(plane, x, y, w, h, LAST_FRAME, mv);
            let maxv = (1i32 << self.f.bit_depth) - 1;
            let cur = &mut self.f.cur.planes[plane];
            for i in 0..h {
                for j in 0..w {
                    cur.set(x + j, y + i, pred[i * w + j].clamp(0, maxv) as u16);
                }
            }
            let tx_sz = if plane == 0 {
                MAX_TX_SIZE_RECT[self.b.mi_size]
            } else {
                MAX_TX_SIZE_RECT[self.f.plane_residual_size(self.b.mi_size, plane)]
            };
            if self.quantise_block(plane, x, y, tx_sz, DCT_DCT) > 0 {
                return false;
            }
        }
        true
    }

    /// The single-reference prediction from `ref_frame`, unclipped.
    #[allow(clippy::too_many_arguments)]
    fn inter_pred(
        &self,
        plane: usize,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        ref_frame: i32,
        mv: Mv,
    ) -> Vec<i32> {
        let ref_idx = self.f.hdr.ref_frame_idx[(ref_frame - LAST_FRAME) as usize];
        let r = self.f.refs[ref_idx].as_ref().expect("a reference");
        let (sx, sy, stx, sty) = self.f.scale_mv(
            plane,
            r.upscaled_width as i32,
            r.frame_height as i32,
            x,
            y,
            mv,
        );
        let (ssx, ssy) = self.f.plane_ss(plane);
        let last_x = ((r.upscaled_width as i32 + ssx as i32) >> ssx) - 1;
        let last_y = ((r.frame_height as i32 + ssy as i32) >> ssy) - 1;
        let rv = self.f.rounding_variables(false);
        let mut pred = vec![0i32; w * h];
        let filt = self.f.hdr.interpolation_filter.min(BILINEAR) as u8;
        block_inter_prediction(
            &r.frame.planes[plane],
            last_x,
            last_y,
            sx,
            sy,
            stx,
            sty,
            w,
            h,
            [filt, filt],
            rv,
            &mut pred,
        );
        pred
    }

    /// Luma SATD of the inter prediction from `ref_frame` with `mv`.
    fn inter_satd(&self, ref_frame: i32, mv: Mv) -> u64 {
        let w = block_width(self.b.mi_size);
        let h = block_height(self.b.mi_size);
        let x = self.b.mi_col * MI_SIZE;
        let y = self.b.mi_row * MI_SIZE;
        let pred = self.inter_pred(0, x, y, w, h, ref_frame, mv);
        let e = self.enc();
        let mut src = vec![0i32; w * h];
        for i in 0..h {
            for j in 0..w {
                src[i * w + j] = e.src(0, x + j, y + i);
            }
        }
        satd(&src, &pred, w, h)
    }

    /// A full-pel search around the predicted and zero vectors, then
    /// half- and quarter-sample refinement with the normative predictor.
    fn motion_search(&self, ref_frame: i32, pred_mv: Mv) -> Mv {
        let w = block_width(self.b.mi_size);
        let h = block_height(self.b.mi_size);
        let x0 = (self.b.mi_col * MI_SIZE) as i32;
        let y0 = (self.b.mi_row * MI_SIZE) as i32;
        let ref_idx = self.f.hdr.ref_frame_idx[(ref_frame - LAST_FRAME) as usize];
        let r = self.f.refs[ref_idx].as_ref().expect("a reference");
        let rp = &r.frame.planes[0];
        let last_x = r.upscaled_width as i32 - 1;
        let last_y = r.frame_height as i32 - 1;
        let e = self.enc();
        let sad = |dx: i32, dy: i32, best: u64| -> u64 {
            let mut s = 0u64;
            let inside = x0 + dx >= 0 && x0 + dx + w as i32 - 1 <= last_x;
            for i in 0..h as i32 {
                let ry = (y0 + i + dy).clamp(0, last_y) as usize;
                let row = rp.row(ry);
                let so = (y0 + i) as usize * e.stride[0] + x0 as usize;
                let srow = &e.src[0][so..so + w];
                if inside {
                    let r0 = (x0 + dx) as usize;
                    let mut acc = 0u32;
                    for (&a, &b) in srow.iter().zip(&row[r0..r0 + w]) {
                        acc += (a as i32 - b as i32).unsigned_abs();
                    }
                    s += acc as u64;
                } else {
                    for j in 0..w as i32 {
                        let rx = (x0 + j + dx).clamp(0, last_x) as usize;
                        s += (srow[j as usize] as i32 - row[rx] as i32).unsigned_abs() as u64;
                    }
                }
                if s >= best {
                    return s;
                }
            }
            s
        };
        let lambda = e.lambda;
        let cost = |dx: i32, dy: i32, best: f64| -> f64 {
            let bits = mv_bits(dy * 8 - pred_mv[0]) + mv_bits(dx * 8 - pred_mv[1]);
            sad(dx, dy, best.min(1e18) as u64) as f64 + lambda * bits
        };
        let range = e.search_range;
        let mut best = (0i32, 0i32, cost(0, 0, f64::MAX));
        let (pcx, pcy) = ((pred_mv[1] + 4) >> 3, (pred_mv[0] + 4) >> 3);
        let c = cost(pcx, pcy, best.2);
        if c < best.2 {
            best = (pcx, pcy, c);
        }
        let (cx, cy) = (best.0, best.1);
        // A coarse square search (step 2), then a fine one around the best.
        let mut step_best = best;
        for dy in (-range..=range).step_by(2) {
            for dx in (-range..=range).step_by(2) {
                let c = cost(cx + dx, cy + dy, step_best.2);
                if c < step_best.2 {
                    step_best = (cx + dx, cy + dy, c);
                }
            }
        }
        best = step_best;
        let (cx, cy) = (best.0, best.1);
        for dy in -2..=2 {
            for dx in -2..=2 {
                let c = cost(cx + dx, cy + dy, best.2);
                if c < best.2 {
                    best = (cx + dx, cy + dy, c);
                }
            }
        }
        let mut mv: Mv = [best.1 * 8, best.0 * 8];
        if self.f.hdr.force_integer_mv {
            return mv;
        }
        // Sub-sample refinement, in eighth samples: 4 then 2 (quarter).
        let mut best_cost = self.inter_satd(ref_frame, mv) as f64
            + lambda * (mv_bits(mv[0] - pred_mv[0]) + mv_bits(mv[1] - pred_mv[1]));
        let min_step = if self.f.hdr.allow_high_precision_mv {
            1
        } else {
            2
        };
        let mut step = 4;
        while step >= min_step {
            let center = mv;
            for (dy, dx) in [
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ] {
                let cand = [center[0] + dy * step, center[1] + dx * step];
                let c = self.inter_satd(ref_frame, cand) as f64
                    + lambda * (mv_bits(cand[0] - pred_mv[0]) + mv_bits(cand[1] - pred_mv[1]));
                if c < best_cost {
                    best_cost = c;
                    mv = cand;
                }
            }
            step /= 2;
        }
        mv
    }

    /// Forward transforms and quantises the residual (source minus
    /// `CurrFrame`) of one transform block into `EncCtx::coefs`; returns the
    /// end of block in the scan of `tx_type`.
    fn quantise_block(
        &mut self,
        plane: usize,
        x: usize,
        y: usize,
        tx_sz: usize,
        tx_type: usize,
    ) -> usize {
        let w = TX_WIDTH[tx_sz];
        let h = TX_HEIGHT[tx_sz];
        let tw = w.min(32);
        let th = h.min(32);
        let mut res = std::mem::take(&mut self.enc.as_mut().expect("encode mode").res);
        let mut c = std::mem::take(&mut self.enc.as_mut().expect("encode mode").fc);
        res.resize(64 * 64, 0);
        c.resize(64 * 64, 0.0);
        {
            let e = self.enc();
            let cur = &self.f.cur.planes[plane];
            for i in 0..h {
                let so = (y + i) * e.stride[plane] + x;
                let co = (y + i) * cur.stride + x;
                let srow = &e.src[plane][so..so + w];
                let crow = &cur.data[co..co + w];
                for (r, (&a, &b)) in res[i * w..(i + 1) * w]
                    .iter_mut()
                    .zip(srow.iter().zip(crow))
                {
                    *r = a as i32 - b as i32;
                }
            }
        }
        forward_2d(&res[..w * h], tx_sz, tx_type, &mut c[..w * h]);
        let bd = self.f.bit_depth;
        let bdi = ((bd - 8) >> 1) as usize;
        let (dc_delta, ac_delta) = match plane {
            0 => (self.f.hdr.delta_q_y_dc, 0),
            1 => (self.f.hdr.delta_q_u_dc, self.f.hdr.delta_q_u_ac),
            _ => (self.f.hdr.delta_q_v_dc, self.f.hdr.delta_q_v_ac),
        };
        let q = self.qindex();
        let dc_q = DC_QLOOKUP[bdi][clip3(0, 255, q + dc_delta) as usize] as f64;
        let ac_q = AC_QLOOKUP[bdi][clip3(0, 255, q + ac_delta) as usize] as f64;
        let denom = match tx_sz {
            TX_32X32 | TX_16X32 | TX_32X16 | TX_16X64 | TX_64X16 => 2.0,
            TX_64X64 | TX_32X64 | TX_64X32 => 4.0,
            _ => 1.0,
        };
        let enc = self.enc.as_mut().expect("encode mode");
        enc.coefs.fill(0);
        for i in 0..th {
            for j in 0..tw {
                let v = c[i * w + j];
                let (qs, dz) = if i == 0 && j == 0 {
                    (dc_q, 0.5)
                } else {
                    (ac_q, 0.62)
                };
                let a = v.abs() * denom / qs;
                let l = (a + 1.0 - dz).floor().max(0.0).min(((1 << 20) - 1) as f64) as i32;
                enc.coefs[i * tw + j] = if v < 0.0 { -l } else { l };
            }
        }
        enc.res = res;
        enc.fc = c;
        // The end of block in the scan the decoder will use.
        self.plane_tx_type = tx_type;
        let scan = crate::decoder::residual::scan_for(tx_sz, tx_type);
        let enc = self.enc.as_ref().expect("encode mode");
        let mut eob = 0;
        for (k, &p) in scan.iter().enumerate() {
            if enc.coefs[p as usize] != 0 {
                eob = k + 1;
            }
        }
        eob
    }

    /// Squared error between the source and the reconstruction over a
    /// sample rectangle of `plane`, inside the frame.
    fn px_sse(&self, plane: usize, x0: usize, y0: usize, w: usize, h: usize) -> u64 {
        let e = self.enc();
        let (ssx, ssy) = self.f.plane_ss(plane);
        let fw = (self.f.hdr.frame_width as usize + ssx) >> ssx;
        let fh = (self.f.hdr.frame_height as usize + ssy) >> ssy;
        let x1 = (x0 + w).min(fw);
        let y1 = (y0 + h).min(fh);
        let pl = &self.f.cur.planes[plane];
        let st = e.stride[plane];
        let mut s = 0u64;
        for y in y0..y1 {
            for x in x0..x1 {
                let d = e.src[plane][y * st + x] as i64 - pl.data[y * pl.stride + x] as i64;
                s += (d * d) as u64;
            }
        }
        s
    }

    /// The luma transform types worth a trial for this transform block.
    fn tx_type_candidates(&self, tx_sz: usize) -> Vec<usize> {
        if self.b.lossless || TX_SIZE_SQR_UP[tx_sz] > TX_32X32 || self.qindex() == 0 {
            return vec![DCT_DCT];
        }
        let set = self.get_tx_set(tx_sz);
        if set == 0 {
            return vec![DCT_DCT];
        }
        const ORDER: [usize; 16] = [
            DCT_DCT,
            ADST_ADST,
            ADST_DCT,
            DCT_ADST,
            IDTX,
            V_DCT,
            H_DCT,
            FLIPADST_DCT,
            DCT_FLIPADST,
            FLIPADST_FLIPADST,
            ADST_FLIPADST,
            FLIPADST_ADST,
            V_ADST,
            H_ADST,
            V_FLIPADST,
            H_FLIPADST,
        ];
        let max = self.enc().tools.tx_types as usize;
        ORDER
            .iter()
            .copied()
            .filter(|&t| self.is_tx_type_in_set(set, t))
            .take(max.max(1))
            .collect()
    }

    /// Chooses the levels of the transform block about to be coded (its
    /// prediction is in `CurrFrame`) and returns its end of block. The
    /// luma transform type is searched by trial coding when there is a
    /// choice.
    pub(crate) fn enc_choose_coeffs(
        &mut self,
        plane: usize,
        start_x: usize,
        start_y: usize,
        tx_sz: usize,
    ) -> usize {
        if plane == 0
            && let Some(t) = self.rdo().forced_tx_type.take()
        {
            self.enc_tx_type = t;
            return self.quantise_block(0, start_x, start_y, tx_sz, t);
        }
        let derived = self.b.lossless || TX_SIZE_SQR_UP[tx_sz] > TX_32X32;
        let t = if derived {
            DCT_DCT
        } else if plane == 0 {
            if self.enc().tools.tx_type_rd {
                // Replayed, or searched (and logged, for replay).
                let rdo = self.rdo();
                if let Some(Decision::TxType(t)) = rdo.replay.front().copied() {
                    rdo.replay.pop_front();
                    rdo.log.push(Decision::TxType(t));
                    t as usize
                } else {
                    let cands = if self.rdo().fast_tx {
                        vec![DCT_DCT]
                    } else {
                        self.tx_type_candidates(tx_sz)
                    };
                    let t = if cands.len() > 1 {
                        self.search_tx_type(start_x, start_y, tx_sz, &cands)
                    } else {
                        cands[0]
                    };
                    self.rdo().log.push(Decision::TxType(t as u8));
                    t
                }
            } else {
                self.plan.tx_type
            }
        } else {
            self.compute_tx_type(plane, tx_sz, start_x >> 2, start_y >> 2)
        };
        if plane == 0 {
            self.enc_tx_type = t;
        }
        self.quantise_block(plane, start_x, start_y, tx_sz, t)
    }

    /// Codes the luma transform block at `(x, y)` with each candidate type
    /// in counting mode and returns the cheapest.
    fn search_tx_type(&mut self, x: usize, y: usize, tx_sz: usize, cands: &[usize]) -> usize {
        let w = TX_WIDTH[tx_sz];
        let h = TX_HEIGHT[tx_sz];
        let (x4, y4, w4, h4) = (x >> 2, y >> 2, w >> 2, h >> 2);
        let ms = self.f.ms;
        // What coding the block changes: its samples, its contexts, its
        // transform types.
        let pl = &self.f.cur.planes[0];
        let x1 = (x + w).min(pl.w);
        let y1 = (y + h).min(pl.h);
        let mut pix = Vec::with_capacity(w * h);
        for yy in y..y1 {
            pix.extend_from_slice(&pl.data[yy * pl.stride + x..yy * pl.stride + x1]);
        }
        let al: Vec<u8> = self.above_level_ctx[0][x4..x4 + w4].to_vec();
        let ad: Vec<u8> = self.above_dc_ctx[0][x4..x4 + w4].to_vec();
        let ll: Vec<u8> = self.left_level_ctx[0][y4..y4 + h4].to_vec();
        let ld: Vec<u8> = self.left_dc_ctx[0][y4..y4 + h4].to_vec();
        let rows = (y4 + h4).min(self.f.mi_rows + 32);
        let cols = (x4 + w4).min(ms);
        let mut txt = Vec::new();
        for r in y4..rows {
            txt.extend_from_slice(&self.f.tx_types[r * ms + x4..r * ms + cols]);
        }
        let restore = |t: &mut Self| {
            let pl = &mut t.f.cur.planes[0];
            let xw = x1 - x;
            for (i, yy) in (y..y1).enumerate() {
                pl.data[yy * pl.stride + x..yy * pl.stride + x1]
                    .copy_from_slice(&pix[i * xw..(i + 1) * xw]);
            }
            t.above_level_ctx[0][x4..x4 + w4].copy_from_slice(&al);
            t.above_dc_ctx[0][x4..x4 + w4].copy_from_slice(&ad);
            t.left_level_ctx[0][y4..y4 + h4].copy_from_slice(&ll);
            t.left_dc_ctx[0][y4..y4 + h4].copy_from_slice(&ld);
            let cw = cols - x4;
            for (i, r) in (y4..rows).enumerate() {
                t.f.tx_types[r * ms + x4..r * ms + cols]
                    .copy_from_slice(&txt[i * cw..(i + 1) * cw]);
            }
        };
        let counting = self.sd.set_counting(true);
        let bits_start = self.sd.trial_bits();
        let nonzero = self.rdo().nonzero;
        let saved_ptt = self.plane_tx_type;
        let mut best = (cands[0], f64::MAX);
        for &t in cands {
            self.rdo().forced_tx_type = Some(t);
            let bits0 = self.sd.trial_bits();
            let eob = self.coeffs(0, x, y, tx_sz);
            if eob > 0 {
                self.reconstruct(0, x, y, tx_sz);
            }
            let bits = self.sd.trial_bits() - bits0;
            let sse = self.px_sse(0, x, y, w, h);
            let cost = self.rd_cost(sse, bits);
            if cost < best.1 {
                best = (t, cost);
            }
            restore(self);
        }
        self.rdo().nonzero = nonzero;
        self.plane_tx_type = saved_ptt;
        self.sd.set_counting(counting);
        self.sd.set_trial_bits(bits_start);
        best.0
    }
}

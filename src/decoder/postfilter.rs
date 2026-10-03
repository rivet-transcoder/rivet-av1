//! The in-loop filters applied when a frame is complete: the loop filter
//! (7.14), CDEF (7.15), super-resolution upscaling (7.16) and loop
//! restoration (7.17).

use crate::bits::floor_log2;
use crate::consts::*;
use crate::decoder::state::{FrameBuf, PlaneBuf};
use crate::decoder::{FrameCtx, count_units_in_frame};
use crate::tables::*;

// ---------------------------------------------------------------------------
// Loop filter (7.14)
// ---------------------------------------------------------------------------

/// The loop filter process (7.14.1), in place on `CurrFrame`.
pub(crate) fn loop_filter(f: &mut FrameCtx) {
    for plane in 0..f.num_planes {
        if plane == 0 || f.hdr.loop_filter_level[1 + plane] != 0 {
            for pass in 0..2 {
                let row_step = if plane == 0 { 1 } else { 1 << f.ssy };
                let col_step = if plane == 0 { 1 } else { 1 << f.ssx };
                let mut row = 0;
                while row < f.mi_rows {
                    let mut col = 0;
                    while col < f.mi_cols {
                        edge_loop_filter(f, plane, pass, row, col);
                        col += col_step;
                    }
                    row += row_step;
                }
            }
        }
    }
}

/// The edge loop filter process (7.14.2).
fn edge_loop_filter(f: &mut FrameCtx, plane: usize, pass: usize, row: usize, col: usize) {
    let (sub_x, sub_y) = f.plane_ss(plane);
    let (dx, dy) = if pass == 0 { (1i32, 0i32) } else { (0, 1) };
    let x = col * MI_SIZE;
    let y = row * MI_SIZE;
    let row = row | sub_y;
    let col = col | sub_x;
    let on_screen = !(x >= f.hdr.frame_width as usize
        || y >= f.hdr.frame_height as usize
        || (pass == 0 && x == 0)
        || (pass == 1 && y == 0));
    if !on_screen {
        return;
    }
    let xp = x >> sub_x;
    let yp = y >> sub_y;
    let prev_row = row - ((dy as usize) << sub_y);
    let prev_col = col - ((dx as usize) << sub_x);
    let cols = f.ms;
    let m = f.mi[row * cols + col];
    let mi_size = m.mi_size as usize;
    let tx_sz = f.lf_tx_sizes[plane][(row >> sub_y) * cols + (col >> sub_x)] as usize;
    let plane_size = f.plane_residual_size(mi_size, plane);
    let skip = m.skip;
    let is_intra = m.ref_frame[0] as i32 <= INTRA_FRAME;
    let prev_tx_sz =
        f.lf_tx_sizes[plane][(prev_row >> sub_y) * cols + (prev_col >> sub_x)] as usize;
    let is_block_edge = if pass == 0 {
        xp.is_multiple_of(block_width(plane_size))
    } else {
        yp.is_multiple_of(block_height(plane_size))
    };
    let is_tx_edge = if pass == 0 {
        xp.is_multiple_of(TX_WIDTH[tx_sz])
    } else {
        yp.is_multiple_of(TX_HEIGHT[tx_sz])
    };
    let apply_filter = is_tx_edge && (is_block_edge || !skip || is_intra);
    // The filter size process (7.14.3).
    let base_size = if pass == 0 {
        TX_WIDTH[prev_tx_sz].min(TX_WIDTH[tx_sz])
    } else {
        TX_HEIGHT[prev_tx_sz].min(TX_HEIGHT[tx_sz])
    };
    let filter_size = if plane == 0 {
        16.min(base_size)
    } else {
        8.min(base_size)
    };
    let (mut lvl, mut limit, mut blimit, mut thresh) = filter_strength(f, row, col, plane, pass);
    if lvl == 0 {
        (lvl, limit, blimit, thresh) = filter_strength(f, prev_row, prev_col, plane, pass);
    }
    if !(apply_filter && lvl > 0) {
        return;
    }
    let bd = f.bit_depth;
    let p = &mut f.cur.planes[plane];
    for i in 0..MI_SIZE as i32 {
        sample_filter(
            p,
            xp as i32 + dy * i,
            yp as i32 + dx * i,
            plane,
            limit,
            blimit,
            thresh,
            dx,
            dy,
            filter_size,
            bd,
        );
    }
}

/// The adaptive filter strength process (7.14.4): `lvl, limit, blimit,
/// thresh`.
fn filter_strength(
    f: &FrameCtx,
    row: usize,
    col: usize,
    plane: usize,
    pass: usize,
) -> (i32, i32, i32, i32) {
    let h = &f.hdr;
    let m = &f.mi[row * f.ms + col];
    let segment = m.segment_id as usize;
    let rf = m.ref_frame[0] as i32;
    let mode = m.y_mode as usize;
    let mode_type = (mode >= NEARESTMV && mode != GLOBALMV && mode != GLOBAL_GLOBALMV) as usize;
    let delta_lf = if h.delta_lf_multi {
        m.delta_lf[if plane == 0 { pass } else { plane + 1 }] as i32
    } else {
        m.delta_lf[0] as i32
    };
    // The adaptive filter strength selection process (7.14.5).
    let i = if plane == 0 { pass } else { plane + 1 };
    let base_filter_level = clip3(0, MAX_LOOP_FILTER, delta_lf + h.loop_filter_level[i]);
    let mut lvl_seg = base_filter_level;
    let feature = SEG_LVL_ALT_LF_Y_V + i;
    if h.segmentation_enabled && h.feature_enabled[segment][feature] {
        lvl_seg = clip3(
            0,
            MAX_LOOP_FILTER,
            h.feature_data[segment][feature] + lvl_seg,
        );
    }
    if h.loop_filter_delta_enabled {
        let n_shift = lvl_seg >> 5;
        if rf == INTRA_FRAME {
            lvl_seg += h.loop_filter_ref_deltas[INTRA_FRAME as usize] << n_shift;
        } else if rf > INTRA_FRAME {
            lvl_seg += (h.loop_filter_ref_deltas[rf as usize] << n_shift)
                + (h.loop_filter_mode_deltas[mode_type] << n_shift);
        }
        lvl_seg = clip3(0, MAX_LOOP_FILTER, lvl_seg);
    }
    let lvl = lvl_seg;
    let sharp = h.loop_filter_sharpness;
    let shift = if sharp > 4 {
        2
    } else if sharp > 0 {
        1
    } else {
        0
    };
    let limit = if sharp > 0 {
        clip3(1, 9 - sharp, lvl >> shift)
    } else {
        (lvl >> shift).max(1)
    };
    let blimit = 2 * (lvl + 2) + limit;
    let thresh = lvl >> 4;
    (lvl, limit, blimit, thresh)
}

/// The sample filtering process (7.14.6).
#[allow(clippy::too_many_arguments)]
fn sample_filter(
    p: &mut PlaneBuf,
    x: i32,
    y: i32,
    plane: usize,
    limit: i32,
    blimit: i32,
    thresh: i32,
    dx: i32,
    dy: i32,
    filter_size: usize,
    bd: u32,
) {
    let stride = p.stride as isize;
    let base = y as isize * stride + x as isize;
    let step = dy as isize * stride + dx as isize;
    let s = |p: &PlaneBuf, k: isize| p.data[(base + k * step) as usize] as i32;
    let q0 = s(p, 0);
    let q1 = s(p, 1);
    let q2 = s(p, 2);
    let q3 = s(p, 3);
    let p0 = s(p, -1);
    let p1 = s(p, -2);
    let p2 = s(p, -3);
    let p3 = s(p, -4);
    // The filter mask process (7.14.6.2).
    let sh = bd - 8;
    let thresh_bd = thresh << sh;
    let hev_mask = (p1 - p0).abs() > thresh_bd || (q1 - q0).abs() > thresh_bd;
    let filter_len = if filter_size == 4 {
        4
    } else if plane != 0 {
        6
    } else if filter_size == 8 {
        8
    } else {
        16
    };
    let limit_bd = limit << sh;
    let blimit_bd = blimit << sh;
    let mut mask = (p1 - p0).abs() > limit_bd
        || (q1 - q0).abs() > limit_bd
        || (p0 - q0).abs() * 2 + (p1 - q1).abs() / 2 > blimit_bd;
    if filter_len >= 6 {
        mask |= (p2 - p1).abs() > limit_bd || (q2 - q1).abs() > limit_bd;
    }
    if filter_len >= 8 {
        mask |= (p3 - p2).abs() > limit_bd || (q3 - q2).abs() > limit_bd;
    }
    if mask {
        return;
    }
    let threshold_bd = 1 << sh;
    let mut flat_mask = false;
    if filter_size >= 8 {
        let mut m = (p1 - p0).abs() > threshold_bd
            || (q1 - q0).abs() > threshold_bd
            || (p2 - p0).abs() > threshold_bd
            || (q2 - q0).abs() > threshold_bd;
        if filter_len >= 8 {
            m |= (p3 - p0).abs() > threshold_bd || (q3 - q0).abs() > threshold_bd;
        }
        flat_mask = !m;
    }
    let mut flat_mask2 = false;
    if filter_size >= 16 {
        let q4 = s(p, 4);
        let q5 = s(p, 5);
        let q6 = s(p, 6);
        let p4 = s(p, -5);
        let p5 = s(p, -6);
        let p6 = s(p, -7);
        let m = (p6 - p0).abs() > threshold_bd
            || (q6 - q0).abs() > threshold_bd
            || (p5 - p0).abs() > threshold_bd
            || (q5 - q0).abs() > threshold_bd
            || (p4 - p0).abs() > threshold_bd
            || (q4 - q0).abs() > threshold_bd;
        flat_mask2 = !m;
    }
    if filter_size == 4 || !flat_mask {
        // The narrow filter process (7.14.6.3).
        let lo = -(1 << (bd - 1));
        let hi = (1 << (bd - 1)) - 1;
        let c = |v: i32| v.clamp(lo, hi);
        let off = 0x80 << sh;
        let ps1 = p1 - off;
        let ps0 = p0 - off;
        let qs0 = q0 - off;
        let qs1 = q1 - off;
        let mut filter = if hev_mask { c(ps1 - qs1) } else { 0 };
        filter = c(filter + 3 * (qs0 - ps0));
        let filter1 = c(filter + 4) >> 3;
        let filter2 = c(filter + 3) >> 3;
        let set =
            |p: &mut PlaneBuf, k: isize, v: i32| p.data[(base + k * step) as usize] = v as u16;
        set(p, 0, c(qs0 - filter1) + off);
        set(p, -1, c(ps0 + filter2) + off);
        if !hev_mask {
            let filter = round2(filter1, 1);
            set(p, 1, c(qs1 - filter) + off);
            set(p, -2, c(ps1 + filter) + off);
        }
    } else {
        // The wide filter process (7.14.6.4).
        let log2_size = if filter_size == 8 || !flat_mask2 {
            3
        } else {
            4
        };
        let n: isize = if log2_size == 4 {
            6
        } else if plane == 0 {
            3
        } else {
            2
        };
        let n2: isize = if log2_size == 3 && plane == 0 { 0 } else { 1 };
        let mut fv = [0i32; 12];
        for i in -n..n {
            let mut t = 0;
            for j in -n..=n {
                let pp = (i + j).clamp(-(n + 1), n);
                let tap = if j.abs() <= n2 { 2 } else { 1 };
                t += s(p, pp) * tap;
            }
            fv[(i + n) as usize] = round2(t, log2_size);
        }
        for i in -n..n {
            p.data[(base + i * step) as usize] = fv[(i + n) as usize] as u16;
        }
    }
}

// ---------------------------------------------------------------------------
// CDEF (7.15)
// ---------------------------------------------------------------------------

/// The CDEF process (7.15): `CdefFrame` from `CurrFrame`.
pub(crate) fn cdef(f: &FrameCtx) -> FrameBuf {
    let mut out = f.cur.clone();
    let h = &f.hdr;
    if h.coded_lossless || h.allow_intrabc || !f.seq.enable_cdef {
        return out;
    }
    let mut r = 0;
    while r < f.mi_rows {
        let mut c = 0;
        while c < f.mi_cols {
            let idx = f.cdef_idx[(r >> 4) * f.cdef_stride + (c >> 4)];
            if idx >= 0 {
                cdef_block(f, &mut out, r, c, idx as usize);
            }
            c += 2;
        }
        r += 2;
    }
    out
}

/// The CDEF block process (7.15.1) for a block whose parameters are set.
fn cdef_block(f: &FrameCtx, out: &mut FrameBuf, r: usize, c: usize, idx: usize) {
    let cols = f.ms;
    let sk = |rr: usize, cc: usize| f.mi[rr * cols + cc].skip;
    let skip = sk(r, c) && sk(r + 1, c) && sk(r, c + 1) && sk(r + 1, c + 1);
    if skip {
        return;
    }
    let coeff_shift = f.bit_depth - 8;
    let (y_dir, var) = cdef_direction(f, r, c);
    let h = &f.hdr;
    let mut pri_str = h.cdef_y_pri_strength[idx] << coeff_shift;
    let sec_str = h.cdef_y_sec_strength[idx] << coeff_shift;
    let dir = if pri_str == 0 { 0 } else { y_dir };
    let var_str = if (var >> 6) != 0 {
        (floor_log2((var >> 6) as u32) as i32).min(12)
    } else {
        0
    };
    pri_str = if var != 0 {
        (pri_str * (4 + var_str) + 8) >> 4
    } else {
        0
    };
    let damping = h.cdef_damping + coeff_shift as i32;
    cdef_filter(f, out, 0, r, c, pri_str, sec_str, damping, dir);
    if f.num_planes == 1 {
        return;
    }
    let pri_str = h.cdef_uv_pri_strength[idx] << coeff_shift;
    let sec_str = h.cdef_uv_sec_strength[idx] << coeff_shift;
    let dir = if pri_str == 0 {
        0
    } else {
        CDEF_UV_DIR[f.ssx][f.ssy][y_dir]
    };
    let damping = h.cdef_damping + coeff_shift as i32 - 1;
    cdef_filter(f, out, 1, r, c, pri_str, sec_str, damping, dir);
    cdef_filter(f, out, 2, r, c, pri_str, sec_str, damping, dir);
}

/// The CDEF direction process (7.15.2): `(yDir, var)`.
fn cdef_direction(f: &FrameCtx, r: usize, c: usize) -> (usize, i32) {
    let mut cost = [0i32; 8];
    let mut partial = [[0i32; 15]; 8];
    let x0 = c << MI_SIZE_LOG2;
    let y0 = r << MI_SIZE_LOG2;
    let p = &f.cur.planes[0];
    let sh = f.bit_depth - 8;
    for i in 0..8 {
        for j in 0..8 {
            let x = (p.get(x0 + j, y0 + i) as i32 >> sh) - 128;
            partial[0][i + j] += x;
            partial[1][i + j / 2] += x;
            partial[2][i] += x;
            partial[3][3 + i - j / 2] += x;
            partial[4][7 + i - j] += x;
            partial[5][3 - i / 2 + j] += x;
            partial[6][j] += x;
            partial[7][i / 2 + j] += x;
        }
    }
    for i in 0..8 {
        cost[2] += partial[2][i] * partial[2][i];
        cost[6] += partial[6][i] * partial[6][i];
    }
    cost[2] *= DIV_TABLE[8];
    cost[6] *= DIV_TABLE[8];
    for i in 0..7 {
        cost[0] += (partial[0][i] * partial[0][i] + partial[0][14 - i] * partial[0][14 - i])
            * DIV_TABLE[i + 1];
        cost[4] += (partial[4][i] * partial[4][i] + partial[4][14 - i] * partial[4][14 - i])
            * DIV_TABLE[i + 1];
    }
    cost[0] += partial[0][7] * partial[0][7] * DIV_TABLE[8];
    cost[4] += partial[4][7] * partial[4][7] * DIV_TABLE[8];
    let mut i = 1;
    while i < 8 {
        for j in 0..5 {
            cost[i] += partial[i][3 + j] * partial[i][3 + j];
        }
        cost[i] *= DIV_TABLE[8];
        for j in 0..3 {
            cost[i] += (partial[i][j] * partial[i][j] + partial[i][10 - j] * partial[i][10 - j])
                * DIV_TABLE[2 * j + 2];
        }
        i += 2;
    }
    let mut best_cost = 0;
    let mut y_dir = 0;
    for (i, &cst) in cost.iter().enumerate() {
        if cst > best_cost {
            best_cost = cst;
            y_dir = i;
        }
    }
    let var = (best_cost - cost[(y_dir + 4) & 7]) >> 10;
    (y_dir, var)
}

fn constrain(diff: i32, threshold: i32, damping: i32) -> i32 {
    if threshold == 0 {
        return 0;
    }
    let damping_adj = (damping - floor_log2(threshold as u32) as i32).max(0);
    let mag = clip3(0, diff.abs(), threshold - (diff.abs() >> damping_adj));
    if diff < 0 { -mag } else { mag }
}

/// The CDEF filter process (7.15.3).
#[allow(clippy::too_many_arguments)]
fn cdef_filter(
    f: &FrameCtx,
    out: &mut FrameBuf,
    plane: usize,
    r: usize,
    c: usize,
    pri_str: i32,
    sec_str: i32,
    damping: i32,
    dir: usize,
) {
    let coeff_shift = f.bit_depth - 8;
    let (sub_x, sub_y) = f.plane_ss(plane);
    let x0 = (c * MI_SIZE) >> sub_x;
    let y0 = (r * MI_SIZE) >> sub_y;
    let w = 8 >> sub_x;
    let h = 8 >> sub_y;
    let src = &f.cur.planes[plane];
    let mi_rows = f.mi_rows as i32;
    let mi_cols = f.mi_cols as i32;
    let get = |i: usize, j: usize, dir: usize, k: usize, sign: i32| -> Option<i32> {
        let y = y0 as i32 + i as i32 + sign * CDEF_DIRECTIONS[dir][k][0];
        let x = x0 as i32 + j as i32 + sign * CDEF_DIRECTIONS[dir][k][1];
        let cand_r = (y << sub_y) >> MI_SIZE_LOG2;
        let cand_c = (x << sub_x) >> MI_SIZE_LOG2;
        if cand_r >= 0 && cand_r < mi_rows && cand_c >= 0 && cand_c < mi_cols {
            Some(src.get(x as usize, y as usize) as i32)
        } else {
            None
        }
    };
    let pri_taps = &CDEF_PRI_TAPS[((pri_str >> coeff_shift) & 1) as usize];
    let sec_taps = &CDEF_SEC_TAPS[((pri_str >> coeff_shift) & 1) as usize];
    let dst = &mut out.planes[plane];
    for i in 0..h {
        for j in 0..w {
            let mut sum = 0i32;
            let x = src.get(x0 + j, y0 + i) as i32;
            let mut max = x;
            let mut min = x;
            for k in 0..2 {
                for sign in [-1i32, 1] {
                    if let Some(p) = get(i, j, dir, k, sign) {
                        sum += pri_taps[k] * constrain(p - x, pri_str, damping);
                        max = max.max(p);
                        min = min.min(p);
                    }
                    for dir_off in [-2i32, 2] {
                        let d2 = ((dir as i32 + dir_off) & 7) as usize;
                        if let Some(s) = get(i, j, d2, k, sign) {
                            sum += sec_taps[k] * constrain(s - x, sec_str, damping);
                            max = max.max(s);
                            min = min.min(s);
                        }
                    }
                }
            }
            let v = x + ((8 + sum - (sum < 0) as i32) >> 4);
            dst.set(x0 + j, y0 + i, clip3(min, max, v) as u16);
        }
    }
}

// ---------------------------------------------------------------------------
// Super-resolution (7.16)
// ---------------------------------------------------------------------------

/// The upscaling process (7.16). Returns `None` without super-resolution
/// (the input is the output).
pub(crate) fn upscale(f: &FrameCtx, input: &FrameBuf) -> Option<FrameBuf> {
    if !f.hdr.use_superres {
        return None;
    }
    let maxv = (1i32 << f.bit_depth) - 1;
    let mut out = FrameBuf::default();
    for plane in 0..f.num_planes {
        let (sub_x, sub_y) = f.plane_ss(plane);
        let downscaled_w = round2(f.hdr.frame_width as i32, sub_x as u32) as i64;
        let upscaled_w = round2(f.hdr.upscaled_width as i32, sub_x as u32) as i64;
        let plane_h = round2(f.hdr.frame_height as i32, sub_y as u32) as usize;
        let step_x = ((downscaled_w << SUPERRES_SCALE_BITS) + upscaled_w / 2) / upscaled_w;
        let err = upscaled_w * step_x - (downscaled_w << SUPERRES_SCALE_BITS);
        let mut initial_subpel_x = (-((upscaled_w - downscaled_w) << (SUPERRES_SCALE_BITS - 1))
            + upscaled_w / 2)
            / upscaled_w
            + (1 << (SUPERRES_EXTRA_BITS - 1))
            - err / 2;
        initial_subpel_x &= SUPERRES_SCALE_MASK;
        let mi_w = (f.mi_cols >> sub_x) as i64;
        let max_x = mi_w * MI_SIZE as i64 - 1;
        let src = &input.planes[plane];
        let mut dst = PlaneBuf::new(upscaled_w as usize + 32, src.h, 0);
        for y in 0..plane_h {
            let row = src.row(y);
            for x in 0..upscaled_w {
                let src_x = -(1i64 << SUPERRES_SCALE_BITS) + initial_subpel_x + x * step_x;
                let src_x_px = src_x >> SUPERRES_SCALE_BITS;
                let src_x_subpel = ((src_x & SUPERRES_SCALE_MASK) >> SUPERRES_EXTRA_BITS) as usize;
                let mut sum = 0i32;
                for k in 0..SUPERRES_FILTER_TAPS {
                    let sx =
                        (src_x_px + (k - SUPERRES_FILTER_OFFSET) as i64).clamp(0, max_x) as usize;
                    sum += row[sx] as i32 * UPSCALE_FILTER[src_x_subpel][k as usize];
                }
                dst.set(
                    x as usize,
                    y,
                    round2(sum, FILTER_BITS as u32).clamp(0, maxv) as u16,
                );
            }
        }
        out.planes.push(dst);
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// Loop restoration (7.17)
// ---------------------------------------------------------------------------

struct LrCtx<'a> {
    cur: &'a PlaneBuf,
    cdef: &'a PlaneBuf,
    stripe_start_y: i32,
    stripe_end_y: i32,
    plane_end_x: i32,
    plane_end_y: i32,
}

impl LrCtx<'_> {
    /// The get source sample process (7.17.6).
    #[inline]
    fn sample(&self, x: i32, y: i32) -> i32 {
        let x = x.min(self.plane_end_x).max(0) as usize;
        let y = y.min(self.plane_end_y).max(0);
        if y < self.stripe_start_y {
            let y = (self.stripe_start_y - 2).max(y) as usize;
            self.cur.get(x, y) as i32
        } else if y > self.stripe_end_y {
            let y = (self.stripe_end_y + 2).min(y) as usize;
            self.cur.get(x, y) as i32
        } else {
            self.cdef.get(x, y as usize) as i32
        }
    }
}

/// The loop restoration process (7.17): `LrFrame` from `UpscaledCurrFrame`
/// and `UpscaledCdefFrame` (taken: it is the starting point of `LrFrame`).
pub(crate) fn loop_restoration(f: &FrameCtx, cur: &FrameBuf, cdef: FrameBuf) -> FrameBuf {
    if !f.hdr.uses_lr {
        return cdef;
    }
    let mut lr = cdef.clone();
    let mut y = 0;
    while y < f.hdr.frame_height as usize {
        let mut x = 0;
        while x < f.hdr.upscaled_width as usize {
            for plane in 0..f.num_planes {
                if f.hdr.frame_restoration_type[plane] != RESTORE_NONE {
                    loop_restore_block(
                        f,
                        cur,
                        &cdef,
                        &mut lr,
                        plane,
                        y >> MI_SIZE_LOG2,
                        x >> MI_SIZE_LOG2,
                    );
                }
            }
            x += MI_SIZE;
        }
        y += MI_SIZE;
    }
    lr
}

/// The loop restore block process (7.17.1).
fn loop_restore_block(
    f: &FrameCtx,
    cur: &FrameBuf,
    cdef: &FrameBuf,
    lr: &mut FrameBuf,
    plane: usize,
    row: usize,
    col: usize,
) {
    let luma_y = row * MI_SIZE;
    let stripe_num = (luma_y + 8) / 64;
    let (sub_x, sub_y) = f.plane_ss(plane);
    let stripe_start_y = (-8 + stripe_num as i32 * 64) >> sub_y;
    let stripe_end_y = stripe_start_y + (64 >> sub_y) - 1;
    let unit_size = f.hdr.loop_restoration_size[plane];
    let unit_rows = count_units_in_frame(
        unit_size,
        round2(f.hdr.frame_height as i32, sub_y as u32) as usize,
    );
    let unit_cols = count_units_in_frame(
        unit_size,
        round2(f.hdr.upscaled_width as i32, sub_x as u32) as usize,
    );
    let unit_row = (unit_rows - 1).min(((row * MI_SIZE + 8) >> sub_y) / unit_size);
    let unit_col = (unit_cols - 1).min(((col * MI_SIZE) >> sub_x) / unit_size);
    let plane_end_x = round2(f.hdr.upscaled_width as i32, sub_x as u32) - 1;
    let plane_end_y = round2(f.hdr.frame_height as i32, sub_y as u32) - 1;
    let x = ((col * MI_SIZE) >> sub_x) as i32;
    let y = ((row * MI_SIZE) >> sub_y) as i32;
    if x > plane_end_x || y > plane_end_y {
        return;
    }
    let w = ((MI_SIZE >> sub_x) as i32).min(plane_end_x - x + 1);
    let h = ((MI_SIZE >> sub_y) as i32).min(plane_end_y - y + 1);
    let lp = &f.lr[plane];
    let uidx = unit_row * lp.unit_cols + unit_col;
    let r_type = lp.lr_type[uidx];
    let ctx = LrCtx {
        cur: &cur.planes[plane],
        cdef: &cdef.planes[plane],
        stripe_start_y,
        stripe_end_y,
        plane_end_x,
        plane_end_y,
    };
    let maxv = (1i32 << f.bit_depth) - 1;
    let out = &mut lr.planes[plane];
    if r_type == RESTORE_WIENER {
        let rv = f.rounding_variables(false);
        let vfilter = wiener_coefficient(&lp.wiener[uidx][0]);
        let hfilter = wiener_coefficient(&lp.wiener[uidx][1]);
        let bd = f.bit_depth as i32;
        let offset = 1i32 << (bd + FILTER_BITS - rv.round0 as i32 - 1);
        let limit = (1i32 << (bd + 1 + FILTER_BITS - rv.round0 as i32)) - 1;
        let mut inter = [[0i32; 4]; 10];
        for r in 0..(h + 6) as usize {
            for c in 0..w as usize {
                let mut s = 0;
                for t in 0..7 {
                    s += hfilter[t] * ctx.sample(x + c as i32 + t as i32 - 3, y + r as i32 - 3);
                }
                let v = round2(s, rv.round0);
                inter[r][c] = clip3(-offset, limit - offset, v);
            }
        }
        for r in 0..h as usize {
            for c in 0..w as usize {
                let mut s = 0;
                for t in 0..7 {
                    s += vfilter[t] * inter[r + t][c];
                }
                let v = round2(s, rv.round1);
                out.set(x as usize + c, y as usize + r, v.clamp(0, maxv) as u16);
            }
        }
    } else if r_type == RESTORE_SGRPROJ {
        let set = lp.sgr_set[uidx] as usize;
        let flt0 = box_filter(f, &ctx, x, y, w, h, set, 0);
        let flt1 = box_filter(f, &ctx, x, y, w, h, set, 1);
        let w0 = lp.sgr_xqd[uidx][0];
        let w1 = lp.sgr_xqd[uidx][1];
        let w2 = (1 << SGRPROJ_PRJ_BITS) - w0 - w1;
        let r0 = SGR_PARAMS[set][0];
        let r1 = SGR_PARAMS[set][2];
        for i in 0..h as usize {
            for j in 0..w as usize {
                let u = (ctx.cdef.get(x as usize + j, y as usize + i) as i32) << SGRPROJ_RST_BITS;
                let mut v = w1 * u;
                v += if r0 != 0 { w0 * flt0[i][j] } else { w0 * u };
                v += if r1 != 0 { w2 * flt1[i][j] } else { w2 * u };
                let s = round2(v, (SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS) as u32);
                out.set(x as usize + j, y as usize + i, s.clamp(0, maxv) as u16);
            }
        }
    }
}

/// The box filter process (7.17.3).
#[allow(clippy::too_many_arguments)]
fn box_filter(
    f: &FrameCtx,
    ctx: &LrCtx,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    set: usize,
    pass: usize,
) -> [[i32; 4]; 4] {
    let mut out = [[0i32; 4]; 4];
    let r = SGR_PARAMS[set][pass * 2];
    if r == 0 {
        return out;
    }
    let eps = SGR_PARAMS[set][pass * 2 + 1];
    let bd = f.bit_depth;
    let n = (2 * r + 1) * (2 * r + 1);
    let n2e = n * n * eps;
    let s = ((1 << SGRPROJ_MTABLE_BITS) + n2e / 2) / n2e;
    let mut a_arr = [[0i32; 6]; 6];
    let mut b_arr = [[0i32; 6]; 6];
    let one_over_n = ((1 << SGRPROJ_RECIP_BITS) + n / 2) / n;
    for i in -1..h + 1 {
        for j in -1..w + 1 {
            let mut a: i64 = 0;
            let mut b: i32 = 0;
            for dy in -r..=r {
                for dx in -r..=r {
                    let c = ctx.sample(x + j + dx, y + i + dy);
                    a += (c * c) as i64;
                    b += c;
                }
            }
            let a = round2_64(a, 2 * (bd - 8));
            let d = round2(b, bd - 8) as i64;
            let p = (a * n as i64 - d * d).max(0);
            let z = round2_64(p * s as i64, SGRPROJ_MTABLE_BITS as u32);
            let a2: i32 = if z >= 255 {
                256
            } else if z == 0 {
                1
            } else {
                (((z << SGRPROJ_SGR_BITS) + z / 2) / (z + 1)) as i32
            };
            let b2 = ((1 << SGRPROJ_SGR_BITS) - a2) as i64 * b as i64 * one_over_n as i64;
            a_arr[(i + 1) as usize][(j + 1) as usize] = a2;
            b_arr[(i + 1) as usize][(j + 1) as usize] =
                round2_64(b2, SGRPROJ_RECIP_BITS as u32) as i32;
        }
    }
    for i in 0..h {
        let shift = if pass == 0 && (i & 1) != 0 { 4 } else { 5 };
        for j in 0..w {
            let mut a = 0i32;
            let mut b = 0i32;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let weight = if pass == 0 {
                        if ((i + dy) & 1) != 0 {
                            if dx == 0 { 6 } else { 5 }
                        } else {
                            0
                        }
                    } else if dx == 0 || dy == 0 {
                        4
                    } else {
                        3
                    };
                    a += weight * a_arr[(i + dy + 1) as usize][(j + dx + 1) as usize];
                    b += weight * b_arr[(i + dy + 1) as usize][(j + dx + 1) as usize];
                }
            }
            let v = a * ctx.cdef.get((x + j) as usize, (y + i) as usize) as i32 + b;
            out[i as usize][j as usize] =
                round2(v, (SGRPROJ_SGR_BITS + shift - SGRPROJ_RST_BITS) as u32);
        }
    }
    out
}

/// The Wiener coefficient process (7.17.5).
fn wiener_coefficient(coeff: &[i32; 3]) -> [i32; 7] {
    let mut filter = [0i32; 7];
    filter[3] = 128;
    for i in 0..3 {
        let c = coeff[i];
        filter[i] = c;
        filter[6 - i] = c;
        filter[3] -= 2 * c;
    }
    filter
}

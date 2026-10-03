//! Adaptive quantisation by temporal importance: each superblock's
//! quantiser, coded as a `delta_qindex`.
//!
//! A superblock that the previous frame predicts well (little change
//! against the previous source, compared with its own texture) is likely to
//! be predicted from again by the frames after it, so its quality carries
//! into them: it is coded finer, and a superblock that changes completely
//! (motion the previous frame does not predict, new content) coarser. With
//! `p` the share of the superblock's texture the previous frame predicts
//! (1 - inter / intra, clamped to 0..1), its importance is `1 + K p`, and
//! its quantiser step the frame's times `(importance / mean)^-ALPHA` (the
//! mean geometric, so the frame's rate stays about where it was).
//!
//! Off by default (`Tools::aq`): without a lookahead the previous frame is
//! a poor guide to which superblocks the next frames will predict from. On
//! the encoder's test clips it measured -0.2 % BD-rate (PSNR-Y) over 10
//! frames and +0.2 % over 30 (one clip -0.8 %, two +0.6 to +0.8 %).

use crate::tables::AC_QLOOKUP;

/// How many later frames a well-predicted superblock is taken to serve.
const K: f64 = 1.0;
/// The strength: the exponent of the step's change.
const ALPHA: f64 = 0.35;
/// The most a superblock's quantiser index moves from the frame's.
const MAX_DELTA: i32 = 32;

/// Each superblock's quantiser index for a frame at `base_q`, from the
/// source luma `y` and the previous frame's source `prev` (same layout,
/// `stride`), `w` x `h`; `None` when every superblock would keep `base_q`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn quantisers(
    y: &[u16],
    prev: &[u16],
    stride: usize,
    w: usize,
    h: usize,
    bit_depth: u32,
    base_q: u32,
) -> Option<(usize, Vec<u8>)> {
    let sb_cols = w.div_ceil(64);
    let sb_rows = h.div_ceil(64);
    let mut importance = Vec::with_capacity(sb_cols * sb_rows);
    for sr in 0..sb_rows {
        for sc in 0..sb_cols {
            let (x0, y0) = (sc * 64, sr * 64);
            let (x1, y1) = ((x0 + 64).min(w), (y0 + 64).min(h));
            // Per 16x16 block (every other sample): deviation from the
            // block's mean, and difference from the previous source.
            let (mut intra, mut inter) = (0u64, 0u64);
            for by in (y0..y1).step_by(16) {
                for bx in (x0..x1).step_by(16) {
                    let (ex, ey) = ((bx + 16).min(x1), (by + 16).min(y1));
                    let (mut sum, mut n) = (0u64, 0u64);
                    for yy in (by..ey).step_by(2) {
                        for xx in (bx..ex).step_by(2) {
                            sum += y[yy * stride + xx] as u64;
                            n += 1;
                        }
                    }
                    let mean = (sum / n.max(1)) as i64;
                    for yy in (by..ey).step_by(2) {
                        for xx in (bx..ex).step_by(2) {
                            let v = y[yy * stride + xx] as i64;
                            intra += (v - mean).unsigned_abs();
                            inter += (v - prev[yy * stride + xx] as i64).unsigned_abs();
                        }
                    }
                }
            }
            let p = if intra == 0 {
                1.0
            } else {
                (1.0 - inter as f64 / intra as f64).clamp(0.0, 1.0)
            };
            importance.push(1.0 + K * p);
        }
    }
    let mean = (importance.iter().map(|v| v.ln()).sum::<f64>() / importance.len() as f64).exp();
    let bdi = ((bit_depth - 8) >> 1) as usize;
    let table = &AC_QLOOKUP[bdi];
    let base = table[base_q as usize] as f64;
    let mut any = false;
    let q: Vec<u8> = importance
        .iter()
        .map(|&imp| {
            let step = base * (imp / mean).powf(-ALPHA);
            // The index whose step is nearest (the table rises).
            let i = table.partition_point(|&t| (t as f64) < step);
            let i =
                if i > 0 && (i == 256 || (step - table[i - 1] as f64) < (table[i] as f64 - step)) {
                    i - 1
                } else {
                    i
                };
            let qi = (i as i32).clamp(base_q as i32 - MAX_DELTA, base_q as i32 + MAX_DELTA);
            let qi = qi.clamp(1, 255);
            any |= qi != base_q as i32;
            qi as u8
        })
        .collect();
    any.then_some((sb_cols, q))
}

//! The encoder's loop restoration search: each unit's filter (none, a
//! Wiener filter fitted to the source, or a self-guided filter with fitted
//! projection weights), and each plane's frame restoration type.
//!
//! Everything is measured with the decoder's own restoration
//! (`postfilter::restore_rect`), over the samples it reads (the stripes'
//! boundary rows come from the deblocked frame), so the error of a choice
//! is the error the decoder's output will have.

use crate::consts::*;
use crate::decoder::FrameCtx;
use crate::decoder::postfilter::{
    LRB, LrBuffers, LrCtx, LrParams, box_filter_rect, lr_stripe, lr_unit_cols, lr_window,
    restore_rect_from_window,
};
use crate::decoder::state::FrameBuf;
use crate::tables::*;

/// One unit's restoration, as `read_lr_unit()` codes it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct LrChoice {
    /// `RESTORE_NONE`, `RESTORE_WIENER` or `RESTORE_SGRPROJ`.
    pub(crate) t: u8,
    pub(crate) wiener: [[i32; 3]; 2],
    pub(crate) set: u8,
    pub(crate) xqd: [i32; 2],
}

/// The frame's loop restoration: per plane, `FrameRestorationType` and
/// each unit's choice.
#[derive(Clone, Debug, Default)]
pub(crate) struct LrPlan {
    pub(crate) frame_type: [u8; 3],
    pub(crate) units: [Vec<LrChoice>; 3],
}

/// The self-guided parameter sets tried.
const SGR_SETS_NORMAL: [usize; 6] = [0, 3, 6, 10, 13, 14];
const SGR_SETS_FAST: [usize; 2] = [3, 10];

/// How hard the search works.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Effort {
    /// Filters fitted on every other row and column; two self-guided
    /// parameter sets.
    Fast,
    /// Six self-guided parameter sets.
    Normal,
    /// All sixteen.
    Thorough,
}

/// Chooses the restoration of every unit of every plane of `f` (whose
/// header asks for switchable restoration in each, so its unit arrays are
/// set up). `cur` is the deblocked frame, `cdef` the CDEF output; `src` the
/// source planes.
pub(crate) fn search(
    f: &FrameCtx,
    cur: &FrameBuf,
    cdef: &FrameBuf,
    src: &[Vec<u16>],
    stride: &[usize],
    lambda: f64,
    effort: Effort,
) -> LrPlan {
    let mut plan = LrPlan::default();
    let sets: Vec<usize> = match effort {
        Effort::Thorough => (0..16).collect(),
        Effort::Normal => SGR_SETS_NORMAL.to_vec(),
        Effort::Fast => SGR_SETS_FAST.to_vec(),
    };
    let step = if effort == Effort::Fast { 2 } else { 1 };
    let stripes = (f.hdr.frame_height as usize + 8).div_ceil(64);
    for plane in 0..f.num_planes {
        let lp = &f.lr[plane];
        let units = lp.unit_rows * lp.unit_cols;
        if units == 0 {
            continue;
        }
        let mut choices = vec![LrChoice::default(); units];
        let mut buf = LrBuffers::default();
        // The stripes of each unit row.
        let mut rows: Vec<Vec<(i32, i32, i32, i32)>> = vec![Vec::new(); lp.unit_rows];
        for s in 0..stripes {
            if let Some((ss, se, y0, y1, ur)) = lr_stripe(f, plane, s) {
                rows[ur].push((ss, se, y0, y1));
            }
        }
        let (sub_x, sub_y) = f.plane_ss(plane);
        let plane_end_x = crate::consts::round2(f.hdr.upscaled_width as i32, sub_x as u32) - 1;
        let plane_end_y = crate::consts::round2(f.hdr.frame_height as i32, sub_y as u32) - 1;
        let mut wiener_bits = 0usize;
        let mut sgr_bits = 0usize;
        for ur in 0..lp.unit_rows {
            for uc in 0..lp.unit_cols {
                let (x0, x1) = lr_unit_cols(f, plane, uc);
                if x0 > plane_end_x || rows[ur].is_empty() {
                    continue;
                }
                let w = (x1 - x0 + 1) as usize;
                // The unit's rectangles, one per stripe.
                let rects: Vec<(LrCtx, i32, usize)> = rows[ur]
                    .iter()
                    .map(|&(ss, se, y0, y1)| {
                        (
                            LrCtx {
                                cur: &cur.planes[plane],
                                cdef: &cdef.planes[plane],
                                stripe_start_y: ss,
                                stripe_end_y: se,
                                plane_end_x,
                                plane_end_y,
                            },
                            y0,
                            (y1 - y0 + 1) as usize,
                        )
                    })
                    .collect();
                let sse_of = |buf: &mut LrBuffers, params: Option<&LrParams>| -> u64 {
                    let mut total = 0u64;
                    for (ctx, y0, h) in &rects {
                        lr_window(ctx, x0, *y0, w, *h, buf);
                        match params {
                            Some(p) => restore_rect_from_window(f, ctx, x0, *y0, w, *h, p, buf),
                            None => {
                                buf.out.resize(w * h, 0);
                                for i in 0..*h {
                                    for j in 0..w {
                                        buf.out[i * w + j] =
                                            ctx.cdef.get(x0 as usize + j, *y0 as usize + i);
                                    }
                                }
                            }
                        }
                        for i in 0..*h {
                            let so = (*y0 as usize + i) * stride[plane] + x0 as usize;
                            for j in 0..w {
                                let d = buf.out[i * w + j] as i64 - src[plane][so + j] as i64;
                                total += (d * d) as u64;
                            }
                        }
                    }
                    total
                };
                let none = sse_of(&mut buf, None);
                let mut best = (none as f64 + lambda, LrChoice::default());
                // Wiener.
                let coef = fit_wiener(plane, &rects, x0, w, src, stride, step, &mut buf);
                let p = LrParams::Wiener(coef);
                let e = sse_of(&mut buf, Some(&p));
                let cost = e as f64 + lambda * if plane == 0 { 40.0 } else { 28.0 };
                if cost < best.0 {
                    best = (
                        cost,
                        LrChoice {
                            t: RESTORE_WIENER,
                            wiener: coef,
                            ..Default::default()
                        },
                    );
                }
                // Self-guided.
                for &set in &sets {
                    let (xqd, e) =
                        fit_sgr(f, set, &rects, x0, w, src, stride, plane, step, &mut buf);
                    let cost = e as f64 + lambda * 20.0;
                    if cost < best.0 {
                        best = (
                            cost,
                            LrChoice {
                                t: RESTORE_SGRPROJ,
                                set: set as u8,
                                xqd,
                                ..Default::default()
                            },
                        );
                    }
                }
                match best.1.t {
                    RESTORE_WIENER => wiener_bits += 1,
                    RESTORE_SGRPROJ => sgr_bits += 1,
                    _ => {}
                }
                choices[ur * lp.unit_cols + uc] = best.1;
            }
        }
        plan.frame_type[plane] = match (wiener_bits > 0, sgr_bits > 0) {
            (false, false) => RESTORE_NONE,
            (true, false) => RESTORE_WIENER,
            (false, true) => RESTORE_SGRPROJ,
            (true, true) => RESTORE_SWITCHABLE,
        };
        plan.units[plane] = choices;
    }
    plan
}

/// Solves the normal equations `m x = v` (Gaussian elimination with
/// partial pivoting); zeros when singular.
fn solve<const N: usize>(mut m: [[f64; N]; N], mut v: [f64; N]) -> [f64; N] {
    for i in 0..N {
        let piv = (i..N)
            .max_by(|&a, &b| m[a][i].abs().total_cmp(&m[b][i].abs()))
            .unwrap_or(i);
        m.swap(i, piv);
        v.swap(i, piv);
        if m[i][i].abs() < 1e-9 {
            return [0.0; N];
        }
        for r in 0..N {
            if r != i {
                let f = m[r][i] / m[i][i];
                for c in i..N {
                    m[r][c] -= f * m[i][c];
                }
                v[r] -= f * v[i];
            }
        }
    }
    std::array::from_fn(|i| v[i] / m[i][i])
}

/// A separable symmetric Wiener filter for the unit, fitted by least
/// squares alternately in each direction, then quantised to the coded
/// ranges: `[vertical, horizontal]` coefficients (outermost first).
#[allow(clippy::too_many_arguments)]
fn fit_wiener(
    plane: usize,
    rects: &[(LrCtx, i32, usize)],
    x0: i32,
    w: usize,
    src: &[Vec<u16>],
    stride: &[usize],
    step: usize,
    buf: &mut LrBuffers,
) -> [[i32; 3]; 2] {
    // Chroma filters have no outermost tap.
    let first = if plane == 0 { 0 } else { 1 };
    let ww = w + 2 * LRB;
    // Coefficients as fractions of 128: index 0 outermost.
    let mut hc = [0f64; 3];
    let mut vc = [0f64; 3];
    for iter in 0..3 {
        let vertical = iter % 2 == 0;
        let mut m = [[0f64; 3]; 3];
        let mut v = [0f64; 3];
        for (ctx, y0, h) in rects {
            lr_window(ctx, x0, *y0, w, *h, buf);
            let win = &buf.win;
            let at = |r: usize, c: usize| win[r * ww + c] as f64;
            for i in (0..*h).step_by(step) {
                let so = (*y0 as usize + i) * stride[plane] + x0 as usize;
                for j in (0..w).step_by(step) {
                    // The other direction's filter applied first: the
                    // samples along this direction (7 of them).
                    let mut line = [0f64; 7];
                    for (t, l) in line.iter_mut().enumerate() {
                        let (r, c) = if vertical {
                            (i + t, j + LRB)
                        } else {
                            (i + LRB, j + t)
                        };
                        // The other direction at (r, c).
                        let other = if vertical { &hc } else { &vc };
                        let center = at(r, c);
                        let mut s = center;
                        for k in 0..3 {
                            let d = 3 - k;
                            let (a, b) = if vertical {
                                (at(r, c - d), at(r, c + d))
                            } else {
                                (at(r - d, c), at(r + d, c))
                            };
                            s += other[k] * (a + b - 2.0 * center);
                        }
                        *l = s;
                    }
                    let target = src[plane][so + j] as f64 - line[3];
                    let d: [f64; 3] =
                        std::array::from_fn(|k| line[k] + line[6 - k] - 2.0 * line[3]);
                    for a in first..3 {
                        for b in first..3 {
                            m[a][b] += d[a] * d[b];
                        }
                        v[a] += d[a] * target;
                    }
                }
            }
        }
        let sol = if first == 0 {
            solve(m, v)
        } else {
            let s2 = solve([[m[1][1], m[1][2]], [m[2][1], m[2][2]]], [v[1], v[2]]);
            [0.0, s2[0], s2[1]]
        };
        if vertical {
            vc = sol;
        } else {
            hc = sol;
        }
    }
    let q = |c: [f64; 3]| -> [i32; 3] {
        std::array::from_fn(|k| {
            if k < first {
                0
            } else {
                ((c[k] * 128.0).round() as i32).clamp(WIENER_TAPS_MIN[k], WIENER_TAPS_MAX[k])
            }
        })
    };
    [q(vc), q(hc)]
}

/// The self-guided projection weights `xqd` for parameter set `set`,
/// fitted by least squares and clamped to the coded ranges (the second
/// derived from the first when the second filter is off, as the decoder
/// derives it).
#[allow(clippy::too_many_arguments)]
fn fit_sgr(
    f: &FrameCtx,
    set: usize,
    rects: &[(LrCtx, i32, usize)],
    x0: i32,
    w: usize,
    src: &[Vec<u16>],
    stride: &[usize],
    plane: usize,
    step: usize,
    buf: &mut LrBuffers,
) -> ([i32; 2], u64) {
    let r0 = SGR_PARAMS[set][0];
    let r1 = SGR_PARAMS[set][2];
    let mut m = [[0f64; 2]; 2];
    let mut v = [0f64; 2];
    // The two filters' outputs over every rectangle, kept for the error.
    let mut flts: Vec<[Vec<i32>; 2]> = Vec::with_capacity(rects.len());
    for (ctx, y0, h) in rects {
        lr_window(ctx, x0, *y0, w, *h, buf);
        for pass in 0..2 {
            box_filter_rect(f, ctx, x0, *y0, w, *h, set, pass, buf);
        }
        flts.push([buf.flt[0].clone(), buf.flt[1].clone()]);
        for i in (0..*h).step_by(step) {
            let so = (*y0 as usize + i) * stride[plane] + x0 as usize;
            for j in (0..w).step_by(step) {
                let u = ((ctx.cdef.get(x0 as usize + j, *y0 as usize + i) as i32)
                    << SGRPROJ_RST_BITS) as f64;
                let a = if r0 != 0 {
                    buf.flt[0][i * w + j] as f64 - u
                } else {
                    0.0
                };
                let b = if r1 != 0 {
                    buf.flt[1][i * w + j] as f64 - u
                } else {
                    0.0
                };
                let t = 128.0 * (((src[plane][so + j] as i32) << SGRPROJ_RST_BITS) as f64 - u);
                m[0][0] += a * a;
                m[0][1] += a * b;
                m[1][1] += b * b;
                v[0] += a * t;
                v[1] += b * t;
            }
        }
    }
    m[1][0] = m[0][1];
    let (w0, w2) = if r0 != 0 && r1 != 0 {
        let s = solve(m, v);
        (s[0], s[1])
    } else if r0 != 0 {
        (if m[0][0] > 0.0 { v[0] / m[0][0] } else { 0.0 }, 0.0)
    } else {
        (0.0, if m[1][1] > 0.0 { v[1] / m[1][1] } else { 0.0 })
    };
    let x0q = if r0 != 0 {
        (w0.round() as i32).clamp(SGRPROJ_XQD_MIN[0], SGRPROJ_XQD_MAX[0])
    } else {
        0
    };
    let x1q = if r1 != 0 {
        ((1 << SGRPROJ_PRJ_BITS) - x0q - w2.round() as i32)
            .clamp(SGRPROJ_XQD_MIN[1], SGRPROJ_XQD_MAX[1])
    } else {
        ((1 << SGRPROJ_PRJ_BITS) - x0q).clamp(SGRPROJ_XQD_MIN[1], SGRPROJ_XQD_MAX[1])
    };
    // The error of the restoration with these weights (7.17.2's
    // projection, as `restore_rect` applies it).
    let (w0, w1) = (x0q, x1q);
    let w2 = (1 << SGRPROJ_PRJ_BITS) - w0 - w1;
    let maxv = (1i32 << f.bit_depth) - 1;
    let mut sse = 0u64;
    for ((ctx, y0, h), flt) in rects.iter().zip(&flts) {
        for i in 0..*h {
            let so = (*y0 as usize + i) * stride[plane] + x0 as usize;
            for j in 0..w {
                let u =
                    (ctx.cdef.get(x0 as usize + j, *y0 as usize + i) as i32) << SGRPROJ_RST_BITS;
                let mut val = w1 * u;
                val += if r0 != 0 {
                    w0 * flt[0][i * w + j]
                } else {
                    w0 * u
                };
                val += if r1 != 0 {
                    w2 * flt[1][i * w + j]
                } else {
                    w2 * u
                };
                let out = round2(val, (SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS) as u32).clamp(0, maxv);
                let d = out as i64 - src[plane][so + j] as i64;
                sse += (d * d) as u64;
            }
        }
    }
    ([x0q, x1q], sse)
}

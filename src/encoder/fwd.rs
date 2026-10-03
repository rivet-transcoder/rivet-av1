//! Forward transforms for the encoder.
//!
//! The specification defines only the inverse transforms. Their 1D kernels
//! (DCT, ADST, identity) have orthogonal basis vectors, so a forward
//! transform is the adjoint of the inverse, normalised: the basis vectors
//! are measured once by running the normative inverse kernels on impulses,
//! and the 2D gain of the inverse (the rectangular scaling and the row and
//! column shifts of 7.13.3) is divided out. The result need not be exact:
//! the decoder reconstructs from the quantised levels with the normative
//! inverse, which the encoder's reconstruction uses too.

use std::sync::OnceLock;

use crate::consts::*;
use crate::dsp::itx::{inverse_1d, kinds};
use crate::tables::{TRANSFORM_ROW_SHIFT, TX_HEIGHT_LOG2, TX_WIDTH_LOG2};

/// The forward kernel of one 1D inverse kernel: `f[i * n + k]` is the
/// weight of input sample `i` in coefficient `k` (the basis vector of `k`
/// divided by its squared length). The DCT's basis vectors are even or
/// odd about the middle, so it is also kept as two half-size matrices
/// over the sums and differences of mirrored inputs (`fe`, `fo`; made
/// exactly symmetric), and the identity as its diagonal.
struct Basis {
    n: usize,
    kind: u8,
    f: Vec<f32>,
    /// `fe[i * (n / 2) + k]`: input pair `i`, coefficient `2k`.
    fe: Vec<f32>,
    /// `fo[i * (n / 2) + k]`: input pair `i`, coefficient `2k + 1`.
    fo: Vec<f32>,
    diag: Vec<f32>,
    /// The largest weight's magnitude.
    max: f32,
}

fn basis(kind: u8, log2n: u32) -> &'static Basis {
    static CACHE: OnceLock<Vec<Basis>> = OnceLock::new();
    let all = CACHE.get_or_init(|| {
        let mut v = Vec::new();
        for kind in 0..3u8 {
            for log2n in 0..7u32 {
                let n = 1usize << log2n;
                let valid = log2n >= 2 && (kind != 1 || log2n <= 4) && (kind != 2 || log2n <= 5);
                let mut f = vec![0.0f32; n * n];
                if valid {
                    const A: i32 = 1 << 16;
                    for k in 0..n {
                        let mut t = [0i32; 64];
                        t[k] = A;
                        inverse_1d(kind, log2n, &mut t[..n]);
                        let col: Vec<f64> = t[..n].iter().map(|&x| x as f64 / A as f64).collect();
                        let norm2: f64 = col.iter().map(|x| x * x).sum();
                        for i in 0..n {
                            f[i * n + k] = (col[i] / norm2) as f32;
                        }
                    }
                }
                let half = n / 2;
                let mut fe = vec![0.0f32; half * half];
                let mut fo = vec![0.0f32; half * half];
                if kind == 0 {
                    for i in 0..half {
                        for k in 0..half {
                            let m = n - 1 - i;
                            fe[i * half + k] = 0.5 * (f[i * n + 2 * k] + f[m * n + 2 * k]);
                            fo[i * half + k] = 0.5 * (f[i * n + 2 * k + 1] - f[m * n + 2 * k + 1]);
                        }
                    }
                }
                let diag = (0..n).map(|i| f[i * n + i]).collect();
                let max = f.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
                v.push(Basis {
                    n,
                    kind,
                    f,
                    fe,
                    fo,
                    diag,
                    max,
                });
            }
        }
        v
    });
    &all[kind as usize * 7 + log2n as usize]
}

/// The 2D gain of the inverse transform at `tx_sz` (divided out).
fn gain(tx_sz: usize) -> f64 {
    let log2w = TX_WIDTH_LOG2[tx_sz] as u32;
    let log2h = TX_HEIGHT_LOG2[tx_sz] as u32;
    let rect = if log2w.abs_diff(log2h) == 1 {
        2896.0 / 4096.0
    } else {
        1.0
    };
    rect / (1u64 << (TRANSFORM_ROW_SHIFT[tx_sz] as u32 + 4)) as f64
}

/// A bound on the magnitude of every coefficient [`forward_2d`] gives for
/// a residual whose absolute values sum to 1.
pub(crate) fn coef_bound(tx_sz: usize, tx_type: usize) -> f64 {
    let (rk, ck) = kinds(tx_type);
    let br = basis(rk, TX_WIDTH_LOG2[tx_sz] as u32);
    let bc = basis(ck, TX_HEIGHT_LOG2[tx_sz] as u32);
    // Slack for the f32 arithmetic.
    1.001 * br.max as f64 * bc.max as f64 / gain(tx_sz)
}

/// The 1D forward transform of `lanes` lines at once: input `i` of line
/// `l` is `x[i * lanes + l]`; coefficient `k` (for `k < keep`) goes to
/// `out[k * lanes + l]`. The loops over the lanes are what vectorises.
#[inline(always)]
fn forward_lanes(b: &Basis, x: &[f32], lanes: usize, keep: usize, out: &mut [f32]) {
    let n = b.n;
    out[..keep * lanes].fill(0.0);
    match b.kind {
        2 => {
            for k in 0..keep {
                let d = b.diag[k];
                let o = &mut out[k * lanes..(k + 1) * lanes];
                for (a, &v) in o.iter_mut().zip(&x[k * lanes..(k + 1) * lanes]) {
                    *a = v * d;
                }
            }
        }
        0 => {
            let half = n / 2;
            let mut s = [0f32; 64];
            let mut d = [0f32; 64];
            for i in 0..half {
                let lo = &x[i * lanes..(i + 1) * lanes];
                let hi = &x[(n - 1 - i) * lanes..(n - i) * lanes];
                for l in 0..lanes {
                    s[l] = lo[l] + hi[l];
                    d[l] = lo[l] - hi[l];
                }
                let fe = &b.fe[i * half..(i + 1) * half];
                let fo = &b.fo[i * half..(i + 1) * half];
                for k in 0..keep.div_ceil(2) {
                    let w = fe[k];
                    let o = &mut out[2 * k * lanes..(2 * k + 1) * lanes];
                    for (a, &v) in o.iter_mut().zip(&s[..lanes]) {
                        *a += v * w;
                    }
                }
                for k in 0..keep / 2 {
                    let w = fo[k];
                    let o = &mut out[(2 * k + 1) * lanes..(2 * k + 2) * lanes];
                    for (a, &v) in o.iter_mut().zip(&d[..lanes]) {
                        *a += v * w;
                    }
                }
            }
        }
        _ => {
            for i in 0..n {
                let xi = &x[i * lanes..(i + 1) * lanes];
                for k in 0..keep {
                    let w = b.f[i * n + k];
                    let o = &mut out[k * lanes..(k + 1) * lanes];
                    for (a, &v) in o.iter_mut().zip(xi) {
                        *a += v * w;
                    }
                }
            }
        }
    }
}

/// The forward 2D transform of a `w` x `h` residual (row-major) for
/// `tx_type` at `tx_sz`, as the coefficients the inverse process expects
/// in `Dequant` (before quantisation), `out[k * w + j]`; only the top-left
/// 32x32, which the syntax can code, is computed. The flipped types flip
/// the residual first, as the reconstruction flips it back (7.13.3).
pub(crate) fn forward_2d(residual: &[i32], tx_sz: usize, tx_type: usize, out: &mut [f64]) {
    SCRATCH.with_borrow_mut(|(t, col)| {
        #[cfg(target_arch = "x86_64")]
        if crate::dsp::avx2() {
            // SAFETY: AVX2 is available.
            unsafe { forward_2d_avx2(residual, tx_sz, tx_type, out, t, col) };
            return;
        }
        forward_2d_body(residual, tx_sz, tx_type, out, t, col)
    })
}

/// [`forward_2d`] compiled for AVX2: the same operations in the same
/// order (no fused multiply-adds), so the same results, eight lanes wide.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn forward_2d_avx2(
    residual: &[i32],
    tx_sz: usize,
    tx_type: usize,
    out: &mut [f64],
    t: &mut [f32],
    col: &mut [f32],
) {
    forward_2d_body(residual, tx_sz, tx_type, out, t, col)
}

/// The 1D forward transform of one line `x` (`b.n` inputs) into its first
/// `keep` coefficients; the loops over the coefficients vectorise.
#[inline(always)]
fn forward_line(b: &Basis, x: &[f32], keep: usize, out: &mut [f32]) {
    let n = b.n;
    let out = &mut out[..keep];
    match b.kind {
        2 => {
            for (k, o) in out.iter_mut().enumerate() {
                *o = x[k] * b.diag[k];
            }
        }
        0 => {
            let half = n / 2;
            let (ke, ko) = (keep.div_ceil(2), keep / 2);
            let mut e = [0f32; 32];
            let mut o = [0f32; 32];
            for i in 0..half {
                let (s, d) = (x[i] + x[n - 1 - i], x[i] - x[n - 1 - i]);
                let fe = &b.fe[i * half..i * half + ke];
                for (a, &w) in e[..ke].iter_mut().zip(fe) {
                    *a += s * w;
                }
                let fo = &b.fo[i * half..i * half + ko];
                for (a, &w) in o[..ko].iter_mut().zip(fo) {
                    *a += d * w;
                }
            }
            for k in 0..ke {
                out[2 * k] = e[k];
            }
            for k in 0..ko {
                out[2 * k + 1] = o[k];
            }
        }
        _ => {
            out.fill(0.0);
            for (i, &v) in x[..n].iter().enumerate() {
                let f = &b.f[i * n..i * n + keep];
                for (a, &w) in out.iter_mut().zip(f) {
                    *a += v * w;
                }
            }
        }
    }
}

thread_local! {
    /// Scratch of the 2D transform: the row transforms, the columns'.
    static SCRATCH: std::cell::RefCell<(Vec<f32>, Vec<f32>)> =
        std::cell::RefCell::new((vec![0.0; 64 * 32], vec![0.0; 32 * 32]));
}

#[inline(always)]
fn forward_2d_body(
    residual: &[i32],
    tx_sz: usize,
    tx_type: usize,
    out: &mut [f64],
    t: &mut [f32],
    col: &mut [f32],
) {
    let log2w = TX_WIDTH_LOG2[tx_sz] as u32;
    let log2h = TX_HEIGHT_LOG2[tx_sz] as u32;
    let w = 1usize << log2w;
    let h = 1usize << log2h;
    let tw = w.min(32);
    let th = h.min(32);
    let flip_ud = matches!(
        tx_type,
        FLIPADST_DCT | FLIPADST_ADST | V_FLIPADST | FLIPADST_FLIPADST
    );
    let flip_lr = matches!(
        tx_type,
        DCT_FLIPADST | ADST_FLIPADST | H_FLIPADST | FLIPADST_FLIPADST
    );
    let (rk, ck) = kinds(tx_type);
    let br = basis(rk, log2w);
    let bc = basis(ck, log2h);
    debug_assert!(br.n == w && bc.n == h);
    {
        // Rows: coefficient k of row i at t[i * tw + k].
        let mut x = [0f32; 64];
        for i in 0..h {
            let si = if flip_ud { h - 1 - i } else { i };
            let row = &residual[si * w..(si + 1) * w];
            if flip_lr {
                for (v, &r) in x[..w].iter_mut().zip(row.iter().rev()) {
                    *v = r as f32;
                }
            } else {
                for (v, &r) in x[..w].iter_mut().zip(row) {
                    *v = r as f32;
                }
            }
            forward_line(br, &x[..w], tw, &mut t[i * tw..(i + 1) * tw]);
        }
        // Columns, the tw columns as lanes: coefficient k of column j at
        // col[k * tw + j].
        forward_lanes(bc, &t[..h * tw], tw, th, col);
        let inv = 1.0 / gain(tx_sz);
        for k in 0..th {
            for j in 0..tw {
                out[k * w + j] = col[k * tw + j] as f64 * inv;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The AVX2 build of the transform gives exactly the portable one's
    /// coefficients.
    #[test]
    fn simd_matches_scalar() {
        for tx_sz in 0..TX_SIZES_ALL {
            let w = 1usize << TX_WIDTH_LOG2[tx_sz];
            let h = 1usize << TX_HEIGHT_LOG2[tx_sz];
            let res: Vec<i32> = (0..w * h)
                .map(|i| ((i * 7919 + i / w * 31) % 511) as i32 - 255)
                .collect();
            for tx_type in [DCT_DCT, IDTX, ADST_ADST, FLIPADST_DCT] {
                if (tx_type == ADST_ADST || tx_type == FLIPADST_DCT) && (w.max(h) > 16)
                    || (tx_type == IDTX && w.max(h) > 32)
                {
                    continue;
                }
                let mut a = vec![0.0; w * h];
                let mut b = vec![0.0; w * h];
                forward_2d(&res, tx_sz, tx_type, &mut a);
                let (mut t, mut col) = (vec![0.0; 64 * 32], vec![0.0; 32 * 32]);
                forward_2d_body(&res, tx_sz, tx_type, &mut b, &mut t, &mut col);
                assert_eq!(a, b, "tx {tx_sz} type {tx_type}");
            }
        }
    }
    use crate::dsp::itx::inverse_transform_2d;

    /// The 64-point sizes: a smooth residual (no energy above the 32
    /// lowest frequencies the syntax keeps) survives the round trip.
    #[test]
    fn round_trip_64() {
        for tx_sz in [TX_64X64, TX_32X64, TX_64X32, TX_16X64, TX_64X16] {
            let w = 1usize << TX_WIDTH_LOG2[tx_sz];
            let h = 1usize << TX_HEIGHT_LOG2[tx_sz];
            let res: Vec<i32> = (0..w * h)
                .map(|k| {
                    let (i, j) = ((k / w) as f64, (k % w) as f64);
                    (40.0 * (i / h as f64 * 3.0).sin() + 30.0 * (j / w as f64 * 2.0).cos()) as i32
                })
                .collect();
            let mut c = vec![0.0; w * h];
            forward_2d(&res, tx_sz, DCT_DCT, &mut c);
            let mut dq = vec![0i32; 64 * 64];
            for i in 0..h.min(32) {
                for j in 0..w.min(32) {
                    dq[i * 64 + j] = c[i * w + j].round() as i32;
                }
            }
            let mut out = vec![0i32; w * h];
            inverse_transform_2d(&dq, tx_sz, DCT_DCT, false, 8, &mut out);
            let err: i64 = out
                .iter()
                .zip(&res)
                .map(|(a, b)| ((a - b) as i64).abs())
                .max()
                .unwrap();
            assert!(err <= 3, "tx {tx_sz}: max error {err}");
        }
    }

    /// Forward then the normative inverse returns the residual, near enough.
    #[test]
    fn round_trip() {
        for tx_sz in [
            TX_4X4, TX_8X8, TX_16X16, TX_32X32, TX_8X16, TX_16X8, TX_4X8, TX_8X4,
        ] {
            for tx_type in [
                DCT_DCT,
                ADST_ADST,
                ADST_DCT,
                DCT_ADST,
                IDTX,
                FLIPADST_DCT,
                DCT_FLIPADST,
                FLIPADST_FLIPADST,
                V_DCT,
                H_FLIPADST,
            ] {
                if tx_sz == TX_32X32 && tx_type != DCT_DCT && tx_type != IDTX {
                    continue;
                }
                let flip_ud = matches!(
                    tx_type,
                    FLIPADST_DCT | FLIPADST_ADST | V_FLIPADST | FLIPADST_FLIPADST
                );
                let flip_lr = matches!(
                    tx_type,
                    DCT_FLIPADST | ADST_FLIPADST | H_FLIPADST | FLIPADST_FLIPADST
                );
                let w = 1usize << TX_WIDTH_LOG2[tx_sz];
                let h = 1usize << TX_HEIGHT_LOG2[tx_sz];
                let res: Vec<i32> = (0..w * h)
                    .map(|i| ((i * 37 + i / w * 11) % 61) as i32 - 30)
                    .collect();
                let mut c = vec![0.0; w * h];
                forward_2d(&res, tx_sz, tx_type, &mut c);
                let mut dq = vec![0i32; 64 * 64];
                for i in 0..h.min(32) {
                    for j in 0..w.min(32) {
                        dq[i * 64 + j] = c[i * w + j].round() as i32;
                    }
                }
                let mut out = vec![0i32; w * h];
                inverse_transform_2d(&dq, tx_sz, tx_type, false, 8, &mut out);
                // The reconstruction's flips.
                let err: i64 = (0..w * h)
                    .map(|k| {
                        let (i, j) = (k / w, k % w);
                        let si = if flip_ud { h - 1 - i } else { i };
                        let sj = if flip_lr { w - 1 - j } else { j };
                        ((out[si * w + sj] - res[k]) as i64).abs()
                    })
                    .max()
                    .unwrap();
                assert!(err <= 2, "tx {tx_sz} type {tx_type}: max error {err}");
            }
        }
    }
}

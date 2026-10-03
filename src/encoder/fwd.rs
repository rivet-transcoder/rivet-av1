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
/// divided by its squared length).
struct Basis {
    n: usize,
    f: Vec<f32>,
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
                v.push(Basis { n, f });
            }
        }
        v
    });
    &all[kind as usize * 7 + log2n as usize]
}

/// The forward 2D transform of a `w` x `h` residual (row-major) for
/// `tx_type` at `tx_sz`, as the coefficients the inverse process expects
/// in `Dequant` (before quantisation), `out[k * w + j]`; only the top-left
/// 32x32, which the syntax can code, is computed. The flipped types flip
/// the residual first, as the reconstruction flips it back (7.13.3).
pub(crate) fn forward_2d(residual: &[i32], tx_sz: usize, tx_type: usize, out: &mut [f64]) {
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
    // Rows: tmp[i][k] = sum_j res[i][j] * F[j][k], for the tw coefficients
    // kept (accumulated along j so the inner loop runs over k).
    let mut tmp = [0f32; 64 * 32];
    for i in 0..h {
        let si = if flip_ud { h - 1 - i } else { i };
        let acc = &mut tmp[i * tw..(i + 1) * tw];
        for j in 0..w {
            let sj = if flip_lr { w - 1 - j } else { j };
            let x = residual[si * w + sj] as f32;
            if x == 0.0 {
                continue;
            }
            let fr = &br.f[j * w..j * w + tw];
            for (a, &b) in acc.iter_mut().zip(fr) {
                *a += x * b;
            }
        }
    }
    // Columns: out[k][j] = sum_i tmp[i][j] * F[i][k].
    let rect = if log2w.abs_diff(log2h) == 1 {
        2896.0 / 4096.0
    } else {
        1.0
    };
    let gain = rect / (1u64 << (TRANSFORM_ROW_SHIFT[tx_sz] as u32 + 4)) as f64;
    let mut col = [0f32; 32 * 32];
    for i in 0..h {
        let fc = &bc.f[i * h..i * h + th];
        let t = &tmp[i * tw..(i + 1) * tw];
        for (k, &b) in fc.iter().enumerate() {
            let o = &mut col[k * tw..(k + 1) * tw];
            for (a, &x) in o.iter_mut().zip(t) {
                *a += x * b;
            }
        }
    }
    let inv = 1.0 / gain;
    for k in 0..th {
        for j in 0..tw {
            out[k * w + j] = col[k * tw + j] as f64 * inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

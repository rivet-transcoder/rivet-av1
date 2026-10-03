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

use crate::dsp::itx::{inverse_1d, kinds};
use crate::tables::{TRANSFORM_ROW_SHIFT, TX_HEIGHT_LOG2, TX_WIDTH_LOG2};

/// The basis of one 1D inverse kernel: `m[k * n + i]` is output sample `i`
/// for a unit coefficient `k`; `norm2[k]` is that vector's squared length.
struct Basis {
    n: usize,
    m: Vec<f64>,
    norm2: Vec<f64>,
}

fn basis(kind: u8, log2n: u32) -> &'static Basis {
    static CACHE: OnceLock<Vec<Basis>> = OnceLock::new();
    let all = CACHE.get_or_init(|| {
        let mut v = Vec::new();
        for kind in 0..3u8 {
            for log2n in 0..6u32 {
                let n = 1usize << log2n;
                let valid = log2n >= 2 && (kind != 1 || log2n <= 4) && (kind != 2 || log2n <= 5);
                let mut m = vec![0.0; n * n];
                let mut norm2 = vec![1.0; n];
                if valid {
                    const A: i32 = 1 << 16;
                    for k in 0..n {
                        let mut t = [0i32; 64];
                        t[k] = A;
                        inverse_1d(kind, log2n, &mut t[..n]);
                        let mut s = 0.0;
                        for i in 0..n {
                            let v = t[i] as f64 / A as f64;
                            m[k * n + i] = v;
                            s += v * v;
                        }
                        norm2[k] = s;
                    }
                }
                v.push(Basis { n, m, norm2 });
            }
        }
        v
    });
    &all[kind as usize * 6 + log2n as usize]
}

/// The forward 2D transform of a `w` x `h` residual (row-major) for
/// `tx_type` at `tx_sz`, as the coefficients the inverse process expects
/// in `Dequant` (before quantisation). Only sizes up to 32x32 are used.
pub(crate) fn forward_2d(residual: &[i32], tx_sz: usize, tx_type: usize, out: &mut [f64]) {
    let log2w = TX_WIDTH_LOG2[tx_sz] as u32;
    let log2h = TX_HEIGHT_LOG2[tx_sz] as u32;
    let w = 1usize << log2w;
    let h = 1usize << log2h;
    let (rk, ck) = kinds(tx_type);
    let br = basis(rk, log2w);
    let bc = basis(ck, log2h);
    debug_assert!(br.n == w && bc.n == h);
    // Rows: project each row on the row basis.
    let mut tmp = vec![0.0f64; w * h];
    for i in 0..h {
        for k in 0..w {
            let mut s = 0.0;
            for j in 0..w {
                s += residual[i * w + j] as f64 * br.m[k * w + j];
            }
            tmp[i * w + k] = s / br.norm2[k];
        }
    }
    // Columns.
    let rect = if log2w.abs_diff(log2h) == 1 { 2896.0 / 4096.0 } else { 1.0 };
    let gain = rect / (1u64 << (TRANSFORM_ROW_SHIFT[tx_sz] as u32 + 4)) as f64;
    for k in 0..h {
        for j in 0..w {
            let mut s = 0.0;
            for i in 0..h {
                s += tmp[i * w + j] * bc.m[k * h + i];
            }
            out[k * w + j] = s / bc.norm2[k] / gain;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consts::*;
    use crate::dsp::itx::inverse_transform_2d;

    /// Forward then the normative inverse returns the residual, near enough.
    #[test]
    fn round_trip() {
        for tx_sz in [TX_4X4, TX_8X8, TX_16X16, TX_32X32, TX_8X16, TX_16X8, TX_4X8, TX_8X4] {
            for tx_type in [DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST, IDTX] {
                if tx_sz == TX_32X32 && tx_type != DCT_DCT && tx_type != IDTX {
                    continue;
                }
                let w = 1usize << TX_WIDTH_LOG2[tx_sz];
                let h = 1usize << TX_HEIGHT_LOG2[tx_sz];
                let res: Vec<i32> = (0..w * h).map(|i| ((i * 37 + i / w * 11) % 61) as i32 - 30).collect();
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
                let err: i64 = out.iter().zip(&res).map(|(a, b)| ((a - b) as i64).abs()).max().unwrap();
                assert!(err <= 2, "tx {tx_sz} type {tx_type}: max error {err}");
            }
        }
    }
}

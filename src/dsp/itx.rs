//! The inverse transforms of 7.13: DCT 4..64, ADST 4/8/16, identity, the
//! Walsh-Hadamard transform, and the 2D inverse transform process, as the
//! butterfly steps of 7.13.2 (bit-exact).

use crate::consts::*;
use crate::tables::{COS128_LOOKUP, TRANSFORM_ROW_SHIFT, TX_HEIGHT_LOG2, TX_WIDTH_LOG2};

fn brev(num_bits: u32, x: usize) -> usize {
    let mut t = 0;
    for i in 0..num_bits {
        let bit = (x >> i) & 1;
        t += bit << (num_bits - 1 - i);
    }
    t
}

#[inline]
fn cos128(angle: i32) -> i64 {
    let a = (angle & 255) as usize;
    (if a <= 64 {
        COS128_LOOKUP[a]
    } else if a <= 128 {
        -COS128_LOOKUP[128 - a]
    } else if a <= 192 {
        -COS128_LOOKUP[a - 128]
    } else {
        COS128_LOOKUP[256 - a]
    }) as i64
}

#[inline]
fn sin128(angle: i32) -> i64 {
    cos128(angle - 64)
}

#[inline]
fn round2_12(x: i64) -> i32 {
    ((x + (1 << 11)) >> 12) as i32
}

/// The working array `T` and the butterflies `B` and `H` (7.13.2.1).
struct Tx<'a> {
    t: &'a mut [i32],
}

impl Tx<'_> {
    #[inline]
    fn b(&mut self, a: usize, b: usize, angle: i32, flip: bool) {
        let ta = self.t[a] as i64;
        let tb = self.t[b] as i64;
        let x = ta * cos128(angle) - tb * sin128(angle);
        let y = ta * sin128(angle) + tb * cos128(angle);
        self.t[a] = round2_12(x);
        self.t[b] = round2_12(y);
        if flip {
            self.t.swap(a, b);
        }
    }

    #[inline]
    fn h(&mut self, a: usize, b: usize, flip: bool, r: u32) {
        let (a, b) = if flip { (b, a) } else { (a, b) };
        let x = self.t[a] as i64;
        let y = self.t[b] as i64;
        let lo = -(1i64 << (r - 1));
        let hi = (1i64 << (r - 1)) - 1;
        self.t[a] = (x + y).clamp(lo, hi) as i32;
        self.t[b] = (x - y).clamp(lo, hi) as i32;
    }

    /// The inverse DCT array permutation (7.13.2.2).
    fn dct_permute(&mut self, n: u32) {
        let len = 1usize << n;
        let mut copy = [0i32; 64];
        copy[..len].copy_from_slice(&self.t[..len]);
        for i in 0..len {
            self.t[i] = copy[brev(n, i)];
        }
    }

    /// The inverse DCT process (7.13.2.3).
    fn idct(&mut self, n: u32, r: u32) {
        self.dct_permute(n);
        if n == 6 {
            for i in 0..16 {
                self.b(32 + i, 63 - i, 63 - 4 * brev(4, i) as i32, false);
            }
        }
        if n >= 5 {
            for i in 0..8 {
                self.b(16 + i, 31 - i, 6 + ((brev(3, 7 - i) as i32) << 3), false);
            }
        }
        if n == 6 {
            for i in 0..16 {
                self.h(32 + i * 2, 33 + i * 2, i & 1 != 0, r);
            }
        }
        if n >= 4 {
            for i in 0..4 {
                self.b(8 + i, 15 - i, 12 + ((brev(2, 3 - i) as i32) << 4), false);
            }
        }
        if n >= 5 {
            for i in 0..8 {
                self.h(16 + 2 * i, 17 + 2 * i, i & 1 != 0, r);
            }
        }
        if n == 6 {
            for i in 0..4 {
                for j in 0..2 {
                    self.b(
                        62 - i * 4 - j,
                        33 + i * 4 + j,
                        60 - 16 * brev(2, i) as i32 + 64 * j as i32,
                        true,
                    );
                }
            }
        }
        if n >= 3 {
            for i in 0..2 {
                self.b(4 + i, 7 - i, 56 - 32 * i as i32, false);
            }
        }
        if n >= 4 {
            for i in 0..4 {
                self.h(8 + 2 * i, 9 + 2 * i, i & 1 != 0, r);
            }
        }
        if n >= 5 {
            for i in 0..2 {
                for j in 0..2 {
                    self.b(
                        30 - 4 * i - j,
                        17 + 4 * i + j,
                        24 + ((j as i32) << 6) + ((1 - i as i32) << 5),
                        true,
                    );
                }
            }
        }
        if n == 6 {
            for i in 0..8 {
                for j in 0..2 {
                    self.h(32 + i * 4 + j, 35 + i * 4 - j, i & 1 != 0, r);
                }
            }
        }
        for i in 0..2 {
            self.b(2 * i, 2 * i + 1, 32 + 16 * i as i32, i == 0);
        }
        if n >= 3 {
            for i in 0..2 {
                self.h(4 + 2 * i, 5 + 2 * i, i != 0, r);
            }
        }
        if n >= 4 {
            for i in 0..2 {
                self.b(14 - i, 9 + i, 48 + 64 * i as i32, true);
            }
        }
        if n >= 5 {
            for i in 0..4 {
                for j in 0..2 {
                    self.h(16 + 4 * i + j, 19 + 4 * i - j, i & 1 != 0, r);
                }
            }
        }
        if n == 6 {
            for i in 0..2 {
                for j in 0..4 {
                    self.b(
                        61 - i * 8 - j,
                        34 + i * 8 + j,
                        56 - i as i32 * 32 + ((j >> 1) as i32) * 64,
                        true,
                    );
                }
            }
        }
        for i in 0..2 {
            self.h(i, 3 - i, false, r);
        }
        if n >= 3 {
            self.b(6, 5, 32, true);
        }
        if n >= 4 {
            for i in 0..2 {
                for j in 0..2 {
                    self.h(8 + 4 * i + j, 11 + 4 * i - j, i != 0, r);
                }
            }
        }
        if n >= 5 {
            for i in 0..4 {
                self.b(29 - i, 18 + i, 48 + ((i >> 1) as i32) * 64, true);
            }
        }
        if n == 6 {
            for i in 0..4 {
                for j in 0..4 {
                    self.h(32 + 8 * i + j, 39 + 8 * i - j, i & 1 != 0, r);
                }
            }
        }
        if n >= 3 {
            for i in 0..4 {
                self.h(i, 7 - i, false, r);
            }
        }
        if n >= 4 {
            for i in 0..2 {
                self.b(13 - i, 10 + i, 32, true);
            }
        }
        if n >= 5 {
            for i in 0..2 {
                for j in 0..4 {
                    self.h(16 + i * 8 + j, 23 + i * 8 - j, i != 0, r);
                }
            }
        }
        if n == 6 {
            for i in 0..8 {
                self.b(59 - i, 36 + i, if i < 4 { 48 } else { 112 }, true);
            }
        }
        if n >= 4 {
            for i in 0..8 {
                self.h(i, 15 - i, false, r);
            }
        }
        if n >= 5 {
            for i in 0..4 {
                self.b(27 - i, 20 + i, 32, true);
            }
        }
        if n == 6 {
            for i in 0..8 {
                self.h(32 + i, 47 - i, false, r);
                self.h(48 + i, 63 - i, true, r);
            }
        }
        if n >= 5 {
            for i in 0..16 {
                self.h(i, 31 - i, false, r);
            }
        }
        if n == 6 {
            for i in 0..8 {
                self.b(55 - i, 40 + i, 32, true);
            }
        }
        if n == 6 {
            for i in 0..32 {
                self.h(i, 63 - i, false, r);
            }
        }
    }

    /// The inverse ADST input array permutation (7.13.2.4).
    fn adst_in_permute(&mut self, n: u32) {
        let n0 = 1usize << n;
        let mut copy = [0i32; 16];
        copy[..n0].copy_from_slice(&self.t[..n0]);
        for i in 0..n0 {
            let idx = if i & 1 != 0 { i - 1 } else { n0 - i - 1 };
            self.t[i] = copy[idx];
        }
    }

    /// The inverse ADST output array permutation (7.13.2.5).
    fn adst_out_permute(&mut self, n: u32) {
        let n0 = 1usize << n;
        let mut copy = [0i32; 16];
        copy[..n0].copy_from_slice(&self.t[..n0]);
        for i in 0..n0 {
            let a = (i >> 3) & 1;
            let b = ((i >> 2) & 1) ^ ((i >> 3) & 1);
            let c = ((i >> 1) & 1) ^ ((i >> 2) & 1);
            let d = (i & 1) ^ ((i >> 1) & 1);
            let idx = ((d << 3) | (c << 2) | (b << 1) | a) >> (4 - n);
            self.t[i] = if i & 1 != 0 { -copy[idx] } else { copy[idx] };
        }
    }

    /// The inverse ADST4 process (7.13.2.6).
    fn iadst4(&mut self) {
        const SINPI_1_9: i64 = 1321;
        const SINPI_2_9: i64 = 2482;
        const SINPI_3_9: i64 = 3344;
        const SINPI_4_9: i64 = 3803;
        let t: [i64; 4] = [
            self.t[0] as i64,
            self.t[1] as i64,
            self.t[2] as i64,
            self.t[3] as i64,
        ];
        let mut s = [0i64; 7];
        s[0] = SINPI_1_9 * t[0];
        s[1] = SINPI_2_9 * t[0];
        s[2] = SINPI_3_9 * t[1];
        s[3] = SINPI_4_9 * t[2];
        s[4] = SINPI_1_9 * t[2];
        s[5] = SINPI_2_9 * t[3];
        s[6] = SINPI_4_9 * t[3];
        let a7 = t[0] - t[2];
        let b7 = a7 + t[3];
        s[0] += s[3];
        s[1] -= s[4];
        s[3] = s[2];
        s[2] = SINPI_3_9 * b7;
        s[0] += s[5];
        s[1] -= s[6];
        let x0 = s[0] + s[3];
        let x1 = s[1] + s[3];
        let x2 = s[2];
        let x3 = s[0] + s[1] - s[3];
        self.t[0] = round2_12(x0);
        self.t[1] = round2_12(x1);
        self.t[2] = round2_12(x2);
        self.t[3] = round2_12(x3);
    }

    /// The inverse ADST8 process (7.13.2.7).
    fn iadst8(&mut self, r: u32) {
        self.adst_in_permute(3);
        for i in 0..4 {
            self.b(2 * i, 2 * i + 1, 60 - 16 * i as i32, true);
        }
        for i in 0..4 {
            self.h(i, 4 + i, false, r);
        }
        for i in 0..2 {
            self.b(4 + 3 * i, 5 + i, 48 - 32 * i as i32, true);
        }
        for i in 0..2 {
            for j in 0..2 {
                self.h(4 * j + i, 2 + 4 * j + i, false, r);
            }
        }
        for i in 0..2 {
            self.b(2 + 4 * i, 3 + 4 * i, 32, true);
        }
        self.adst_out_permute(3);
    }

    /// The inverse ADST16 process (7.13.2.8).
    fn iadst16(&mut self, r: u32) {
        self.adst_in_permute(4);
        for i in 0..8 {
            self.b(2 * i, 2 * i + 1, 62 - 8 * i as i32, true);
        }
        for i in 0..8 {
            self.h(i, 8 + i, false, r);
        }
        for i in 0..2 {
            self.b(8 + 2 * i, 9 + 2 * i, 56 - 32 * i as i32, true);
            self.b(13 + 2 * i, 12 + 2 * i, 8 + 32 * i as i32, true);
        }
        for i in 0..4 {
            for j in 0..2 {
                self.h(8 * j + i, 4 + 8 * j + i, false, r);
            }
        }
        for i in 0..2 {
            for j in 0..2 {
                self.b(4 + 8 * j + 3 * i, 5 + 8 * j + i, 48 - 32 * i as i32, true);
            }
        }
        for i in 0..2 {
            for j in 0..4 {
                self.h(4 * j + i, 2 + 4 * j + i, false, r);
            }
        }
        for i in 0..4 {
            self.b(2 + 4 * i, 3 + 4 * i, 32, true);
        }
        self.adst_out_permute(4);
    }

    fn iadst(&mut self, n: u32, r: u32) {
        match n {
            2 => self.iadst4(),
            3 => self.iadst8(r),
            _ => self.iadst16(r),
        }
    }

    /// The inverse identity transforms (7.13.2.11 to 7.13.2.15).
    fn iidentity(&mut self, n: u32) {
        let len = 1usize << n;
        for v in self.t[..len].iter_mut() {
            *v = match n {
                2 => round2_12(*v as i64 * 5793),
                3 => v.wrapping_mul(2),
                4 => round2_12(*v as i64 * 11586),
                _ => v.wrapping_mul(4),
            };
        }
    }

    /// The inverse Walsh-Hadamard transform (7.13.2.10).
    fn iwht(&mut self, shift: u32) {
        let mut a = self.t[0] >> shift;
        let mut c = self.t[1] >> shift;
        let mut d = self.t[2] >> shift;
        let mut b = self.t[3] >> shift;
        a += c;
        d -= b;
        let e = (a - d) >> 1;
        b = e - b;
        c = e - c;
        a -= b;
        d += c;
        self.t[0] = a;
        self.t[1] = b;
        self.t[2] = c;
        self.t[3] = d;
    }
}

/// Which 1D transform a direction uses.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Dct,
    Adst,
    Identity,
}

fn row_kind(tx_type: usize) -> Kind {
    match tx_type {
        DCT_DCT | ADST_DCT | FLIPADST_DCT | H_DCT => Kind::Dct,
        DCT_ADST | ADST_ADST | DCT_FLIPADST | FLIPADST_FLIPADST | ADST_FLIPADST | FLIPADST_ADST
        | H_ADST | H_FLIPADST => Kind::Adst,
        _ => Kind::Identity,
    }
}

fn col_kind(tx_type: usize) -> Kind {
    match tx_type {
        DCT_DCT | DCT_ADST | DCT_FLIPADST | V_DCT => Kind::Dct,
        ADST_DCT | ADST_ADST | FLIPADST_DCT | FLIPADST_FLIPADST | ADST_FLIPADST | FLIPADST_ADST
        | V_ADST | V_FLIPADST => Kind::Adst,
        _ => Kind::Identity,
    }
}

fn run(t: &mut [i32], kind: Kind, n: u32, r: u32, lossless: bool, wht_shift: u32) {
    let mut tx = Tx { t };
    if lossless {
        tx.iwht(wht_shift);
        return;
    }
    match kind {
        Kind::Dct => tx.idct(n, r),
        Kind::Adst => tx.iadst(n, r),
        Kind::Identity => tx.iidentity(n),
    }
}

/// One 1D inverse transform, without intermediate clamping (`r` wide
/// enough): 0 DCT, 1 ADST, 2 identity, of length `1 << n`. The encoder
/// derives its forward transforms from these.
pub(crate) fn inverse_1d(kind: u8, n: u32, t: &mut [i32]) {
    let k = match kind {
        0 => Kind::Dct,
        1 => Kind::Adst,
        _ => Kind::Identity,
    };
    run(t, k, n, 30, false, 0);
}

/// Which 1D transforms a type applies: `(rows, columns)`, 0 DCT, 1 ADST,
/// 2 identity.
pub(crate) fn kinds(tx_type: usize) -> (u8, u8) {
    let f = |k: Kind| match k {
        Kind::Dct => 0,
        Kind::Adst => 1,
        Kind::Identity => 2,
    };
    (f(row_kind(tx_type)), f(col_kind(tx_type)))
}

/// The 2D inverse transform process (7.13.3). `dequant` is the
/// `Dequant` array with a row stride of 64 (only its top-left 32x32 can be
/// non-zero); the result is written to `residual` with a row stride of `w`.
pub(crate) fn inverse_transform_2d(
    dequant: &[i32],
    tx_sz: usize,
    tx_type: usize,
    lossless: bool,
    bit_depth: u32,
    residual: &mut [i32],
) {
    let log2w = TX_WIDTH_LOG2[tx_sz] as u32;
    let log2h = TX_HEIGHT_LOG2[tx_sz] as u32;
    let w = 1usize << log2w;
    let h = 1usize << log2h;
    let row_shift = if lossless {
        0
    } else {
        TRANSFORM_ROW_SHIFT[tx_sz] as u32
    };
    let col_shift = if lossless { 0 } else { 4 };
    let row_clamp = bit_depth + 8;
    let col_clamp = (bit_depth + 6).max(16);
    let rk = row_kind(tx_type);
    let ck = col_kind(tx_type);
    let rect = log2w.abs_diff(log2h) == 1;
    let mut t = [0i32; 64];
    for i in 0..h {
        let out = &mut residual[i * w..(i + 1) * w];
        if i >= 32 {
            out.fill(0);
            continue;
        }
        let src = &dequant[i * 64..i * 64 + 32];
        let nz = w.min(32);
        if !lossless && src[..nz].iter().all(|&v| v == 0) {
            out.fill(0);
            continue;
        }
        for j in 0..w {
            t[j] = if j < 32 { src[j] } else { 0 };
        }
        if rect {
            for v in t[..w].iter_mut() {
                *v = round2_12(*v as i64 * 2896);
            }
        }
        run(&mut t, rk, log2w, row_clamp, lossless, 2);
        for j in 0..w {
            out[j] = round2(t[j], row_shift);
        }
    }
    let lo = -(1i32 << (col_clamp - 1));
    let hi = (1i32 << (col_clamp - 1)) - 1;
    for v in residual[..w * h].iter_mut() {
        *v = (*v).clamp(lo, hi);
    }
    for j in 0..w {
        for i in 0..h {
            t[i] = residual[i * w + j];
        }
        run(&mut t, ck, log2h, col_clamp, lossless, 0);
        for i in 0..h {
            residual[i * w + j] = round2(t[i], col_shift);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A DC-only block reconstructs to a flat residual whose value is the
    /// DC scaled through both passes.
    #[test]
    fn dct_dc_is_flat() {
        for tx_sz in 0..TX_SIZES_ALL {
            let mut dq = vec![0i32; 64 * 64];
            dq[0] = 1024;
            let w = 1usize << TX_WIDTH_LOG2[tx_sz];
            let h = 1usize << TX_HEIGHT_LOG2[tx_sz];
            let mut res = vec![0i32; w * h];
            inverse_transform_2d(&dq, tx_sz, DCT_DCT, false, 8, &mut res);
            assert!(res.iter().all(|&v| v == res[0]), "tx {tx_sz}");
            assert!(res[0] > 0);
        }
    }

    /// The lossless WHT of a single DC coefficient is exactly invertible
    /// with the forward transform's scaling: a flat 4x4 block.
    #[test]
    fn wht_dc() {
        let mut dq = vec![0i32; 64 * 64];
        dq[0] = 4 * 16;
        let mut res = vec![0i32; 16];
        inverse_transform_2d(&dq, TX_4X4, DCT_DCT, true, 8, &mut res);
        assert!(res.iter().all(|&v| v == 4), "{res:?}");
    }
}

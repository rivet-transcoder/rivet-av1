//! The 8-tap interpolation filters of inter prediction (7.11.3.4), one
//! row at a time: scalar, and AVX2 versions chosen at run time (NEON
//! comes from the compiler: it is part of the aarch64 baseline and these
//! loops vectorise), bit-exact.

/// The horizontal filter: `out[c] = Round2(sum_t f[t] * src[c + t],
/// round0)`, `src` holding `out.len() + 7` samples.
#[inline]
pub(crate) fn filter_row(src: &[u16], f: &[i32; 8], round0: u32, out: &mut [i32]) {
    debug_assert!(src.len() >= out.len() + 7);
    #[cfg(target_arch = "x86_64")]
    if crate::dsp::avx2() && out.len() >= 8 {
        // SAFETY: AVX2 is available; lengths checked above.
        unsafe { filter_row_avx2(src, f, round0, out) };
        return;
    }
    filter_row_scalar(src, f, round0, out)
}

/// The vertical filter of output row `r` over the intermediate array
/// (`w` wide): `out[c] = Round2(sum_t f[t] * inter[r + t][c], round1)`.
#[inline]
pub(crate) fn filter_col(
    inter: &[i32],
    r: usize,
    w: usize,
    f: &[i32; 8],
    round1: u32,
    out: &mut [i32],
) {
    debug_assert!(inter.len() >= (r + 8) * w && out.len() == w);
    #[cfg(target_arch = "x86_64")]
    if crate::dsp::avx2() && w >= 8 {
        // SAFETY: AVX2 is available; lengths checked above.
        unsafe { filter_col_avx2(inter, r, w, f, round1, out) };
        return;
    }
    filter_col_scalar(inter, r, w, f, round1, out)
}

pub(crate) fn filter_row_scalar(src: &[u16], f: &[i32; 8], round0: u32, out: &mut [i32]) {
    let w = out.len();
    let add = (1 << round0) >> 1;
    if f[3] == 128 {
        for c in 0..w {
            out[c] = (128 * src[c + 3] as i32 + add) >> round0;
        }
        return;
    }
    let mut acc = [0i32; 128];
    let acc = &mut acc[..w];
    for t in 0..8 {
        let ft = f[t];
        let s = &src[t..t + w];
        for c in 0..w {
            acc[c] += ft * s[c] as i32;
        }
    }
    for c in 0..w {
        out[c] = (acc[c] + add) >> round0;
    }
}

pub(crate) fn filter_col_scalar(
    inter: &[i32],
    r: usize,
    w: usize,
    f: &[i32; 8],
    round1: u32,
    out: &mut [i32],
) {
    let add = (1 << round1) >> 1;
    if f[3] == 128 {
        let s = &inter[(r + 3) * w..(r + 4) * w];
        for c in 0..w {
            out[c] = (128 * s[c] + add) >> round1;
        }
        return;
    }
    let mut acc = [0i32; 128];
    let acc = &mut acc[..w];
    for t in 0..8 {
        let ft = f[t];
        let s = &inter[(r + t) * w..(r + t + 1) * w];
        for c in 0..w {
            acc[c] += ft * s[c];
        }
    }
    for c in 0..w {
        out[c] = (acc[c] + add) >> round1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn filter_row_avx2(src: &[u16], f: &[i32; 8], round0: u32, out: &mut [i32]) {
    use std::arch::x86_64::*;
    let w = out.len();
    let add = _mm256_set1_epi32((1 << round0) >> 1);
    let shift = _mm_cvtsi32_si128(round0 as i32);
    let taps: [__m256i; 8] = std::array::from_fn(|t| _mm256_set1_epi32(f[t]));
    let sp = src.as_ptr();
    let op = out.as_mut_ptr();
    let mut c = 0;
    // Whole vectors, then one more ending at the last column (overlapping
    // what is done: the same values are stored again).
    while c < w {
        let c0 = c.min(w - 8);
        let mut acc = _mm256_setzero_si256();
        if f[3] == 128 {
            // SAFETY: c0 + 3 + 8 <= w + 7 <= src.len().
            let s = unsafe { _mm_loadu_si128(sp.add(c0 + 3) as *const __m128i) };
            acc = _mm256_slli_epi32(_mm256_cvtepu16_epi32(s), 7);
        } else {
            for (t, tap) in taps.iter().enumerate() {
                // SAFETY: c0 + t + 8 <= w + 7 <= src.len().
                let s = unsafe { _mm_loadu_si128(sp.add(c0 + t) as *const __m128i) };
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(_mm256_cvtepu16_epi32(s), *tap));
            }
        }
        let v = _mm256_sra_epi32(_mm256_add_epi32(acc, add), shift);
        // SAFETY: c0 + 8 <= w.
        unsafe { _mm256_storeu_si256(op.add(c0) as *mut __m256i, v) };
        c = c0 + 8;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn filter_col_avx2(
    inter: &[i32],
    r: usize,
    w: usize,
    f: &[i32; 8],
    round1: u32,
    out: &mut [i32],
) {
    use std::arch::x86_64::*;
    let add = _mm256_set1_epi32((1 << round1) >> 1);
    let shift = _mm_cvtsi32_si128(round1 as i32);
    let taps: [__m256i; 8] = std::array::from_fn(|t| _mm256_set1_epi32(f[t]));
    let ip = inter.as_ptr();
    let op = out.as_mut_ptr();
    let mut c = 0;
    while c < w {
        let c0 = c.min(w - 8);
        let mut acc = _mm256_setzero_si256();
        if f[3] == 128 {
            // SAFETY: row r + 3 of the intermediate array.
            let s = unsafe { _mm256_loadu_si256(ip.add((r + 3) * w + c0) as *const __m256i) };
            acc = _mm256_slli_epi32(s, 7);
        } else {
            for (t, tap) in taps.iter().enumerate() {
                // SAFETY: rows r..r + 8 of the intermediate array.
                let s = unsafe { _mm256_loadu_si256(ip.add((r + t) * w + c0) as *const __m256i) };
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(s, *tap));
            }
        }
        let v = _mm256_sra_epi32(_mm256_add_epi32(acc, add), shift);
        // SAFETY: c0 + 8 <= w = out.len().
        unsafe { _mm256_storeu_si256(op.add(c0) as *mut __m256i, v) };
        c = c0 + 8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::SUBPEL_FILTERS;

    /// The SIMD filters agree with the scalar ones for every filter and
    /// phase, at several widths, on 8-, 10- and 12-bit samples.
    #[test]
    fn simd_matches_scalar() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % n
        };
        for bd in [8u32, 10, 12] {
            for filt in SUBPEL_FILTERS.iter() {
                for f in filt.iter() {
                    for w in [4usize, 8, 12, 16, 32, 64, 128] {
                        let src: Vec<u16> = (0..w + 7).map(|_| rnd(1 << bd) as u16).collect();
                        let round0 = if bd == 12 { 5 } else { 3 };
                        let mut a = vec![0i32; w];
                        let mut b = vec![0i32; w];
                        filter_row_scalar(&src, f, round0, &mut a);
                        filter_row(&src, f, round0, &mut b);
                        assert_eq!(a, b, "row {bd}-bit w {w} {f:?}");
                        let inter: Vec<i32> = (0..(w * 9))
                            .map(|_| rnd(1 << (bd + 4)) as i32 - (1 << (bd + 2)))
                            .collect();
                        for round1 in [7u32, 11, 9] {
                            filter_col_scalar(&inter, 1, w, f, round1, &mut a);
                            filter_col(&inter, 1, w, f, round1, &mut b);
                            assert_eq!(a, b, "col {bd}-bit w {w} {f:?}");
                        }
                    }
                }
            }
        }
    }
}

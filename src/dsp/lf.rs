//! The loop filter's sample filtering (7.14.6) for the four samples along
//! one edge at once: the decisions and every filter computed for each lane
//! without branches, then selected per lane. Written lane by lane in 32-bit
//! arithmetic, which the compiler turns into vector code (AVX2 where the
//! kernel is compiled for it, NEON on aarch64); bit-exact with the one-
//! sample process (tested).

/// The samples across an edge, four lanes: `v[7 + k]` is the k-th from the
/// edge (q0 at 7, p0 at 6), `v[k][lane]`.
pub(crate) type Edge = [[i32; 4]; 14];

/// The edge's filter parameters (7.14.4, 7.14.3).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Params {
    pub(crate) limit: i32,
    pub(crate) blimit: i32,
    pub(crate) thresh: i32,
    pub(crate) filter_size: usize,
    pub(crate) chroma: bool,
    pub(crate) bit_depth: u32,
}

/// Filters the four lanes of `v` in place.
#[inline]
pub(crate) fn filter(v: &mut Edge, p: &Params) {
    #[cfg(target_arch = "x86_64")]
    if crate::dsp::avx2() {
        // SAFETY: AVX2 is available.
        unsafe { filter_avx2(v, p) };
        return;
    }
    filter_lanes(v, p)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn filter_avx2(v: &mut Edge, p: &Params) {
    filter_lanes(v, p)
}

#[inline(always)]
fn abs4(a: [i32; 4], b: [i32; 4]) -> [i32; 4] {
    std::array::from_fn(|l| (a[l] - b[l]).abs())
}

#[inline(always)]
fn gt(a: [i32; 4], t: i32) -> [bool; 4] {
    std::array::from_fn(|l| a[l] > t)
}

#[inline(always)]
fn or(a: [bool; 4], b: [bool; 4]) -> [bool; 4] {
    std::array::from_fn(|l| a[l] | b[l])
}

/// The lane-by-lane process.
#[inline(always)]
pub(crate) fn filter_lanes(v: &mut Edge, p: &Params) {
    let sh = p.bit_depth - 8;
    let (q0, q1, q2, q3) = (v[7], v[8], v[9], v[10]);
    let (p0, p1, p2, p3) = (v[6], v[5], v[4], v[3]);
    let thresh_bd = p.thresh << sh;
    let limit_bd = p.limit << sh;
    let blimit_bd = p.blimit << sh;
    let filter_len = if p.filter_size == 4 {
        4
    } else if p.chroma {
        6
    } else if p.filter_size == 8 {
        8
    } else {
        16
    };
    // The filter mask process (7.14.6.2).
    let hev = or(gt(abs4(p1, p0), thresh_bd), gt(abs4(q1, q0), thresh_bd));
    let a = abs4(p0, q0);
    let b = abs4(p1, q1);
    let mut skip: [bool; 4] = std::array::from_fn(|l| a[l] * 2 + b[l] / 2 > blimit_bd);
    skip = or(
        skip,
        or(gt(abs4(p1, p0), limit_bd), gt(abs4(q1, q0), limit_bd)),
    );
    if filter_len >= 6 {
        skip = or(
            skip,
            or(gt(abs4(p2, p1), limit_bd), gt(abs4(q2, q1), limit_bd)),
        );
    }
    if filter_len >= 8 {
        skip = or(
            skip,
            or(gt(abs4(p3, p2), limit_bd), gt(abs4(q3, q2), limit_bd)),
        );
    }
    if skip == [true; 4] {
        return;
    }
    let t = 1 << sh;
    let mut flat = [false; 4];
    if p.filter_size >= 8 {
        let mut m = or(
            or(gt(abs4(p1, p0), t), gt(abs4(q1, q0), t)),
            or(gt(abs4(p2, p0), t), gt(abs4(q2, q0), t)),
        );
        if filter_len >= 8 {
            m = or(m, or(gt(abs4(p3, p0), t), gt(abs4(q3, q0), t)));
        }
        flat = std::array::from_fn(|l| !m[l]);
    }
    let mut flat2 = [false; 4];
    if p.filter_size >= 16 {
        let (q4, q5, q6) = (v[11], v[12], v[13]);
        let (p4, p5, p6) = (v[2], v[1], v[0]);
        let m = or(
            or(
                or(gt(abs4(p6, p0), t), gt(abs4(q6, q0), t)),
                or(gt(abs4(p5, p0), t), gt(abs4(q5, q0), t)),
            ),
            or(gt(abs4(p4, p0), t), gt(abs4(q4, q0), t)),
        );
        flat2 = std::array::from_fn(|l| !m[l]);
    }
    // The narrow filter (7.14.6.3), every lane.
    let bd = p.bit_depth;
    let lo = -(1 << (bd - 1));
    let hi = (1 << (bd - 1)) - 1;
    let c = |x: i32| x.clamp(lo, hi);
    let off = 0x80 << sh;
    let mut n_out = [[0i32; 4]; 4]; // p1, p0, q0, q1
    for l in 0..4 {
        let ps1 = p1[l] - off;
        let ps0 = p0[l] - off;
        let qs0 = q0[l] - off;
        let qs1 = q1[l] - off;
        let mut fl = if hev[l] { c(ps1 - qs1) } else { 0 };
        fl = c(fl + 3 * (qs0 - ps0));
        let filter1 = c(fl + 4) >> 3;
        let filter2 = c(fl + 3) >> 3;
        n_out[2][l] = c(qs0 - filter1) + off;
        n_out[1][l] = c(ps0 + filter2) + off;
        if hev[l] {
            n_out[0][l] = p1[l];
            n_out[3][l] = q1[l];
        } else {
            let f = (filter1 + 1) >> 1;
            n_out[3][l] = c(qs1 - f) + off;
            n_out[0][l] = c(ps1 + f) + off;
        }
    }
    // The wide filters (7.14.6.4): log2 size 3 (n = 3 luma, 2 chroma) and 4
    // (n = 6), every lane, for the sizes this edge can use.
    let wide = |log2_size: u32, v: &Edge| -> [[i32; 4]; 12] {
        let n: isize = if log2_size == 4 {
            6
        } else if !p.chroma {
            3
        } else {
            2
        };
        let n2: isize = if log2_size == 3 && !p.chroma { 0 } else { 1 };
        let mut out = [[0i32; 4]; 12];
        for i in -n..n {
            let mut acc = [0i32; 4];
            for j in -n..=n {
                let pp = (i + j).clamp(-(n + 1), n);
                let tap = if j.abs() <= n2 { 2 } else { 1 };
                let s = &v[(7 + pp) as usize];
                for l in 0..4 {
                    acc[l] += s[l] * tap;
                }
            }
            let add = 1 << (log2_size - 1);
            for l in 0..4 {
                out[(i + 6) as usize][l] = (acc[l] + add) >> log2_size;
            }
        }
        out
    };
    let w8 = if p.filter_size >= 8 {
        Some(wide(3, v))
    } else {
        None
    };
    let w16 = if p.filter_size >= 16 {
        Some(wide(4, v))
    } else {
        None
    };
    let n8: isize = if p.chroma { 2 } else { 3 };
    for l in 0..4 {
        if skip[l] {
            continue;
        }
        if p.filter_size == 4 || !flat[l] {
            v[5][l] = n_out[0][l];
            v[6][l] = n_out[1][l];
            v[7][l] = n_out[2][l];
            v[8][l] = n_out[3][l];
        } else if p.filter_size == 8 || !flat2[l] {
            let w = w8.as_ref().expect("filter size 8 or more");
            for i in -n8..n8 {
                v[(7 + i) as usize][l] = w[(i + 6) as usize][l];
            }
        } else {
            let w = w16.as_ref().expect("filter size 16");
            for i in -6isize..6 {
                v[(7 + i) as usize][l] = w[(i + 6) as usize][l];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference: the one-sample process of 7.14.6, as written.
    fn reference(v: &mut [i32; 14], p: &Params) {
        let sh = p.bit_depth - 8;
        let (q0, q1, q2, q3) = (v[7], v[8], v[9], v[10]);
        let (p0, p1, p2, p3) = (v[6], v[5], v[4], v[3]);
        let thresh_bd = p.thresh << sh;
        let hev_mask = (p1 - p0).abs() > thresh_bd || (q1 - q0).abs() > thresh_bd;
        let filter_len = if p.filter_size == 4 {
            4
        } else if p.chroma {
            6
        } else if p.filter_size == 8 {
            8
        } else {
            16
        };
        let limit_bd = p.limit << sh;
        let blimit_bd = p.blimit << sh;
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
        let t = 1 << sh;
        let mut flat = false;
        if p.filter_size >= 8 {
            let mut m = (p1 - p0).abs() > t
                || (q1 - q0).abs() > t
                || (p2 - p0).abs() > t
                || (q2 - q0).abs() > t;
            if filter_len >= 8 {
                m |= (p3 - p0).abs() > t || (q3 - q0).abs() > t;
            }
            flat = !m;
        }
        let mut flat2 = false;
        if p.filter_size >= 16 {
            let m = (v[0] - p0).abs() > t
                || (v[13] - q0).abs() > t
                || (v[1] - p0).abs() > t
                || (v[12] - q0).abs() > t
                || (v[2] - p0).abs() > t
                || (v[11] - q0).abs() > t;
            flat2 = !m;
        }
        if p.filter_size == 4 || !flat {
            let bd = p.bit_depth;
            let c = |x: i32| x.clamp(-(1 << (bd - 1)), (1 << (bd - 1)) - 1);
            let off = 0x80 << sh;
            let (ps1, ps0, qs0, qs1) = (p1 - off, p0 - off, q0 - off, q1 - off);
            let mut fl = if hev_mask { c(ps1 - qs1) } else { 0 };
            fl = c(fl + 3 * (qs0 - ps0));
            let filter1 = c(fl + 4) >> 3;
            let filter2 = c(fl + 3) >> 3;
            v[7] = c(qs0 - filter1) + off;
            v[6] = c(ps0 + filter2) + off;
            if !hev_mask {
                let f = (filter1 + 1) >> 1;
                v[8] = c(qs1 - f) + off;
                v[5] = c(ps1 + f) + off;
            }
        } else {
            let log2_size = if p.filter_size == 8 || !flat2 { 3 } else { 4 };
            let n: isize = if log2_size == 4 {
                6
            } else if !p.chroma {
                3
            } else {
                2
            };
            let n2: isize = if log2_size == 3 && !p.chroma { 0 } else { 1 };
            let orig = *v;
            for i in -n..n {
                let mut t = 0;
                for j in -n..=n {
                    let pp = (i + j).clamp(-(n + 1), n);
                    let tap = if j.abs() <= n2 { 2 } else { 1 };
                    t += orig[(7 + pp) as usize] * tap;
                }
                v[(7 + i) as usize] = (t + (1 << (log2_size - 1))) >> log2_size;
            }
        }
    }

    /// Every edge kind, random samples near and across the thresholds.
    #[test]
    fn lanes_match_the_process() {
        let mut seed = 0x0bad_5eed_1234_5678u64;
        let mut rnd = |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % n
        };
        for case in 0..20000 {
            let bd = [8u32, 10, 12][case % 3];
            let params = Params {
                limit: 1 + rnd(9) as i32,
                blimit: rnd(130) as i32,
                thresh: rnd(4) as i32,
                filter_size: [4, 8, 16, 6][rnd(4) as usize].min(if case % 5 == 0 { 8 } else { 16 }),
                chroma: case % 5 == 0,
                bit_depth: bd,
            };
            let mut p = params;
            if p.chroma {
                p.filter_size = p.filter_size.min(8);
            }
            let base = rnd(1 << bd) as i32;
            let spread = [1u64, 4, 16, 64][rnd(4) as usize] << (bd - 8);
            let mut lanes: Edge = [[0; 4]; 14];
            for k in 0..14 {
                for l in 0..4 {
                    lanes[k][l] =
                        (base + rnd(spread) as i32 - (spread / 2) as i32).clamp(0, (1 << bd) - 1);
                }
            }
            let mut want = lanes;
            for l in 0..4 {
                let mut one: [i32; 14] = std::array::from_fn(|k| lanes[k][l]);
                reference(&mut one, &p);
                for k in 0..14 {
                    want[k][l] = one[k];
                }
            }
            let mut got = lanes;
            filter(&mut got, &p);
            assert_eq!(got, want, "case {case} {p:?}");
        }
    }
}

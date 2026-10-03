//! The film grain synthesis process (7.18.3), applied to output frames.

use crate::consts::*;
use crate::header::FilmGrainParams;
use crate::obu::SequenceHeader;
use crate::tables::GAUSSIAN_SEQUENCE;

struct Rng(u32);

impl Rng {
    /// The random number process (7.18.3.2).
    fn get(&mut self, bits: u32) -> i32 {
        let r = self.0;
        let bit = ((r) ^ (r >> 1) ^ (r >> 3) ^ (r >> 12)) & 1;
        let r = (r >> 1) | (bit << 15);
        self.0 = r;
        ((r >> (16 - bits)) & ((1 << bits) - 1)) as i32
    }
}

/// Adds film grain to the output planes (each tightly packed, luma `w` x
/// `h`, chroma subsampled and rounded up).
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply(
    seq: &SequenceHeader,
    bit_depth: u32,
    sub_x: usize,
    sub_y: usize,
    w: usize,
    h: usize,
    g: &FilmGrainParams,
    planes: &mut [Vec<u16>],
) {
    let bd = bit_depth as i32;
    let mono = seq.color.mono_chrome;
    let num_planes = if mono { 1 } else { 3 };
    let grain_center = 128i32 << (bd - 8);
    let grain_min = -grain_center;
    let grain_max = (256 << (bd - 8)) - 1 - grain_center;
    let mut rng = Rng(g.grain_seed);

    // The generate grain process (7.18.3.3).
    let mut luma_grain = vec![[0i32; 82]; 73];
    let shift = (12 - bd + g.grain_scale_shift as i32) as u32;
    for row in luma_grain.iter_mut() {
        for v in row.iter_mut() {
            let gv = if g.num_y_points > 0 {
                GAUSSIAN_SEQUENCE[rng.get(11) as usize]
            } else {
                0
            };
            *v = round2(gv, shift);
        }
    }
    let ar_shift = g.ar_coeff_shift_minus_6 + 6;
    let lag = g.ar_coeff_lag;
    for y in 3..73 {
        for x in 3..(82 - 3) {
            let mut s = 0;
            let mut pos = 0;
            'outer: for dr in -lag..=0 {
                for dc in -lag..=lag {
                    if dr == 0 && dc == 0 {
                        break 'outer;
                    }
                    let c = g.ar_coeffs_y_plus_128[pos] - 128;
                    s += luma_grain[(y as i32 + dr) as usize][(x as i32 + dc) as usize] * c;
                    pos += 1;
                }
            }
            luma_grain[y][x] = clip3(grain_min, grain_max, luma_grain[y][x] + round2(s, ar_shift));
        }
    }
    let chroma_w = if sub_x != 0 { 44 } else { 82 };
    let chroma_h = if sub_y != 0 { 38 } else { 73 };
    let mut cb_grain = vec![[0i32; 82]; 73];
    let mut cr_grain = vec![[0i32; 82]; 73];
    if !mono {
        rng = Rng(g.grain_seed ^ 0xb524);
        for row in cb_grain.iter_mut().take(chroma_h) {
            for v in row.iter_mut().take(chroma_w) {
                let gv = if g.num_cb_points > 0 || g.chroma_scaling_from_luma {
                    GAUSSIAN_SEQUENCE[rng.get(11) as usize]
                } else {
                    0
                };
                *v = round2(gv, shift);
            }
        }
        rng = Rng(g.grain_seed ^ 0x49d8);
        for row in cr_grain.iter_mut().take(chroma_h) {
            for v in row.iter_mut().take(chroma_w) {
                let gv = if g.num_cr_points > 0 || g.chroma_scaling_from_luma {
                    GAUSSIAN_SEQUENCE[rng.get(11) as usize]
                } else {
                    0
                };
                *v = round2(gv, shift);
            }
        }
        for y in 3..chroma_h {
            for x in 3..(chroma_w - 3) {
                let mut s0 = 0;
                let mut s1 = 0;
                let mut pos = 0;
                'outer2: for dr in -lag..=0 {
                    for dc in -lag..=lag {
                        let c0 = g.ar_coeffs_cb_plus_128[pos] - 128;
                        let c1 = g.ar_coeffs_cr_plus_128[pos] - 128;
                        if dr == 0 && dc == 0 {
                            if g.num_y_points > 0 {
                                let mut luma = 0;
                                let luma_x = ((x - 3) << sub_x) + 3;
                                let luma_y = ((y - 3) << sub_y) + 3;
                                for i in 0..=sub_y {
                                    for j in 0..=sub_x {
                                        luma += luma_grain[luma_y + i][luma_x + j];
                                    }
                                }
                                luma = round2(luma, (sub_x + sub_y) as u32);
                                s0 += luma * c0;
                                s1 += luma * c1;
                            }
                            break 'outer2;
                        }
                        s0 += cb_grain[(y as i32 + dr) as usize][(x as i32 + dc) as usize] * c0;
                        s1 += cr_grain[(y as i32 + dr) as usize][(x as i32 + dc) as usize] * c1;
                        pos += 1;
                    }
                }
                cb_grain[y][x] = clip3(grain_min, grain_max, cb_grain[y][x] + round2(s0, ar_shift));
                cr_grain[y][x] = clip3(grain_min, grain_max, cr_grain[y][x] + round2(s1, ar_shift));
            }
        }
    }

    // The scaling lookup initialization process (7.18.3.4).
    let mut scaling_lut = [[0i32; 256]; 3];
    for (plane, lut) in scaling_lut.iter_mut().enumerate().take(num_planes) {
        let (num_points, xs, ys) = if plane == 0 || g.chroma_scaling_from_luma {
            (g.num_y_points, &g.point_y_value, &g.point_y_scaling)
        } else if plane == 1 {
            (g.num_cb_points, &g.point_cb_value, &g.point_cb_scaling)
        } else {
            (g.num_cr_points, &g.point_cr_value, &g.point_cr_scaling)
        };
        if num_points == 0 {
            continue;
        }
        for x in 0..xs[0] as usize {
            lut[x] = ys[0];
        }
        for i in 0..num_points - 1 {
            let delta_y = ys[i + 1] - ys[i];
            let delta_x = xs[i + 1] - xs[i];
            let delta = delta_y * ((65536 + (delta_x >> 1)) / delta_x);
            for x in 0..delta_x {
                let v = ys[i] + ((x * delta + 32768) >> 16);
                lut[(xs[i] + x) as usize] = v;
            }
        }
        for x in xs[num_points - 1] as usize..256 {
            lut[x] = ys[num_points - 1];
        }
    }
    let scale_lut = |plane: usize, index: i32| -> i32 {
        let shift = bd - 8;
        let x = index >> shift;
        let rem = index - (x << shift);
        if bd == 8 || x == 255 {
            scaling_lut[plane][x as usize]
        } else {
            let start = scaling_lut[plane][x as usize];
            let end = scaling_lut[plane][x as usize + 1];
            start + round2((end - start) * rem, shift as u32)
        }
    };

    // The add noise synthesis process (7.18.3.5).
    let stripes = h.div_ceil(2).div_ceil(16) + 1;
    let chroma_cols = (w + sub_x) >> sub_x;
    let noise_w = [w + 34, chroma_cols + 34, chroma_cols + 34];
    let mut noise_stripe: Vec<[Vec<i32>; 3]> = Vec::with_capacity(stripes);
    let mut luma_num = 0;
    let mut y = 0;
    while y < h.div_ceil(2) {
        let mut ns = [
            vec![0i32; 34 * noise_w[0]],
            vec![0i32; 34 * noise_w[1]],
            vec![0i32; 34 * noise_w[2]],
        ];
        rng = Rng(g.grain_seed);
        rng.0 ^= (((luma_num * 37 + 178) & 255) << 8) as u32;
        rng.0 ^= ((luma_num * 173 + 105) & 255) as u32;
        let mut x = 0;
        while x < w.div_ceil(2) {
            let rand = rng.get(8);
            let offset_x = (rand >> 4) as usize;
            let offset_y = (rand & 15) as usize;
            for plane in 0..num_planes {
                let psx = if plane > 0 { sub_x } else { 0 };
                let psy = if plane > 0 { sub_y } else { 0 };
                let pox = if psx != 0 {
                    6 + offset_x
                } else {
                    9 + offset_x * 2
                };
                let poy = if psy != 0 {
                    6 + offset_y
                } else {
                    9 + offset_y * 2
                };
                let nw = noise_w[plane];
                for i in 0..(34 >> psy) {
                    for j in 0..(34 >> psx) {
                        let mut gv = match plane {
                            0 => luma_grain[poy + i][pox + j],
                            1 => cb_grain[poy + i][pox + j],
                            _ => cr_grain[poy + i][pox + j],
                        };
                        if psx == 0 {
                            let idx = i * nw + x * 2 + j;
                            if j < 2 && g.overlap_flag && x > 0 {
                                let old = ns[plane][idx];
                                gv = if j == 0 {
                                    old * 27 + gv * 17
                                } else {
                                    old * 17 + gv * 27
                                };
                                gv = clip3(grain_min, grain_max, round2(gv, 5));
                            }
                            ns[plane][idx] = gv;
                        } else {
                            let idx = i * nw + x + j;
                            if j == 0 && g.overlap_flag && x > 0 {
                                let old = ns[plane][idx];
                                gv = old * 23 + gv * 22;
                                gv = clip3(grain_min, grain_max, round2(gv, 5));
                            }
                            ns[plane][idx] = gv;
                        }
                    }
                }
            }
            x += 16;
        }
        noise_stripe.push(ns);
        luma_num += 1;
        y += 16;
    }
    let mut noise_image: [Vec<i32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for plane in 0..num_planes {
        let psx = if plane > 0 { sub_x } else { 0 };
        let psy = if plane > 0 { sub_y } else { 0 };
        let pw = (w + psx) >> psx;
        let ph = (h + psy) >> psy;
        let nw = noise_w[plane];
        let mut img = vec![0i32; pw * ph];
        for yy in 0..ph {
            let luma_num = yy >> (5 - psy);
            let i = yy - (luma_num << (5 - psy));
            for xx in 0..pw {
                let mut gv = noise_stripe[luma_num][plane][i * nw + xx];
                if psy == 0 {
                    if i < 2 && luma_num > 0 && g.overlap_flag {
                        let old = noise_stripe[luma_num - 1][plane][(i + 32) * nw + xx];
                        gv = if i == 0 {
                            old * 27 + gv * 17
                        } else {
                            old * 17 + gv * 27
                        };
                        gv = clip3(grain_min, grain_max, round2(gv, 5));
                    }
                } else if i < 1 && luma_num > 0 && g.overlap_flag {
                    let old = noise_stripe[luma_num - 1][plane][(i + 16) * nw + xx];
                    gv = old * 23 + gv * 22;
                    gv = clip3(grain_min, grain_max, round2(gv, 5));
                }
                img[yy * pw + xx] = gv;
            }
        }
        noise_image[plane] = img;
    }
    let (min_value, max_luma, max_chroma) = if g.clip_to_restricted_range {
        let min_value = 16 << (bd - 8);
        let max_luma = 235 << (bd - 8);
        let max_chroma = if seq.color.matrix_coefficients == MC_IDENTITY {
            max_luma
        } else {
            240 << (bd - 8)
        };
        (min_value, max_luma, max_chroma)
    } else {
        (0, (256 << (bd - 8)) - 1, (256 << (bd - 8)) - 1)
    };
    let scaling_shift = g.grain_scaling_minus_8 + 8;
    let maxv = (1i32 << bd) - 1;
    if num_planes > 1 {
        let cw = (w + sub_x) >> sub_x;
        let ch = (h + sub_y) >> sub_y;
        for yy in 0..ch {
            for xx in 0..cw {
                let luma_x = xx << sub_x;
                let luma_y = yy << sub_y;
                let luma_next_x = (luma_x + 1).min(w - 1);
                let average_luma = if sub_x != 0 {
                    round2(
                        planes[0][luma_y * w + luma_x] as i32
                            + planes[0][luma_y * w + luma_next_x] as i32,
                        1,
                    )
                } else {
                    planes[0][luma_y * w + luma_x] as i32
                };
                if g.num_cb_points > 0 || g.chroma_scaling_from_luma {
                    let orig = planes[1][yy * cw + xx] as i32;
                    let merged = if g.chroma_scaling_from_luma {
                        average_luma
                    } else {
                        let combined =
                            average_luma * (g.cb_luma_mult - 128) + orig * (g.cb_mult - 128);
                        ((combined >> 6) + ((g.cb_offset - 256) << (bd - 8))).clamp(0, maxv)
                    };
                    let noise = noise_image[1][yy * cw + xx];
                    let noise = round2(scale_lut(1, merged) * noise, scaling_shift);
                    planes[1][yy * cw + xx] = clip3(min_value, max_chroma, orig + noise) as u16;
                }
                if g.num_cr_points > 0 || g.chroma_scaling_from_luma {
                    let orig = planes[2][yy * cw + xx] as i32;
                    let merged = if g.chroma_scaling_from_luma {
                        average_luma
                    } else {
                        let combined =
                            average_luma * (g.cr_luma_mult - 128) + orig * (g.cr_mult - 128);
                        ((combined >> 6) + ((g.cr_offset - 256) << (bd - 8))).clamp(0, maxv)
                    };
                    let noise = noise_image[2][yy * cw + xx];
                    let noise = round2(scale_lut(2, merged) * noise, scaling_shift);
                    planes[2][yy * cw + xx] = clip3(min_value, max_chroma, orig + noise) as u16;
                }
            }
        }
    }
    if g.num_y_points > 0 {
        for yy in 0..h {
            for xx in 0..w {
                let orig = planes[0][yy * w + xx] as i32;
                let noise = noise_image[0][yy * w + xx];
                let noise = round2(scale_lut(0, orig) * noise, scaling_shift);
                planes[0][yy * w + xx] = clip3(min_value, max_luma, orig + noise) as u16;
            }
        }
    }
}

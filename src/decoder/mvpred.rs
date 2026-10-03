//! Motion vector prediction (7.10.2), overlappable candidates (7.10.3) and
//! the warp sample search (7.10.4).

use crate::consts::*;
use crate::decoder::state::Mv;
use crate::decoder::tile::TileDecoder;
use crate::tables::*;

impl TileDecoder<'_, '_> {
    /// The find MV stack process (7.10.2).
    pub(crate) fn find_mv_stack(&mut self, is_compound: bool) {
        let ms = self.b.mi_size;
        let bw4 = NUM_4X4_BLOCKS_WIDE[ms];
        let bh4 = NUM_4X4_BLOCKS_HIGH[ms];
        self.b.num_mv_found = 0;
        self.b.new_mv_count = 0;
        self.b.global_mvs[0] = self.setup_global_mv(0);
        if is_compound {
            self.b.global_mvs[1] = self.setup_global_mv(1);
        }
        self.b.found_match = false;
        self.scan_row(-1, is_compound);
        let mut found_above_match = self.b.found_match;
        self.b.found_match = false;
        self.scan_col(-1, is_compound);
        let mut found_left_match = self.b.found_match;
        self.b.found_match = false;
        if bw4.max(bh4) <= 16 {
            self.scan_point(-1, bw4 as isize, is_compound);
        }
        if self.b.found_match {
            found_above_match = true;
        }
        self.b.close_matches = found_above_match as usize + found_left_match as usize;
        let num_nearest = self.b.num_mv_found;
        let num_new = self.b.new_mv_count;
        if num_nearest > 0 {
            for idx in 0..num_nearest {
                self.b.weight_stack[idx] += REF_CAT_LEVEL;
            }
        }
        self.b.zero_mv_context = 0;
        if self.f.hdr.use_ref_frame_mvs {
            self.temporal_scan(is_compound);
        }
        self.scan_point(-1, -1, is_compound);
        if self.b.found_match {
            found_above_match = true;
        }
        self.b.found_match = false;
        self.scan_row(-3, is_compound);
        if self.b.found_match {
            found_above_match = true;
        }
        self.b.found_match = false;
        self.scan_col(-3, is_compound);
        if self.b.found_match {
            found_left_match = true;
        }
        self.b.found_match = false;
        if bh4 > 1 {
            self.scan_row(-5, is_compound);
        }
        if self.b.found_match {
            found_above_match = true;
        }
        self.b.found_match = false;
        if bw4 > 1 {
            self.scan_col(-5, is_compound);
        }
        if self.b.found_match {
            found_left_match = true;
        }
        self.b.total_matches = found_above_match as usize + found_left_match as usize;
        self.sort_stack(0, num_nearest, is_compound);
        self.sort_stack(num_nearest, self.b.num_mv_found, is_compound);
        if self.b.num_mv_found < 2 {
            self.extra_search(is_compound);
        }
        self.context_and_clamping(is_compound, num_new);
    }

    /// The setup global MV process (7.10.2.1).
    fn setup_global_mv(&self, ref_list: usize) -> Mv {
        let rf = self.b.ref_frame[ref_list];
        let mut mv;
        let typ = if rf != INTRA_FRAME {
            self.f.hdr.gm_type[rf as usize]
        } else {
            IDENTITY
        };
        if rf == INTRA_FRAME || typ == IDENTITY {
            mv = [0, 0];
        } else if typ == TRANSLATION {
            let gm = &self.f.hdr.gm_params[rf as usize];
            mv = [
                gm[0] >> (WARPEDMODEL_PREC_BITS - 3),
                gm[1] >> (WARPEDMODEL_PREC_BITS - 3),
            ];
        } else {
            let gm = &self.f.hdr.gm_params[rf as usize];
            let ms = self.b.mi_size;
            let x = (self.b.mi_col * MI_SIZE + block_width(ms) / 2) as i64 - 1;
            let y = (self.b.mi_row * MI_SIZE + block_height(ms) / 2) as i64 - 1;
            let xc = (gm[2] as i64 - (1 << WARPEDMODEL_PREC_BITS)) * x + gm[3] as i64 * y + gm[0] as i64;
            let yc = gm[4] as i64 * x + (gm[5] as i64 - (1 << WARPEDMODEL_PREC_BITS)) * y + gm[1] as i64;
            if self.f.hdr.allow_high_precision_mv {
                mv = [
                    round2signed_64(yc, WARPEDMODEL_PREC_BITS - 3) as i32,
                    round2signed_64(xc, WARPEDMODEL_PREC_BITS - 3) as i32,
                ];
            } else {
                mv = [
                    round2signed_64(yc, WARPEDMODEL_PREC_BITS - 2) as i32 * 2,
                    round2signed_64(xc, WARPEDMODEL_PREC_BITS - 2) as i32 * 2,
                ];
            }
        }
        self.lower_mv_precision(&mut mv);
        mv
    }

    /// The lower precision process (7.10.2.10).
    pub(crate) fn lower_mv_precision(&self, mv: &mut Mv) {
        if self.f.hdr.allow_high_precision_mv {
            return;
        }
        for v in mv.iter_mut() {
            if self.f.hdr.force_integer_mv {
                let a = v.abs();
                let a_int = (a + 3) >> 3;
                *v = if *v > 0 { a_int << 3 } else { -(a_int << 3) };
            } else if *v & 1 != 0 {
                if *v > 0 {
                    *v -= 1;
                } else {
                    *v += 1;
                }
            }
        }
    }

    fn scan_row(&mut self, delta_row: isize, is_compound: bool) {
        let ms = self.b.mi_size;
        let bw4 = NUM_4X4_BLOCKS_WIDE[ms];
        let end4 = bw4.min(self.f.mi_cols - self.b.mi_col).min(16);
        let mut delta_col: isize = 0;
        let use_step16 = bw4 >= 16;
        let mut delta_row = delta_row;
        if delta_row.abs() > 1 {
            delta_row += (self.b.mi_row & 1) as isize;
            delta_col = 1 - (self.b.mi_col & 1) as isize;
        }
        let mut i = 0usize;
        while i < end4 {
            let mv_row = self.b.mi_row as isize + delta_row;
            let mv_col = self.b.mi_col as isize + delta_col + i as isize;
            if !self.is_inside(mv_row, mv_col) {
                break;
            }
            let cand_size = self.mi(mv_row as usize, mv_col as usize).mi_size as usize;
            let mut len = bw4.min(NUM_4X4_BLOCKS_WIDE[cand_size]);
            if delta_row.abs() > 1 {
                len = len.max(2);
            }
            if use_step16 {
                len = len.max(4);
            }
            let weight = len as u32 * 2;
            self.add_ref_mv_candidate(mv_row as usize, mv_col as usize, is_compound, weight);
            i += len;
        }
    }

    fn scan_col(&mut self, delta_col: isize, is_compound: bool) {
        let ms = self.b.mi_size;
        let bh4 = NUM_4X4_BLOCKS_HIGH[ms];
        let end4 = bh4.min(self.f.mi_rows - self.b.mi_row).min(16);
        let mut delta_row: isize = 0;
        let use_step16 = bh4 >= 16;
        let mut delta_col = delta_col;
        if delta_col.abs() > 1 {
            delta_row = 1 - (self.b.mi_row & 1) as isize;
            delta_col += (self.b.mi_col & 1) as isize;
        }
        let mut i = 0usize;
        while i < end4 {
            let mv_row = self.b.mi_row as isize + delta_row + i as isize;
            let mv_col = self.b.mi_col as isize + delta_col;
            if !self.is_inside(mv_row, mv_col) {
                break;
            }
            let cand_size = self.mi(mv_row as usize, mv_col as usize).mi_size as usize;
            let mut len = bh4.min(NUM_4X4_BLOCKS_HIGH[cand_size]);
            if delta_col.abs() > 1 {
                len = len.max(2);
            }
            if use_step16 {
                len = len.max(4);
            }
            let weight = len as u32 * 2;
            self.add_ref_mv_candidate(mv_row as usize, mv_col as usize, is_compound, weight);
            i += len;
        }
    }

    fn scan_point(&mut self, delta_row: isize, delta_col: isize, is_compound: bool) {
        let mv_row = self.b.mi_row as isize + delta_row;
        let mv_col = self.b.mi_col as isize + delta_col;
        if self.is_inside(mv_row, mv_col) && self.mi(mv_row as usize, mv_col as usize).written {
            self.add_ref_mv_candidate(mv_row as usize, mv_col as usize, is_compound, 4);
        }
    }

    fn temporal_scan(&mut self, is_compound: bool) {
        let ms = self.b.mi_size;
        let bw4 = NUM_4X4_BLOCKS_WIDE[ms] as isize;
        let bh4 = NUM_4X4_BLOCKS_HIGH[ms] as isize;
        let step_w4 = if bw4 >= 16 { 4 } else { 2 };
        let step_h4 = if bh4 >= 16 { 4 } else { 2 };
        let mut dr = 0;
        while dr < bh4.min(16) {
            let mut dc = 0;
            while dc < bw4.min(16) {
                self.add_tpl_ref_mv(dr, dc, is_compound);
                dc += step_w4;
            }
            dr += step_h4;
        }
        let allow_extension = bh4 >= NUM_4X4_BLOCKS_HIGH[BLOCK_8X8] as isize
            && bh4 < NUM_4X4_BLOCKS_HIGH[BLOCK_64X64] as isize
            && bw4 >= NUM_4X4_BLOCKS_WIDE[BLOCK_8X8] as isize
            && bw4 < NUM_4X4_BLOCKS_WIDE[BLOCK_64X64] as isize;
        if allow_extension {
            let pos = [(bh4, -2), (bh4, bw4), (bh4 - 2, bw4)];
            for (dr, dc) in pos {
                let row = (self.b.mi_row & 15) as isize + dr;
                let col = (self.b.mi_col & 15) as isize + dc;
                if (0..16).contains(&row) && (0..16).contains(&col) {
                    self.add_tpl_ref_mv(dr, dc, is_compound);
                }
            }
        }
    }

    /// The temporal sample process (7.10.2.6).
    fn add_tpl_ref_mv(&mut self, delta_row: isize, delta_col: isize, is_compound: bool) {
        let mv_row = (self.b.mi_row as isize + delta_row) | 1;
        let mv_col = (self.b.mi_col as isize + delta_col) | 1;
        if !self.is_inside(mv_row, mv_col) {
            return;
        }
        let x8 = (mv_col >> 1) as usize;
        let y8 = (mv_row >> 1) as usize;
        let w8 = self.f.mi_cols >> 1;
        if delta_row == 0 && delta_col == 0 {
            self.b.zero_mv_context = 1;
        }
        let invalid = -1 << 15;
        if !is_compound {
            let mut cand = self.f.motion_field[self.b.ref_frame[0] as usize][y8 * w8 + x8];
            if cand[0] == invalid {
                return;
            }
            self.lower_mv_precision(&mut cand);
            if delta_row == 0 && delta_col == 0 {
                self.b.zero_mv_context = ((cand[0] - self.b.global_mvs[0][0]).abs() >= 16
                    || (cand[1] - self.b.global_mvs[0][1]).abs() >= 16)
                    as usize;
            }
            let n = self.b.num_mv_found;
            let mut idx = 0;
            while idx < n {
                if cand == self.b.ref_stack_mv[idx][0] {
                    break;
                }
                idx += 1;
            }
            if idx < n {
                self.b.weight_stack[idx] += 2;
            } else if n < MAX_REF_MV_STACK_SIZE {
                self.b.ref_stack_mv[n][0] = cand;
                self.b.weight_stack[n] = 2;
                self.b.num_mv_found += 1;
            }
        } else {
            let mut c0 = self.f.motion_field[self.b.ref_frame[0] as usize][y8 * w8 + x8];
            if c0[0] == invalid {
                return;
            }
            let mut c1 = self.f.motion_field[self.b.ref_frame[1] as usize][y8 * w8 + x8];
            if c1[0] == invalid {
                return;
            }
            self.lower_mv_precision(&mut c0);
            self.lower_mv_precision(&mut c1);
            if delta_row == 0 && delta_col == 0 {
                let g = self.b.global_mvs;
                self.b.zero_mv_context = ((c0[0] - g[0][0]).abs() >= 16
                    || (c0[1] - g[0][1]).abs() >= 16
                    || (c1[0] - g[1][0]).abs() >= 16
                    || (c1[1] - g[1][1]).abs() >= 16) as usize;
            }
            let n = self.b.num_mv_found;
            let mut idx = 0;
            while idx < n {
                if c0 == self.b.ref_stack_mv[idx][0] && c1 == self.b.ref_stack_mv[idx][1] {
                    break;
                }
                idx += 1;
            }
            if idx < n {
                self.b.weight_stack[idx] += 2;
            } else if n < MAX_REF_MV_STACK_SIZE {
                self.b.ref_stack_mv[n] = [c0, c1];
                self.b.weight_stack[n] = 2;
                self.b.num_mv_found += 1;
            }
        }
    }

    /// The add reference motion vector process (7.10.2.7).
    fn add_ref_mv_candidate(&mut self, mv_row: usize, mv_col: usize, is_compound: bool, weight: u32) {
        let m = *self.mi(mv_row, mv_col);
        if !m.is_inter {
            return;
        }
        if !is_compound {
            for cand_list in 0..2 {
                if m.ref_frame[cand_list] as i32 == self.b.ref_frame[0] {
                    self.search_stack(mv_row, mv_col, cand_list, weight);
                }
            }
        } else if m.ref_frame[0] as i32 == self.b.ref_frame[0] && m.ref_frame[1] as i32 == self.b.ref_frame[1] {
            self.compound_search_stack(mv_row, mv_col, weight);
        }
    }

    fn search_stack(&mut self, mv_row: usize, mv_col: usize, cand_list: usize, weight: u32) {
        let m = *self.mi(mv_row, mv_col);
        let cand_mode = m.y_mode as usize;
        let cand_size = m.mi_size as usize;
        let large = block_width(cand_size).min(block_height(cand_size)) >= 8;
        let mut cand_mv = if (cand_mode == GLOBALMV || cand_mode == GLOBAL_GLOBALMV)
            && self.f.hdr.gm_type[self.b.ref_frame[0] as usize] > TRANSLATION
            && large
        {
            self.b.global_mvs[0]
        } else {
            m.mv[cand_list]
        };
        self.lower_mv_precision(&mut cand_mv);
        if has_newmv(cand_mode) {
            self.b.new_mv_count += 1;
        }
        self.b.found_match = true;
        let n = self.b.num_mv_found;
        for idx in 0..n {
            if self.b.ref_stack_mv[idx][0] == cand_mv {
                self.b.weight_stack[idx] += weight;
                return;
            }
        }
        if n < MAX_REF_MV_STACK_SIZE {
            self.b.ref_stack_mv[n][0] = cand_mv;
            self.b.weight_stack[n] = weight;
            self.b.num_mv_found += 1;
        }
    }

    fn compound_search_stack(&mut self, mv_row: usize, mv_col: usize, weight: u32) {
        let m = *self.mi(mv_row, mv_col);
        let mut cand_mvs = m.mv;
        let cand_mode = m.y_mode as usize;
        if cand_mode == GLOBAL_GLOBALMV {
            for ref_list in 0..2 {
                if self.f.hdr.gm_type[self.b.ref_frame[ref_list] as usize] > TRANSLATION {
                    cand_mvs[ref_list] = self.b.global_mvs[ref_list];
                }
            }
        }
        for mv in cand_mvs.iter_mut() {
            self.lower_mv_precision(mv);
        }
        self.b.found_match = true;
        let n = self.b.num_mv_found;
        let mut found = false;
        for idx in 0..n {
            if self.b.ref_stack_mv[idx][0] == cand_mvs[0] && self.b.ref_stack_mv[idx][1] == cand_mvs[1] {
                self.b.weight_stack[idx] += weight;
                found = true;
                break;
            }
        }
        if !found && n < MAX_REF_MV_STACK_SIZE {
            self.b.ref_stack_mv[n] = cand_mvs;
            self.b.weight_stack[n] = weight;
            self.b.num_mv_found += 1;
        }
        if has_newmv(cand_mode) {
            self.b.new_mv_count += 1;
        }
    }

    fn sort_stack(&mut self, start: usize, end: usize, is_compound: bool) {
        let mut end = end;
        while end > start {
            let mut new_end = start;
            for idx in (start + 1)..end {
                if self.b.weight_stack[idx - 1] < self.b.weight_stack[idx] {
                    self.b.weight_stack.swap(idx - 1, idx);
                    let lists = if is_compound { 2 } else { 1 };
                    for list in 0..lists {
                        let t = self.b.ref_stack_mv[idx - 1][list];
                        self.b.ref_stack_mv[idx - 1][list] = self.b.ref_stack_mv[idx][list];
                        self.b.ref_stack_mv[idx][list] = t;
                    }
                    new_end = idx;
                }
            }
            end = new_end;
        }
    }

    /// The extra search process (7.10.2.12).
    fn extra_search(&mut self, is_compound: bool) {
        let mut ref_id_count = [0usize; 2];
        let mut ref_diff_count = [0usize; 2];
        let mut ref_id_mvs = [[[0i32; 2]; 2]; 2];
        let mut ref_diff_mvs = [[[0i32; 2]; 2]; 2];
        let ms = self.b.mi_size;
        let mut w4 = NUM_4X4_BLOCKS_WIDE[ms].min(16);
        let mut h4 = NUM_4X4_BLOCKS_HIGH[ms].min(16);
        w4 = w4.min(self.f.mi_cols - self.b.mi_col);
        h4 = h4.min(self.f.mi_rows - self.b.mi_row);
        let num4x4 = w4.min(h4);
        for pass in 0..2 {
            let mut idx = 0;
            while idx < num4x4 && self.b.num_mv_found < 2 {
                let (mv_row, mv_col) = if pass == 0 {
                    (self.b.mi_row as isize - 1, (self.b.mi_col + idx) as isize)
                } else {
                    ((self.b.mi_row + idx) as isize, self.b.mi_col as isize - 1)
                };
                if !self.is_inside(mv_row, mv_col) {
                    break;
                }
                let (r, c) = (mv_row as usize, mv_col as usize);
                // add_extra_mv_candidate (7.10.2.13)
                let m = *self.mi(r, c);
                if is_compound {
                    for cand_list in 0..2 {
                        let cand_ref = m.ref_frame[cand_list] as i32;
                        if cand_ref > INTRA_FRAME {
                            for list in 0..2 {
                                let mut cand_mv = m.mv[cand_list];
                                if cand_ref == self.b.ref_frame[list] && ref_id_count[list] < 2 {
                                    ref_id_mvs[list][ref_id_count[list]] = cand_mv;
                                    ref_id_count[list] += 1;
                                } else if ref_diff_count[list] < 2 {
                                    if self.f.hdr.ref_frame_sign_bias[cand_ref as usize]
                                        != self.f.hdr.ref_frame_sign_bias[self.b.ref_frame[list] as usize]
                                    {
                                        cand_mv[0] *= -1;
                                        cand_mv[1] *= -1;
                                    }
                                    ref_diff_mvs[list][ref_diff_count[list]] = cand_mv;
                                    ref_diff_count[list] += 1;
                                }
                            }
                        }
                    }
                } else {
                    for cand_list in 0..2 {
                        let cand_ref = m.ref_frame[cand_list] as i32;
                        if cand_ref > INTRA_FRAME {
                            let mut cand_mv = m.mv[cand_list];
                            if self.f.hdr.ref_frame_sign_bias[cand_ref as usize]
                                != self.f.hdr.ref_frame_sign_bias[self.b.ref_frame[0] as usize]
                            {
                                cand_mv[0] *= -1;
                                cand_mv[1] *= -1;
                            }
                            let n = self.b.num_mv_found;
                            let mut k = 0;
                            while k < n {
                                if cand_mv == self.b.ref_stack_mv[k][0] {
                                    break;
                                }
                                k += 1;
                            }
                            if k == n {
                                self.b.ref_stack_mv[k][0] = cand_mv;
                                self.b.weight_stack[k] = 2;
                                self.b.num_mv_found += 1;
                            }
                        }
                    }
                }
                if pass == 0 {
                    idx += NUM_4X4_BLOCKS_WIDE[m.mi_size as usize];
                } else {
                    idx += NUM_4X4_BLOCKS_HIGH[m.mi_size as usize];
                }
            }
        }
        if is_compound {
            let mut combined = [[[0i32; 2]; 2]; 2];
            for list in 0..2 {
                let mut comp_count = 0;
                for idx in 0..ref_id_count[list] {
                    combined[comp_count][list] = ref_id_mvs[list][idx];
                    comp_count += 1;
                }
                let mut idx = 0;
                while idx < ref_diff_count[list] && comp_count < 2 {
                    combined[comp_count][list] = ref_diff_mvs[list][idx];
                    comp_count += 1;
                    idx += 1;
                }
                while comp_count < 2 {
                    combined[comp_count][list] = self.b.global_mvs[list];
                    comp_count += 1;
                }
            }
            let n = self.b.num_mv_found;
            if n == 1 {
                if combined[0][0] == self.b.ref_stack_mv[0][0] && combined[0][1] == self.b.ref_stack_mv[0][1] {
                    self.b.ref_stack_mv[n] = combined[1];
                } else {
                    self.b.ref_stack_mv[n] = combined[0];
                }
                self.b.weight_stack[n] = 2;
                self.b.num_mv_found += 1;
            } else {
                for c in combined.iter() {
                    let n = self.b.num_mv_found;
                    self.b.ref_stack_mv[n] = *c;
                    self.b.weight_stack[n] = 2;
                    self.b.num_mv_found += 1;
                }
            }
        } else {
            for idx in self.b.num_mv_found..2 {
                self.b.ref_stack_mv[idx][0] = self.b.global_mvs[0];
            }
        }
    }

    /// The context and clamping process (7.10.2.14).
    fn context_and_clamping(&mut self, is_compound: bool, num_new: usize) {
        let ms = self.b.mi_size;
        let bw = block_width(ms) as i32;
        let bh = block_height(ms) as i32;
        let n = self.b.num_mv_found;
        for idx in 0..n {
            let mut z = 0;
            if idx + 1 < n {
                let w0 = self.b.weight_stack[idx];
                let w1 = self.b.weight_stack[idx + 1];
                if w0 >= REF_CAT_LEVEL {
                    if w1 < REF_CAT_LEVEL {
                        z = 1;
                    }
                } else {
                    z = 2;
                }
            }
            self.b.drl_ctx_stack[idx] = z;
        }
        let lists = if is_compound { 2 } else { 1 };
        for list in 0..lists {
            for idx in 0..n {
                let mv = self.b.ref_stack_mv[idx][list];
                self.b.ref_stack_mv[idx][list] = [
                    self.clamp_mv_row(mv[0], MV_BORDER + bh * 8),
                    self.clamp_mv_col(mv[1], MV_BORDER + bw * 8),
                ];
            }
        }
        let b = &mut self.b;
        if b.close_matches == 0 {
            b.new_mv_context = b.total_matches.min(1);
            b.ref_mv_context = b.total_matches;
        } else if b.close_matches == 1 {
            b.new_mv_context = 3 - num_new.min(1);
            b.ref_mv_context = 2 + b.total_matches;
        } else {
            b.new_mv_context = 5 - num_new.min(1);
            b.ref_mv_context = 5;
        }
    }

    pub(crate) fn clamp_mv_row(&self, mvec: i32, border: i32) -> i32 {
        let bh4 = NUM_4X4_BLOCKS_HIGH[self.b.mi_size] as i32;
        let mb_to_top_edge = -((self.b.mi_row as i32 * MI_SIZE as i32) * 8);
        let mb_to_bottom_edge = ((self.f.mi_rows as i32 - bh4 - self.b.mi_row as i32) * MI_SIZE as i32) * 8;
        clip3(mb_to_top_edge - border, mb_to_bottom_edge + border, mvec)
    }

    pub(crate) fn clamp_mv_col(&self, mvec: i32, border: i32) -> i32 {
        let bw4 = NUM_4X4_BLOCKS_WIDE[self.b.mi_size] as i32;
        let mb_to_left_edge = -((self.b.mi_col as i32 * MI_SIZE as i32) * 8);
        let mb_to_right_edge = ((self.f.mi_cols as i32 - bw4 - self.b.mi_col as i32) * MI_SIZE as i32) * 8;
        clip3(mb_to_left_edge - border, mb_to_right_edge + border, mvec)
    }

    /// The has overlappable candidates process (7.10.3).
    pub(crate) fn has_overlappable_candidates(&self) -> bool {
        let ms = self.b.mi_size;
        if self.b.avail_u {
            let w4 = NUM_4X4_BLOCKS_WIDE[ms];
            let mut x4 = self.b.mi_col;
            while x4 < self.f.mi_cols.min(self.b.mi_col + w4) {
                if self.mi(self.b.mi_row - 1, x4 | 1).ref_frame[0] as i32 > INTRA_FRAME {
                    return true;
                }
                x4 += 2;
            }
        }
        if self.b.avail_l {
            let h4 = NUM_4X4_BLOCKS_HIGH[ms];
            let mut y4 = self.b.mi_row;
            while y4 < self.f.mi_rows.min(self.b.mi_row + h4) {
                if self.mi(y4 | 1, self.b.mi_col - 1).ref_frame[0] as i32 > INTRA_FRAME {
                    return true;
                }
                y4 += 2;
            }
        }
        false
    }

    /// The find warp samples process (7.10.4).
    pub(crate) fn find_warp_samples(&mut self) {
        self.b.num_samples = 0;
        self.b.num_samples_scanned = 0;
        let ms = self.b.mi_size;
        let w4 = NUM_4X4_BLOCKS_WIDE[ms] as isize;
        let h4 = NUM_4X4_BLOCKS_HIGH[ms] as isize;
        let mut do_top_left = true;
        let mut do_top_right = true;
        let (mi_row, mi_col) = (self.b.mi_row, self.b.mi_col);
        if self.b.avail_u {
            let src_size = self.mi(mi_row - 1, mi_col).mi_size as usize;
            let src_w = NUM_4X4_BLOCKS_WIDE[src_size] as isize;
            if w4 <= src_w {
                let col_offset = -((mi_col as isize) & (src_w - 1));
                if col_offset < 0 {
                    do_top_left = false;
                }
                if col_offset + src_w > w4 {
                    do_top_right = false;
                }
                self.add_sample(-1, 0);
            } else {
                let mut i = 0isize;
                let lim = w4.min((self.f.mi_cols - mi_col) as isize);
                while i < lim {
                    let src_size = self.mi(mi_row - 1, mi_col + i as usize).mi_size as usize;
                    let src_w = NUM_4X4_BLOCKS_WIDE[src_size] as isize;
                    let step = w4.min(src_w);
                    self.add_sample(-1, i);
                    i += step;
                }
            }
        }
        if self.b.avail_l {
            let src_size = self.mi(mi_row, mi_col - 1).mi_size as usize;
            let src_h = NUM_4X4_BLOCKS_HIGH[src_size] as isize;
            if h4 <= src_h {
                let row_offset = -((mi_row as isize) & (src_h - 1));
                if row_offset < 0 {
                    do_top_left = false;
                }
                self.add_sample(0, -1);
            } else {
                let mut i = 0isize;
                let lim = h4.min((self.f.mi_rows - mi_row) as isize);
                while i < lim {
                    let src_size = self.mi(mi_row + i as usize, mi_col - 1).mi_size as usize;
                    let src_h = NUM_4X4_BLOCKS_HIGH[src_size] as isize;
                    let step = h4.min(src_h);
                    self.add_sample(i, -1);
                    i += step;
                }
            }
        }
        if do_top_left {
            self.add_sample(-1, -1);
        }
        if do_top_right && w4.max(h4) <= 16 {
            self.add_sample(-1, w4);
        }
        if self.b.num_samples == 0 && self.b.num_samples_scanned > 0 {
            self.b.num_samples = 1;
        }
    }

    /// The add sample process (7.10.4.2).
    fn add_sample(&mut self, delta_row: isize, delta_col: isize) {
        if self.b.num_samples_scanned >= LEAST_SQUARES_SAMPLES_MAX {
            return;
        }
        let mv_row = self.b.mi_row as isize + delta_row;
        let mv_col = self.b.mi_col as isize + delta_col;
        if !self.is_inside(mv_row, mv_col) {
            return;
        }
        let m = *self.mi(mv_row as usize, mv_col as usize);
        if !m.written {
            return;
        }
        if m.ref_frame[0] as i32 != self.b.ref_frame[0] {
            return;
        }
        if m.ref_frame[1] as i32 != NONE {
            return;
        }
        let cand_sz = m.mi_size as usize;
        let cand_w4 = NUM_4X4_BLOCKS_WIDE[cand_sz] as isize;
        let cand_h4 = NUM_4X4_BLOCKS_HIGH[cand_sz] as isize;
        let cand_row = mv_row & !(cand_h4 - 1);
        let cand_col = mv_col & !(cand_w4 - 1);
        let mid_y = (cand_row * 4 + cand_h4 * 2 - 1) as i32;
        let mid_x = (cand_col * 4 + cand_w4 * 2 - 1) as i32;
        let ms = self.b.mi_size;
        let threshold = clip3(16, 112, block_width(ms).max(block_height(ms)) as i32);
        let cmv = self.mi(cand_row as usize, cand_col as usize).mv[0];
        let mv_diff_row = (cmv[0] - self.b.mv[0][0]).abs();
        let mv_diff_col = (cmv[1] - self.b.mv[0][1]).abs();
        let valid = mv_diff_row + mv_diff_col <= threshold;
        let cand = [mid_y * 8, mid_x * 8, mid_y * 8 + cmv[0], mid_x * 8 + cmv[1]];
        self.b.num_samples_scanned += 1;
        if !valid && self.b.num_samples_scanned > 1 {
            return;
        }
        self.b.cand_list[self.b.num_samples] = cand;
        if valid {
            self.b.num_samples += 1;
        }
    }
}

fn has_newmv(mode: usize) -> bool {
    mode == NEWMV
        || mode == NEW_NEWMV
        || mode == NEAR_NEWMV
        || mode == NEW_NEARMV
        || mode == NEAREST_NEWMV
        || mode == NEW_NEARESTMV
}

//! The AV1 decoder: OBUs in, frames out (7.1 to 7.21).
//!
//! [`Decoder::decode`] takes a temporal unit (one IVF frame, one Matroska
//! block: a sequence of OBUs in the low-overhead format) and returns the
//! frame it shows. Decoding follows the specification's processes in their
//! order: the frame header, the tiles (mode info, prediction, residual),
//! then the loop filter, CDEF, super-resolution, loop restoration, the
//! reference update and, on output, film grain synthesis.

pub(crate) mod grain;
pub(crate) mod mvpred;
pub(crate) mod postfilter;
pub(crate) mod predict;
pub(crate) mod residual;
pub(crate) mod state;
pub(crate) mod tile;

use std::sync::Arc;

use crate::bits::BitReader;
use crate::cdf::CdfContext;
use crate::consts::*;
use crate::frame::{ChromaFormat, ColorInfo, Frame};
use crate::header::{get_relative_dist, FrameHeader, RefHeaderState, RefState};
use crate::obu::{split_obus, SequenceHeader};
use crate::tables::*;
use crate::{Error, Result};
use state::{FrameBuf, Mi, Mv, PlaneBuf, RefData};
use tile::TileDecoder;

/// Loop restoration parameters of one plane (`LrType`, `LrWiener`,
/// `LrSgrSet`, `LrSgrXqd`).
#[derive(Clone, Default)]
pub(crate) struct LrPlane {
    pub(crate) unit_rows: usize,
    pub(crate) unit_cols: usize,
    pub(crate) lr_type: Vec<u8>,
    pub(crate) wiener: Vec<[[i32; 3]; 2]>,
    pub(crate) sgr_set: Vec<u8>,
    pub(crate) sgr_xqd: Vec<[i32; 2]>,
}

/// The state of the frame being decoded.
pub(crate) struct FrameCtx {
    pub(crate) seq: Arc<SequenceHeader>,
    pub(crate) hdr: FrameHeader,
    pub(crate) bit_depth: u32,
    pub(crate) ssx: usize,
    pub(crate) ssy: usize,
    pub(crate) num_planes: usize,
    pub(crate) mi_rows: usize,
    pub(crate) mi_cols: usize,
    /// Row stride of the per-4x4 arrays: `mi_cols` plus a margin of 32,
    /// which blocks overhanging the frame write into as the specification's
    /// unbounded arrays allow (and later reads may see).
    pub(crate) ms: usize,
    pub(crate) mi: Vec<Mi>,
    pub(crate) tx_types: Vec<u8>,
    pub(crate) lf_tx_sizes: [Vec<u8>; 3],
    pub(crate) palette_colors: [Vec<[u16; 8]>; 2],
    pub(crate) segment_ids: Vec<u8>,
    pub(crate) prev_segment_ids: Vec<u8>,
    pub(crate) cdef_idx: Vec<i8>,
    pub(crate) cdef_stride: usize,
    pub(crate) lr: [LrPlane; 3],
    /// `MotionFieldMvs[ ref ]` on the 8x8 grid.
    pub(crate) motion_field: Vec<Vec<Mv>>,
    pub(crate) cur: FrameBuf,
    pub(crate) refs: [Option<Arc<RefData>>; 8],
    /// The frame's CDFs (6.8.2).
    pub(crate) cdfs: Box<CdfContext>,
    /// The `Saved` CDFs from tile `context_update_tile_id`.
    pub(crate) saved_cdfs: Option<Box<CdfContext>>,
    /// `Mask` of the inter prediction process, 128 wide.
    pub(crate) mask: Box<[u8; 128 * 128]>,
    /// Tiles decoded so far (`TileNum`).
    pub(crate) tile_num: usize,
    /// Every tile's padding checked out (see `SymbolDecoder::trailing_ok`).
    pub(crate) tiles_ok: bool,
}

impl FrameCtx {
    /// `get_plane_residual_size( subsize, plane )`.
    #[inline]
    pub(crate) fn plane_residual_size(&self, subsize: usize, plane: usize) -> usize {
        let (sx, sy) = self.plane_ss(plane);
        SUBSAMPLED_SIZE[subsize][sx][sy]
    }

    /// `is_scaled( refFrame )`.
    pub(crate) fn is_scaled(&self, ref_frame: i32) -> bool {
        if ref_frame <= INTRA_FRAME {
            return false;
        }
        let idx = self.hdr.ref_frame_idx[(ref_frame - LAST_FRAME) as usize];
        let Some(r) = &self.refs[idx] else {
            return false;
        };
        let fw = self.hdr.frame_width as i64;
        let fh = self.hdr.frame_height as i64;
        let x_scale = (((r.upscaled_width as i64) << REF_SCALE_SHIFT) + fw / 2) / fw;
        let y_scale = (((r.frame_height as i64) << REF_SCALE_SHIFT) + fh / 2) / fh;
        let no_scale = 1i64 << REF_SCALE_SHIFT;
        x_scale != no_scale || y_scale != no_scale
    }
}

/// Decodes AV1 temporal units into frames.
pub struct Decoder {
    seq: Option<Arc<SequenceHeader>>,
    ref_state: RefState,
    refs: [Option<Arc<RefData>>; 8],
    frame: Option<FrameCtx>,
    seen_frame_header: bool,
    shown: Vec<Frame>,
    strict: bool,
}

impl Default for Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Decoder {
    /// A decoder awaiting a sequence header.
    pub fn new() -> Self {
        Decoder {
            seq: None,
            ref_state: RefState::default(),
            refs: Default::default(),
            frame: None,
            seen_frame_header: false,
            shown: Vec::new(),
            strict: false,
        }
    }

    /// Makes the decoder report a tile whose arithmetic-coded data does not
    /// end with the padding `exit_symbol()` requires as a bitstream error.
    /// A conformance check; off by default.
    pub fn set_strict(&mut self, strict: bool) {
        self.strict = strict;
    }

    /// Decodes one temporal unit (or any sequence of whole OBUs) and
    /// returns the last frame it shows, if any.
    pub fn decode(&mut self, data: &[u8]) -> Result<Option<Frame>> {
        Ok(self.decode_all(data)?.pop())
    }

    /// Decodes one temporal unit and returns every frame it shows, in
    /// order (more than one only with spatial layers).
    pub fn decode_all(&mut self, data: &[u8]) -> Result<Vec<Frame>> {
        self.shown.clear();
        for obu in split_obus(data)? {
            self.decode_obu(obu.obu_type, obu.temporal_id, obu.spatial_id, obu.has_extension, obu.payload)?;
        }
        Ok(std::mem::take(&mut self.shown))
    }

    fn decode_obu(&mut self, obu_type: u32, temporal_id: u32, spatial_id: u32, ext: bool, payload: &[u8]) -> Result<()> {
        if obu_type != OBU_SEQUENCE_HEADER && obu_type != OBU_TEMPORAL_DELIMITER && ext {
            if let Some(seq) = &self.seq {
                let idc = seq.op_idc;
                if idc != 0 {
                    let in_t = (idc >> temporal_id) & 1;
                    let in_s = (idc >> (spatial_id + 8)) & 1;
                    if in_t == 0 || in_s == 0 {
                        return Ok(());
                    }
                }
            }
        }
        match obu_type {
            OBU_SEQUENCE_HEADER => {
                let s = SequenceHeader::parse(payload)?;
                self.seq = Some(Arc::new(s));
            }
            OBU_TEMPORAL_DELIMITER => {
                self.seen_frame_header = false;
            }
            OBU_FRAME_HEADER | OBU_REDUNDANT_FRAME_HEADER | OBU_FRAME => {
                if self.seen_frame_header {
                    if obu_type == OBU_FRAME {
                        return Err(Error::bitstream("OBU_FRAME while a frame is in progress"));
                    }
                    // frame_header_copy(): identical to the header in force.
                    return Ok(());
                }
                let seq = self
                    .seq
                    .clone()
                    .ok_or_else(|| Error::bitstream("frame header before any sequence header"))?;
                let mut r = BitReader::new(payload);
                let hdr = FrameHeader::parse(&mut r, &seq, &mut self.ref_state, temporal_id, spatial_id)?;
                if hdr.show_existing_frame {
                    self.show_existing(seq, hdr)?;
                    self.seen_frame_header = false;
                    return Ok(());
                }
                self.seen_frame_header = true;
                self.frame = Some(self.setup_frame(seq, hdr)?);
                if obu_type == OBU_FRAME {
                    r.byte_align()?;
                    let pos = r.position() / 8;
                    self.tile_group(&payload[pos..])?;
                }
            }
            OBU_TILE_GROUP => {
                if !self.seen_frame_header || self.frame.is_none() {
                    return Err(Error::bitstream("tile group without a frame header"));
                }
                self.tile_group(payload)?;
            }
            OBU_TILE_LIST => {
                return Err(Error::unsupported("large scale tile decoding (tile list OBUs)"));
            }
            _ => {}
        }
        Ok(())
    }

    /// Sets up the frame state after `uncompressed_header()`: CDFs, the
    /// previous segment map, the motion field, the sample buffers.
    fn setup_frame(&mut self, seq: Arc<SequenceHeader>, hdr: FrameHeader) -> Result<FrameCtx> {
        let c = &seq.color;
        let mi_rows = hdr.mi_rows;
        let mi_cols = hdr.mi_cols;
        if !hdr.frame_is_intra {
            for i in 0..REFS_PER_FRAME {
                let idx = hdr.ref_frame_idx[i];
                let r = self.refs[idx]
                    .as_ref()
                    .ok_or_else(|| Error::bitstream("inter frame references an empty slot"))?;
                if r.bit_depth != c.bit_depth
                    || r.subsampling_x != c.subsampling_x
                    || r.subsampling_y != c.subsampling_y
                {
                    return Err(Error::bitstream("reference frame format differs"));
                }
                let (fw, fh) = (hdr.frame_width as usize, hdr.frame_height as usize);
                if 2 * fw < r.upscaled_width || 2 * fh < r.frame_height || fw > 16 * r.upscaled_width || fh > 16 * r.frame_height {
                    return Err(Error::bitstream("reference frame scale out of range"));
                }
            }
        }
        let cdfs = if hdr.primary_ref_frame == PRIMARY_REF_NONE {
            CdfContext::new(hdr.base_q_idx)
        } else {
            let idx = hdr.ref_frame_idx[hdr.primary_ref_frame];
            let r = self.refs[idx]
                .as_ref()
                .ok_or_else(|| Error::bitstream("primary_ref_frame refers to an empty slot"))?;
            let mut c = Box::new((*r.cdfs).clone());
            c.clear_counts();
            c
        };
        let ms = mi_cols + 32;
        let n = (mi_rows + 32) * ms;
        let mut prev_segment_ids = vec![0u8; n];
        if hdr.primary_ref_frame != PRIMARY_REF_NONE && hdr.segmentation_enabled {
            let idx = hdr.ref_frame_idx[hdr.primary_ref_frame];
            if let Some(r) = &self.refs[idx] {
                if r.mi_rows == mi_rows && r.mi_cols == mi_cols {
                    prev_segment_ids.copy_from_slice(&r.saved_segment_ids);
                }
            }
        }
        let ssx = c.subsampling_x as usize;
        let ssy = c.subsampling_y as usize;
        let num_planes = c.num_planes;
        let aw = (mi_cols * MI_SIZE + 127) & !127;
        let ah = (mi_rows * MI_SIZE + 127) & !127;
        let mut planes = vec![PlaneBuf::new(aw + 32, ah + 32, 0)];
        for _ in 1..num_planes {
            planes.push(PlaneBuf::new((aw >> ssx) + 32, (ah >> ssy) + 32, 0));
        }
        let cdef_stride = (mi_cols >> 4) + 2;
        let mut lr: [LrPlane; 3] = Default::default();
        for (plane, l) in lr.iter_mut().enumerate().take(num_planes) {
            if hdr.frame_restoration_type[plane] != RESTORE_NONE {
                let (sx, sy) = if plane == 0 { (0, 0) } else { (ssx as u32, ssy as u32) };
                let unit_size = hdr.loop_restoration_size[plane];
                l.unit_rows = count_units_in_frame(unit_size, round2(hdr.frame_height as i32, sy) as usize);
                l.unit_cols = count_units_in_frame(unit_size, round2(hdr.upscaled_width as i32, sx) as usize);
                let units = l.unit_rows * l.unit_cols;
                l.lr_type = vec![RESTORE_NONE; units];
                l.wiener = vec![[[0; 3]; 2]; units];
                l.sgr_set = vec![0; units];
                l.sgr_xqd = vec![[0; 2]; units];
            }
        }
        let mut f = FrameCtx {
            seq: seq.clone(),
            hdr,
            bit_depth: c.bit_depth,
            ssx,
            ssy,
            num_planes,
            mi_rows,
            mi_cols,
            ms,
            mi: vec![Mi::default(); n],
            tx_types: vec![0; n],
            lf_tx_sizes: [vec![0; n], vec![0; n], vec![0; n]],
            palette_colors: [vec![[0; 8]; n], vec![[0; 8]; n]],
            segment_ids: vec![0; n],
            prev_segment_ids,
            cdef_idx: vec![-1; cdef_stride * ((mi_rows >> 4) + 2)],
            cdef_stride,
            lr,
            motion_field: Vec::new(),
            cur: FrameBuf { planes },
            refs: self.refs.clone(),
            cdfs,
            saved_cdfs: None,
            mask: Box::new([0; 128 * 128]),
            tile_num: 0,
            tiles_ok: true,
        };
        if f.hdr.use_ref_frame_mvs {
            self.motion_field_estimation(&mut f);
        } else {
            f.motion_field = vec![Vec::new(); 8];
        }
        Ok(f)
    }

    /// `tile_group_obu( sz )` (5.11.1).
    fn tile_group(&mut self, data: &[u8]) -> Result<()> {
        let f = self.frame.as_mut().expect("checked by the caller");
        let ti = f.hdr.tile_info.clone();
        let num_tiles = ti.cols * ti.rows;
        let mut r = BitReader::new(data);
        let mut tile_start_and_end_present = false;
        if num_tiles > 1 {
            tile_start_and_end_present = r.flag()?;
        }
        let (tg_start, tg_end) = if num_tiles == 1 || !tile_start_and_end_present {
            (0, num_tiles - 1)
        } else {
            let bits = ti.cols_log2 + ti.rows_log2;
            (r.f(bits)? as usize, r.f(bits)? as usize)
        };
        if tg_end >= num_tiles || tg_start > tg_end {
            return Err(Error::bitstream("tile group range out of range"));
        }
        r.byte_align()?;
        let mut pos = r.position() / 8;
        for tile_num in tg_start..=tg_end {
            let tile_row = tile_num / ti.cols;
            let tile_col = tile_num % ti.cols;
            let last = tile_num == tg_end;
            let tile_size = if last {
                data.len() - pos
            } else {
                let n = ti.tile_size_bytes as usize;
                if pos + n > data.len() {
                    return Err(Error::bitstream("tile size runs past the tile group"));
                }
                let mut v = 0usize;
                for i in 0..n {
                    v |= (data[pos + i] as usize) << (8 * i);
                }
                pos += n;
                v + 1
            };
            if pos + tile_size > data.len() {
                return Err(Error::bitstream("tile data runs past the tile group"));
            }
            let tile_data = &data[pos..pos + tile_size];
            pos += tile_size;
            let (ok, saved) = {
                let mut td = TileDecoder::new(f, tile_data, tile_row, tile_col);
                td.decode_tile()?;
                let ok = td.sd.trailing_ok();
                let saved = if !td.f.hdr.disable_frame_end_update_cdf && tile_num == ti.context_update_tile_id {
                    Some(td.cdf)
                } else {
                    None
                };
                (ok, saved)
            };
            if !ok {
                f.tiles_ok = false;
                if self.strict {
                    return Err(Error::bitstream(format!("tile {tile_num} does not end with its padding")));
                }
            }
            if saved.is_some() {
                f.saved_cdfs = saved;
            }
            f.tile_num = tile_num + 1;
        }
        if tg_end == num_tiles - 1 {
            let mut f = self.frame.take().expect("frame in progress");
            if !f.hdr.disable_frame_end_update_cdf {
                if let Some(s) = f.saved_cdfs.take() {
                    // frame_end_update_cdf(); the counters do not matter: every
                    // load clears them.
                    f.cdfs = s;
                }
            }
            self.decode_frame_wrapup(f)?;
            self.seen_frame_header = false;
        }
        Ok(())
    }

    /// The decode frame wrapup process (7.4) for a decoded frame.
    fn decode_frame_wrapup(&mut self, mut f: FrameCtx) -> Result<()> {
        if f.hdr.loop_filter_level[0] != 0 || f.hdr.loop_filter_level[1] != 0 {
            postfilter::loop_filter(&mut f);
        }
        let cdef = postfilter::cdef(&f);
        let up_cdef = match postfilter::upscale(&f, &cdef) {
            Some(u) => u,
            None => cdef,
        };
        let up_cur_owned = postfilter::upscale(&f, &f.cur);
        let up_cur = up_cur_owned.as_ref().unwrap_or(&f.cur);
        let lr = postfilter::loop_restoration(&f, up_cur, up_cdef);
        drop(up_cur_owned);
        let (mf_ref_frames, mf_mvs) = self.motion_vector_storage(&f);
        if f.hdr.segmentation_enabled && !f.hdr.segmentation_update_map {
            f.segment_ids.copy_from_slice(&f.prev_segment_ids);
        }
        let seq = f.seq.clone();
        let h = &f.hdr;
        let film_grain = h.film_grain.clone();
        let data = Arc::new(RefData {
            frame: lr,
            frame_type: h.frame_type,
            upscaled_width: h.upscaled_width as usize,
            frame_width: h.frame_width as usize,
            frame_height: h.frame_height as usize,
            render_width: h.render_width,
            render_height: h.render_height,
            mi_rows: f.mi_rows,
            mi_cols: f.mi_cols,
            bit_depth: f.bit_depth,
            subsampling_x: f.ssx as u32,
            subsampling_y: f.ssy as u32,
            order_hint: h.order_hint,
            saved_order_hints: h.order_hints,
            saved_ref_frames: Arc::new(mf_ref_frames),
            saved_mvs: Arc::new(mf_mvs),
            saved_segment_ids: Arc::new(std::mem::take(&mut f.segment_ids)),
            cdfs: Arc::new(*f.cdfs),
            film_grain: film_grain.clone(),
            showable_frame: h.showable_frame,
            color_range: seq.color.color_range,
            matrix_coefficients: seq.color.matrix_coefficients,
        });
        let hs = Arc::new(RefHeaderState {
            frame_id: h.current_frame_id,
            frame_type: h.frame_type,
            upscaled_width: h.upscaled_width,
            frame_width: h.frame_width,
            frame_height: h.frame_height,
            render_width: h.render_width,
            render_height: h.render_height,
            mi_cols: f.mi_cols as u32,
            mi_rows: f.mi_rows as u32,
            loop_filter_ref_deltas: h.loop_filter_ref_deltas,
            loop_filter_mode_deltas: h.loop_filter_mode_deltas,
            feature_enabled: h.feature_enabled,
            feature_data: h.feature_data,
            gm_params: h.gm_params,
            film_grain,
            saved_order_hints: h.order_hints,
            bit_depth: f.bit_depth,
            subsampling_x: f.ssx as u32,
            subsampling_y: f.ssy as u32,
            showable_frame: h.showable_frame,
        });
        // The reference frame update process (7.20).
        for i in 0..NUM_REF_FRAMES {
            if (h.refresh_frame_flags >> i) & 1 != 0 {
                self.refs[i] = Some(data.clone());
                self.ref_state.slots[i] = Some(hs.clone());
                self.ref_state.valid[i] = true;
                self.ref_state.order_hint[i] = h.order_hint;
            }
        }
        self.ref_state.current_frame_id = h.current_frame_id;
        if h.show_frame {
            let out = self.output(&seq, &data, &h.film_grain);
            self.shown.push(out);
        }
        Ok(())
    }

    /// `show_existing_frame` (7.4, 7.21).
    fn show_existing(&mut self, seq: Arc<SequenceHeader>, hdr: FrameHeader) -> Result<()> {
        let idx = hdr.frame_to_show_map_idx;
        let data = self.refs[idx]
            .clone()
            .ok_or_else(|| Error::bitstream("show_existing_frame of an empty slot"))?;
        let hs = self.ref_state.slots[idx]
            .clone()
            .ok_or_else(|| Error::bitstream("show_existing_frame of an empty slot"))?;
        if hdr.frame_type == KEY_FRAME {
            // The reference frame loading process, then the update with
            // refresh_frame_flags = allFrames.
            let order_hint = data.order_hint;
            for i in 0..NUM_REF_FRAMES {
                if (hdr.refresh_frame_flags >> i) & 1 != 0 {
                    self.refs[i] = Some(data.clone());
                    self.ref_state.slots[i] = Some(hs.clone());
                    self.ref_state.valid[i] = true;
                    self.ref_state.order_hint[i] = order_hint;
                }
            }
        }
        let out = self.output(&seq, &data, &hdr.film_grain);
        self.shown.push(out);
        Ok(())
    }

    /// The output process (7.18): the intermediate output and film grain.
    fn output(&self, seq: &SequenceHeader, data: &RefData, grain: &crate::header::FilmGrainParams) -> Frame {
        let w = data.upscaled_width;
        let h = data.frame_height;
        let mono = seq.color.mono_chrome;
        let chroma = if mono {
            ChromaFormat::Mono
        } else {
            ChromaFormat::from_shifts(data.subsampling_x, data.subsampling_y)
        };
        let mut frame = Frame::new(w as u32, h as u32, data.bit_depth, chroma);
        frame.render_width = data.render_width;
        frame.render_height = data.render_height;
        frame.color = ColorInfo {
            color_primaries: seq.color.color_primaries,
            transfer_characteristics: seq.color.transfer_characteristics,
            matrix_coefficients: seq.color.matrix_coefficients,
            full_range: seq.color.color_range,
            chroma_sample_position: seq.color.chroma_sample_position,
        };
        let mut planes: Vec<Vec<u16>> = Vec::new();
        for (p, pl) in data.frame.planes.iter().enumerate() {
            let (sx, sy) = if p == 0 {
                (0, 0)
            } else {
                (data.subsampling_x as usize, data.subsampling_y as usize)
            };
            let pw = (w + sx) >> sx;
            let ph = (h + sy) >> sy;
            let mut v = Vec::with_capacity(pw * ph);
            for y in 0..ph {
                v.extend_from_slice(&pl.row(y)[..pw]);
            }
            planes.push(v);
        }
        if seq.film_grain_params_present && grain.apply_grain {
            grain::apply(seq, data.bit_depth, data.subsampling_x as usize, data.subsampling_y as usize, w, h, grain, &mut planes);
        }
        for (p, v) in planes.iter().enumerate() {
            let pl = frame.planes[p];
            for y in 0..pl.height {
                for x in 0..pl.width {
                    frame.set_sample(p, x, y, v[(y * pl.width + x) as usize]);
                }
            }
        }
        frame
    }

    /// The motion field estimation process (7.9).
    fn motion_field_estimation(&self, f: &mut FrameCtx) {
        let w8 = f.mi_cols >> 1;
        let h8 = f.mi_rows >> 1;
        let invalid: Mv = [-1 << 15, -1 << 15];
        f.motion_field = vec![vec![invalid; w8 * h8]; 8];
        let seq = f.seq.clone();
        let h = f.hdr.clone();
        let last_idx = h.ref_frame_idx[0];
        let cur_gold_order_hint = h.order_hints[GOLDEN_FRAME as usize];
        let last_alt_order_hint = self.refs[last_idx]
            .as_ref()
            .map_or(0, |r| r.saved_order_hints[ALTREF_FRAME as usize]);
        let use_last = last_alt_order_hint != cur_gold_order_hint;
        if use_last {
            self.project(f, LAST_FRAME, -1);
        }
        let mut ref_stamp = MFMV_STACK_SIZE - 2;
        let dist = |a: u32, b: u32| get_relative_dist(&seq, a, b);
        if dist(h.order_hints[BWDREF_FRAME as usize], h.order_hint) > 0 && self.project(f, BWDREF_FRAME, 1) {
            ref_stamp -= 1;
        }
        if dist(h.order_hints[ALTREF2_FRAME as usize], h.order_hint) > 0 && self.project(f, ALTREF2_FRAME, 1) {
            ref_stamp -= 1;
        }
        if dist(h.order_hints[ALTREF_FRAME as usize], h.order_hint) > 0
            && ref_stamp >= 0
            && self.project(f, ALTREF_FRAME, 1)
        {
            ref_stamp -= 1;
        }
        if ref_stamp >= 0 {
            self.project(f, LAST2_FRAME, -1);
        }
    }

    /// The projection process (7.9.2).
    fn project(&self, f: &mut FrameCtx, src: i32, dst_sign: i32) -> bool {
        let src_idx = f.hdr.ref_frame_idx[(src - LAST_FRAME) as usize];
        let w8 = (f.mi_cols >> 1) as i32;
        let h8 = (f.mi_rows >> 1) as i32;
        let Some(r) = self.refs[src_idx].clone() else {
            return false;
        };
        if r.mi_rows != f.mi_rows
            || r.mi_cols != f.mi_cols
            || r.frame_type == INTRA_ONLY_FRAME
            || r.frame_type == KEY_FRAME
        {
            return false;
        }
        let seq = f.seq.clone();
        let dist = |a: u32, b: u32| get_relative_dist(&seq, a, b);
        let order_hint = f.hdr.order_hint;
        let order_hints = f.hdr.order_hints;
        for y8 in 0..h8 {
            for x8 in 0..w8 {
                let i = (y8 * w8 + x8) as usize;
                let src_ref = r.saved_ref_frames[i] as i32;
                if src_ref <= INTRA_FRAME {
                    continue;
                }
                let ref_to_cur = dist(order_hints[src as usize], order_hint);
                let ref_offset = dist(order_hints[src as usize], r.saved_order_hints[src_ref as usize]);
                let pos_valid = ref_to_cur.abs() <= MAX_FRAME_DISTANCE
                    && ref_offset.abs() <= MAX_FRAME_DISTANCE
                    && ref_offset > 0;
                if !pos_valid {
                    continue;
                }
                let mv = r.saved_mvs[i];
                let proj_mv = get_mv_projection(mv, ref_to_cur * dst_sign, ref_offset);
                let (pos_y8, vy) = project_pos(y8, proj_mv[0], dst_sign, h8, MAX_OFFSET_HEIGHT);
                let (pos_x8, vx) = project_pos(x8, proj_mv[1], dst_sign, w8, MAX_OFFSET_WIDTH);
                if vy && vx {
                    for dst in LAST_FRAME..=ALTREF_FRAME {
                        let ref_to_dst = dist(order_hint, order_hints[dst as usize]);
                        let p = get_mv_projection(mv, ref_to_dst, ref_offset);
                        f.motion_field[dst as usize][(pos_y8 * w8 + pos_x8) as usize] = p;
                    }
                }
            }
        }
        true
    }

    /// The motion field motion vector storage process (7.19), kept on the
    /// 8x8 grid (row and col odd).
    fn motion_vector_storage(&self, f: &FrameCtx) -> (Vec<i8>, Vec<Mv>) {
        let w8 = f.mi_cols >> 1;
        let h8 = f.mi_rows >> 1;
        let mut refs = vec![NONE as i8; w8 * h8];
        let mut mvs = vec![[0i32; 2]; w8 * h8];
        let seq = &f.seq;
        for y8 in 0..h8 {
            for x8 in 0..w8 {
                let row = 2 * y8 + 1;
                let col = 2 * x8 + 1;
                let m = &f.mi[row * f.ms + col];
                for list in 0..2 {
                    let r = m.ref_frame[list] as i32;
                    if r > INTRA_FRAME {
                        let ref_idx = f.hdr.ref_frame_idx[(r - LAST_FRAME) as usize];
                        let dist = get_relative_dist(seq, self.ref_state.order_hint[ref_idx], f.hdr.order_hint);
                        if dist < 0 {
                            let mv = m.mv[list];
                            if mv[0].abs() <= REFMVS_LIMIT && mv[1].abs() <= REFMVS_LIMIT {
                                refs[y8 * w8 + x8] = r as i8;
                                mvs[y8 * w8 + x8] = mv;
                            }
                        }
                    }
                }
            }
        }
        (refs, mvs)
    }
}

/// The get MV projection process (7.9.3).
fn get_mv_projection(mv: Mv, numerator: i32, denominator: i32) -> Mv {
    let clipped_den = denominator.min(MAX_FRAME_DISTANCE);
    let clipped_num = clip3(-MAX_FRAME_DISTANCE, MAX_FRAME_DISTANCE, numerator);
    let mut out = [0i32; 2];
    for i in 0..2 {
        let scaled = round2signed(mv[i] * clipped_num * DIV_MULT[clipped_den as usize], 14);
        out[i] = clip3(-(1 << 14) + 1, (1 << 14) - 1, scaled);
    }
    out
}

/// `project()` of the get block position process (7.9.4): the position and
/// whether it is valid.
fn project_pos(v8: i32, delta: i32, dst_sign: i32, max8: i32, max_off8: i32) -> (i32, bool) {
    let base8 = (v8 >> 3) << 3;
    let offset8 = if delta >= 0 {
        delta >> (3 + 1 + MI_SIZE_LOG2)
    } else {
        -((-delta) >> (3 + 1 + MI_SIZE_LOG2))
    };
    let v = v8 + dst_sign * offset8;
    let valid = !(v < 0 || v >= max8 || v < base8 - max_off8 || v >= base8 + 8 + max_off8);
    (v, valid)
}

/// `count_units_in_frame( unitSize, frameSize )`.
pub(crate) fn count_units_in_frame(unit_size: usize, frame_size: usize) -> usize {
    ((frame_size + (unit_size >> 1)) / unit_size).max(1)
}

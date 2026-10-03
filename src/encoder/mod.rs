//! The AV1 encoder: key frames and inter frames, one tile per frame.
//!
//! Each frame is coded by the decoder's own tile walker running in encode
//! mode (see `encoder::tile`): the encoder plants its decisions — the
//! partition, modes, motion vector and quantised levels of each block —
//! and the walker codes them with the normative contexts and CDF
//! adaptation while reconstructing exactly as the decoder will. The frame
//! header is written, then parsed back with the decoder's parser, so the
//! two cannot disagree; the in-loop filters and the reference update are
//! the decoder's.
//!
//! What it uses: 64x64 superblocks split down to 8x8 (and 4x4 at frame
//! edges); intra DC, V, H, smooth, Paeth and two directional modes; inter
//! prediction from the previous frame with NEWMV (full-pel search and
//! quarter-sample refinement), NEARESTMV and GLOBALMV; DCT; a fixed
//! quantiser per frame or simple rate control; the loop filter.

pub(crate) mod fwd;
pub(crate) mod rdo;
pub(crate) mod tile;

use std::sync::Arc;

use crate::bits::{BitReader, BitWriter};
use crate::consts::*;
use crate::decoder::Decoder;
use crate::decoder::tile::TileDecoder;
use crate::frame::{ChromaFormat, ColorInfo, Frame, HdrMetadata};
use crate::header::FrameHeader;
use crate::obu::{ColorConfig, SequenceHeader, write_obu};
use crate::tables::AC_QLOOKUP;
use crate::{Error, Result};

/// The default [`Config::speed`].
pub const DEFAULT_SPEED: u32 = 4;

/// Encoder settings.
#[derive(Debug, Clone)]
pub struct Config {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Bits per sample: 8 or 10 (profile 0, 4:2:0).
    pub bit_depth: u32,
    /// The quantiser index, 1 (finest) to 255 (coarsest); the starting
    /// point under rate control.
    pub quantizer: u32,
    /// A key frame every this many frames (1: all key frames).
    pub keyframe_interval: u32,
    /// Target bits per frame for rate control; `None` for a fixed
    /// quantiser.
    pub target_bits_per_frame: Option<u64>,
    /// Loop filter level (0 to 63) at `quantizer`; scaled with it.
    pub loop_filter: Option<u32>,
    /// Half-width of the full-pel motion search, in pixels.
    pub search_range: i32,
    /// The colour description written into the sequence header's
    /// `color_config()`: primaries, transfer characteristics and matrix
    /// (ITU-T H.273 code points), range and chroma sample position. All
    /// three code points 2 (unspecified, the default) writes
    /// `color_description_present_flag` = 0.
    pub color: ColorInfo,
    /// HDR metadata, written as metadata OBUs (`METADATA_TYPE_HDR_CLL`,
    /// `METADATA_TYPE_HDR_MDCV`) after the sequence header of every key
    /// frame's temporal unit.
    pub hdr: HdrMetadata,
    /// Encoder effort, 0 (slowest, best) to 10 (fastest): how much of the
    /// rate-distortion search runs (see [`Tools`] for what it switches).
    pub speed: u32,
    /// Tile columns, as a log2 (0: one tile column; clamped to what the
    /// frame allows). Tiles are coded in parallel.
    pub tile_cols_log2: u32,
    /// Which coding tools the encoder uses; `Tools::for_speed(speed)` by
    /// default.
    pub tools: Tools,
}

/// The encoder's coding tools, each switchable (for measurement, or to
/// trade quality for speed). [`Tools::for_speed`] gives each speed's set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tools {
    /// Rate-distortion search: partitions and block modes chosen by trial
    /// coding, the rate from the CDFs in force. Off: a variance test and
    /// SATD (the fastest).
    pub rdo: bool,
    /// Horizontal and vertical partitions (searched with `rdo`).
    pub partition_rect: bool,
    /// 8x8 blocks split to 4x4.
    pub partition_4x4: bool,
    /// Leave a block that codes flat (no residual) unsplit.
    pub prune_split: bool,
    /// Try skipping the residual of inter blocks.
    pub rd_skip: bool,
    /// `TX_MODE_SELECT`: transform sizes below the block size, searched.
    pub tx_size: bool,
    /// Luma transform types searched by trial coding (per transform block).
    pub tx_type_rd: bool,
    /// How many luma transform types the search tries (in a fixed order of
    /// usefulness: DCT, ADST, the mixes, identity, the 1D ones, the
    /// flipped ones).
    pub tx_types: u8,
    /// The full transform sets (`reduced_tx_set` = 0).
    pub full_tx_set: bool,
    /// Intra luma modes trialled per block (the best by SATD).
    pub intra_candidates: u8,
    /// Inter modes / vectors trialled per block.
    pub inter_candidates: u8,
}

impl Tools {
    /// The tools a speed setting uses (0 slowest to 10 fastest).
    pub fn for_speed(speed: u32) -> Self {
        let s = speed.min(10);
        Tools {
            rdo: s <= 8,
            partition_rect: s <= 5,
            partition_4x4: s <= 2,
            prune_split: s >= 3,
            rd_skip: s <= 6,
            tx_size: s <= 6,
            tx_type_rd: s <= 8,
            tx_types: match s {
                0..=1 => 16,
                2..=3 => 7,
                4..=5 => 5,
                _ => 4,
            },
            full_tx_set: s <= 3,
            intra_candidates: match s {
                0..=1 => 5,
                2..=4 => 3,
                5..=6 => 2,
                _ => 1,
            },
            inter_candidates: match s {
                0..=1 => 5,
                2..=4 => 3,
                5..=6 => 2,
                _ => 1,
            },
        }
    }

    /// Sets the switch called `name` (a field name; numbers as 0 / 1 or a
    /// count); false if there is none.
    pub fn set(&mut self, name: &str, value: u32) -> bool {
        let b = value != 0;
        match name {
            "rdo" => self.rdo = b,
            "partition_rect" => self.partition_rect = b,
            "partition_4x4" => self.partition_4x4 = b,
            "prune_split" => self.prune_split = b,
            "rd_skip" => self.rd_skip = b,
            "tx_size" => self.tx_size = b,
            "tx_type_rd" => self.tx_type_rd = b,
            "tx_types" => self.tx_types = value.min(16) as u8,
            "full_tx_set" => self.full_tx_set = b,
            "intra_candidates" => self.intra_candidates = value.min(13) as u8,
            "inter_candidates" => self.inter_candidates = value.min(16) as u8,
            _ => return false,
        }
        true
    }
}

impl Config {
    /// Settings for a `width` x `height` 8-bit 4:2:0 stream at quantiser
    /// 100, speed [`DEFAULT_SPEED`].
    pub fn new(width: u32, height: u32) -> Self {
        Config {
            width,
            height,
            bit_depth: 8,
            quantizer: 100,
            keyframe_interval: 60,
            target_bits_per_frame: None,
            loop_filter: None,
            search_range: 16,
            color: ColorInfo::default(),
            hdr: HdrMetadata::default(),
            speed: DEFAULT_SPEED,
            tile_cols_log2: 0,
            tools: Tools::for_speed(DEFAULT_SPEED),
        }
    }
}

/// Encodes frames into AV1 temporal units.
pub struct Encoder {
    cfg: Config,
    seq: Arc<SequenceHeader>,
    /// The decoder whose reference state and in-loop filters the encoder
    /// shares; its output is the encoder's reconstruction.
    dec: Decoder,
    frame_num: u64,
    /// Frames since the last key frame.
    since_key: u64,
    /// The next frame is to be a key frame whatever the interval says.
    force_key: bool,
    /// Whether the last frame encoded was a key frame.
    last_key: bool,
    q: f64,
    recon: Option<Frame>,
}

impl Encoder {
    /// An encoder with the given settings.
    pub fn new(cfg: Config) -> Self {
        let bits = |v: u32| 32 - (v.max(1) - 1).leading_zeros().min(31);
        let w_bits = bits(cfg.width).max(1);
        let h_bits = bits(cfg.height).max(1);
        let seq = SequenceHeader {
            seq_profile: 0,
            still_picture: false,
            reduced_still_picture_header: false,
            timing_info_present: false,
            decoder_model_info_present: false,
            equal_picture_interval: false,
            buffer_removal_time_length_minus_1: 0,
            frame_presentation_time_length_minus_1: 0,
            operating_points_cnt_minus_1: 0,
            operating_point_idc: vec![0],
            seq_level_idx: vec![31],
            decoder_model_present_for_this_op: vec![false],
            op_idc: 0,
            frame_width_bits: w_bits,
            frame_height_bits: h_bits,
            max_frame_width: cfg.width,
            max_frame_height: cfg.height,
            frame_id_numbers_present: false,
            delta_frame_id_length_minus_2: 0,
            additional_frame_id_length_minus_1: 0,
            use_128x128_superblock: false,
            enable_filter_intra: false,
            enable_intra_edge_filter: true,
            enable_interintra_compound: false,
            enable_masked_compound: false,
            enable_warped_motion: false,
            enable_dual_filter: false,
            enable_order_hint: true,
            enable_jnt_comp: false,
            enable_ref_frame_mvs: false,
            seq_force_screen_content_tools: 0,
            seq_force_integer_mv: SELECT_INTEGER_MV,
            order_hint_bits: 7,
            enable_superres: false,
            enable_cdef: false,
            enable_restoration: false,
            color: ColorConfig {
                bit_depth: cfg.bit_depth,
                mono_chrome: false,
                num_planes: 3,
                color_primaries: cfg.color.color_primaries,
                transfer_characteristics: cfg.color.transfer_characteristics,
                matrix_coefficients: cfg.color.matrix_coefficients,
                color_range: cfg.color.full_range,
                subsampling_x: 1,
                subsampling_y: 1,
                chroma_sample_position: cfg.color.chroma_sample_position,
                separate_uv_delta_q: false,
                color_description_present: cfg.color.color_primaries != CP_UNSPECIFIED
                    || cfg.color.transfer_characteristics != TC_UNSPECIFIED
                    || cfg.color.matrix_coefficients != MC_UNSPECIFIED,
            },
            film_grain_params_present: false,
        };
        let q = cfg.quantizer.clamp(1, 255) as f64;
        Encoder {
            cfg,
            seq: Arc::new(seq),
            dec: Decoder::new(),
            frame_num: 0,
            since_key: 0,
            force_key: false,
            last_key: false,
            q,
            recon: None,
        }
    }

    /// The reconstruction of the last frame encoded: what a decoder will
    /// output for it.
    pub fn reconstruction(&self) -> Option<&Frame> {
        self.recon.as_ref()
    }

    /// Makes the next frame a key frame (with its sequence header and
    /// metadata OBUs, so the stream can be entered there), whatever the
    /// key frame interval says; the interval restarts from it. Nothing
    /// else is reset: the rate controller carries on.
    pub fn force_keyframe(&mut self) {
        self.force_key = true;
    }

    /// Whether the next frame will be a key frame.
    pub fn next_is_keyframe(&self) -> bool {
        self.force_key
            || self.frame_num == 0
            || self.since_key >= self.cfg.keyframe_interval.max(1) as u64
    }

    /// Whether the last frame encoded was a key frame.
    pub fn last_was_keyframe(&self) -> bool {
        self.last_key
    }

    /// The configuration.
    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// The quantiser index the next frame will use.
    pub fn quantizer(&self) -> u32 {
        self.q.round().clamp(1.0, 255.0) as u32
    }

    /// Encodes one frame and returns its temporal unit (temporal delimiter,
    /// sequence header on key frames, frame OBU).
    pub fn encode(&mut self, frame: &Frame) -> Result<Vec<u8>> {
        let cfg = &self.cfg;
        if frame.width != cfg.width || frame.height != cfg.height {
            return Err(Error::invalid("frame size differs from the configuration"));
        }
        if frame.bit_depth != cfg.bit_depth || frame.chroma != ChromaFormat::Yuv420 {
            return Err(Error::invalid(
                "the encoder takes 4:2:0 frames of the configured bit depth",
            ));
        }
        if cfg.bit_depth != 8 && cfg.bit_depth != 10 {
            return Err(Error::invalid("bit depth must be 8 or 10"));
        }
        check_color(&cfg.color)?;
        let key = self.next_is_keyframe();
        let qidx = self.quantizer();
        let header = self.write_frame_header(key, qidx);
        // Parse it back with the decoder's parser: the frame state is then
        // exactly what a decoder will set up.
        let seq = self.seq.clone();
        if self.dec.seq.is_none() || key {
            self.dec.seq = Some(seq.clone());
        }
        let mut r = BitReader::new(&header);
        let hdr = FrameHeader::parse(&mut r, &seq, &mut self.dec.ref_state, 0, 0)?;
        let mut f = self.dec.setup_frame(seq.clone(), hdr)?;
        let enc = self.enc_ctx(&f, frame, key, qidx);
        let (tile, saved) = {
            let mut td = TileDecoder::new_encoder(&mut f, enc, 0, 0);
            td.decode_tile()?;
            let saved = if !td.f.hdr.disable_frame_end_update_cdf {
                Some(td.cdf.clone())
            } else {
                None
            };
            let crate::symbol::Coder::Enc(e) = td.sd else {
                unreachable!("encode mode")
            };
            (e.finish(), saved)
        };
        f.saved_cdfs = saved;
        self.dec.shown.clear();
        self.dec.finish_frame(f)?;
        self.recon = self.dec.shown.pop();
        if let Some(r) = self.recon.as_mut() {
            // What a decoder reports after this temporal unit's metadata.
            r.hdr = self.cfg.hdr;
        }
        // The temporal unit.
        let mut out = Vec::new();
        write_obu(&mut out, OBU_TEMPORAL_DELIMITER, &[]);
        if key {
            write_obu(&mut out, OBU_SEQUENCE_HEADER, &seq.write());
            if let Some(c) = &self.cfg.hdr.content_light {
                write_obu(&mut out, OBU_METADATA, &crate::obu::write_hdr_cll(c));
            }
            if let Some(m) = &self.cfg.hdr.mastering_display {
                write_obu(&mut out, OBU_METADATA, &crate::obu::write_hdr_mdcv(m));
            }
        }
        let mut payload = header;
        payload.extend_from_slice(&tile);
        write_obu(&mut out, OBU_FRAME, &payload);
        self.frame_num += 1;
        self.since_key = if key { 1 } else { self.since_key + 1 };
        self.force_key = false;
        self.last_key = key;
        self.rate_control(out.len() as u64 * 8, key);
        Ok(out)
    }

    fn rate_control(&mut self, bits: u64, key: bool) {
        let Some(target) = self.cfg.target_bits_per_frame else {
            return;
        };
        // Key frames cost several times an inter frame; aim them higher.
        let target = if key { target * 4 } else { target } as f64;
        let ratio = (bits.max(1) as f64 / target.max(1.0)).log2();
        self.q = (self.q + 12.0 * ratio.clamp(-2.0, 2.0)).clamp(1.0, 255.0);
    }

    fn enc_ctx(
        &self,
        f: &crate::decoder::FrameCtx,
        frame: &Frame,
        key: bool,
        qidx: u32,
    ) -> Box<tile::EncCtx> {
        let mut src = Vec::new();
        let mut stride = Vec::new();
        for p in 0..3 {
            let cur = &f.cur.planes[p];
            let pl = frame.planes[p];
            let (w, h) = (pl.width as usize, pl.height as usize);
            let mut v = vec![0u16; cur.stride * cur.h];
            for y in 0..cur.h {
                let sy = y.min(h - 1);
                for x in 0..cur.stride {
                    let sx = x.min(w - 1);
                    v[y * cur.stride + x] = frame.sample(p, sx as u32, sy as u32);
                }
            }
            src.push(v);
            stride.push(cur.stride);
        }
        let bdi = ((self.cfg.bit_depth - 8) >> 1) as usize;
        let qstep = AC_QLOOKUP[bdi][qidx as usize] as f64 / (1 << (self.cfg.bit_depth - 8)) as f64;
        let scale = (1 << (self.cfg.bit_depth - 8)) as f64;
        let step = qstep * scale / 8.0;
        Box::new(tile::EncCtx {
            src,
            stride,
            coefs: Box::new([0; 1024]),
            lambda: 0.4 * qstep * scale,
            rd_lambda: rd_lambda_factor() * step * step,
            refs: if key { Vec::new() } else { vec![LAST_FRAME] },
            search_range: self.cfg.search_range,
            tools: self.cfg.tools,
            rdo: Default::default(),
            res: Vec::new(),
            fc: Vec::new(),
        })
    }

    /// `uncompressed_header()` for the encoder's frames, byte aligned.
    fn write_frame_header(&self, key: bool, qidx: u32) -> Vec<u8> {
        let seq = &self.seq;
        let mut w = BitWriter::new();
        w.flag(false); // show_existing_frame
        w.f(2, if key { KEY_FRAME } else { INTER_FRAME });
        w.flag(true); // show_frame
        if !key {
            w.flag(false); // error_resilient_mode
        }
        w.flag(false); // disable_cdf_update
        // allow_screen_content_tools: seq_force_screen_content_tools = 0.
        w.flag(false); // frame_size_override_flag
        w.f(
            seq.order_hint_bits,
            (self.frame_num & ((1 << seq.order_hint_bits) - 1)) as u32,
        );
        if !key {
            w.f(3, 0); // primary_ref_frame: LAST_FRAME's slot
            w.f(8, 1); // refresh_frame_flags: slot 0
            w.flag(false); // frame_refs_short_signaling
            for _ in 0..REFS_PER_FRAME {
                w.f(3, 0); // ref_frame_idx
            }
            // frame_size_with_refs() is not used (no override): frame_size()
            // has nothing to write, then render_size().
            w.flag(false); // render_and_frame_size_different
            w.flag(false); // allow_high_precision_mv
            w.flag(false); // is_filter_switchable
            w.f(2, EIGHTTAP);
            w.flag(false); // is_motion_mode_switchable
        } else {
            w.flag(false); // render_and_frame_size_different
        }
        w.flag(false); // disable_frame_end_update_cdf
        // tile_info(): uniform, one tile column and row when possible.
        let mi_cols = 2 * ((self.cfg.width + 7) >> 3);
        let mi_rows = 2 * ((self.cfg.height + 7) >> 3);
        let sb_cols = (mi_cols + 15) >> 4;
        let sb_rows = (mi_rows + 15) >> 4;
        let tl = crate::header::tile_log2;
        let min_cols = tl(MAX_TILE_WIDTH >> 6, sb_cols);
        let max_cols = tl(1, sb_cols.min(MAX_TILE_COLS));
        let max_rows = tl(1, sb_rows.min(MAX_TILE_ROWS));
        w.flag(true); // uniform_tile_spacing_flag
        if min_cols < max_cols {
            w.flag(false);
        }
        let min_tiles = min_cols.max(tl(MAX_TILE_AREA >> 12, sb_rows * sb_cols));
        let min_rows = min_tiles.saturating_sub(min_cols);
        if min_rows < max_rows {
            w.flag(false);
        }
        // quantization_params()
        w.f(8, qidx);
        w.flag(false); // DeltaQYDc
        w.flag(false); // DeltaQUDc
        w.flag(false); // DeltaQUAc
        w.flag(false); // using_qmatrix
        w.flag(false); // segmentation_enabled
        w.flag(false); // delta_q_present
        // loop_filter_params()
        let lf = self
            .cfg
            .loop_filter
            .unwrap_or_else(|| ((qidx as f64) * 0.18 + 2.0).min(40.0) as u32);
        w.f(6, lf);
        w.f(6, lf);
        if lf != 0 {
            w.f(6, lf / 2 + 1);
            w.f(6, lf / 2 + 1);
        }
        w.f(3, 0); // loop_filter_sharpness
        w.flag(false); // loop_filter_delta_enabled
        // read_tx_mode(): TX_MODE_SELECT when transform sizes are searched.
        w.flag(self.cfg.tools.tx_size && self.cfg.tools.rdo);
        if !key {
            w.flag(false); // reference_select
            // skip_mode not allowed without reference_select.
        }
        w.flag(!(self.cfg.tools.full_tx_set && self.cfg.tools.rdo)); // reduced_tx_set
        if !key {
            for _ in LAST_FRAME..=ALTREF_FRAME {
                w.flag(false); // is_global
            }
        }
        w.byte_align();
        w.finish()
    }
}

/// Whether the encoder can write this colour description: code points in
/// range, and none that profile 0's 4:2:0 cannot carry (the identity
/// matrix, and the sRGB triple that implies 4:4:4).
fn check_color(c: &ColorInfo) -> Result<()> {
    if c.color_primaries > 255 || c.transfer_characteristics > 255 || c.matrix_coefficients > 255 {
        return Err(Error::invalid("colour code points are 8-bit values"));
    }
    if c.chroma_sample_position > 3 {
        return Err(Error::invalid("chroma_sample_position is 0 to 3"));
    }
    if c.matrix_coefficients == MC_IDENTITY {
        return Err(Error::invalid(
            "the identity matrix (RGB) needs 4:4:4; the encoder writes 4:2:0",
        ));
    }
    Ok(())
}

/// The rate-distortion Lagrange multiplier as a multiple of the squared
/// quantiser step (`AV1_RD_LAMBDA` overrides it, for tuning).
fn rd_lambda_factor() -> f64 {
    std::env::var("AV1_RD_LAMBDA")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.1)
}

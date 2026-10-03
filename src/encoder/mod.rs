//! The AV1 encoder: key frames and inter frames.
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
//! Decisions are searched by trial coding (`encoder::rdo`); the in-loop
//! filters' parameters (loop filter levels, CDEF strengths) are searched on
//! a first pass's reconstruction, and the frame is then coded again
//! replaying the first pass's decisions with them.

pub(crate) mod cdef;
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
    /// Threads coding tiles in parallel (with several tile columns); 1
    /// codes on the caller's thread.
    pub threads: usize,
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
    /// CDEF: strengths searched per frame, an index per 64x64 block.
    pub cdef: bool,
    /// The CDEF search tries more strengths.
    pub cdef_thorough: bool,
    /// Loop filter levels searched per frame (else derived from the
    /// quantiser).
    pub lf_search: bool,
    /// Several references: the last two frames and a golden frame (the
    /// key frame, then every 16th frame, coded finer).
    pub multi_ref: bool,
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
            cdef: s <= 8,
            cdef_thorough: s <= 3,
            lf_search: s <= 8,
            multi_ref: s <= 6,
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
            "cdef" => self.cdef = b,
            "cdef_thorough" => self.cdef_thorough = b,
            "lf_search" => self.lf_search = b,
            "multi_ref" => self.multi_ref = b,
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
            threads: 1,
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
    /// The frame number each reference slot holds.
    slot_frame: [u64; NUM_REF_FRAMES],
    /// The slots of `LAST_FRAME` and `LAST2_FRAME`.
    last_slot: usize,
    last2_slot: usize,
}

/// The slot the golden frame (the key frame, or a periodic boosted frame)
/// lives in when several references are used.
const GOLDEN_SLOT: usize = 2;

/// What one frame's header says, as the encoder chooses it.
#[derive(Clone, Debug)]
struct FrameParams {
    key: bool,
    qidx: u32,
    refresh: u8,
    /// `ref_frame_idx[ LAST_FRAME..ALTREF_FRAME ]`.
    ref_slots: [usize; REFS_PER_FRAME],
    /// The references inter prediction may search (distinct frames).
    search_refs: Vec<i32>,
    /// `loop_filter_level[ 0..4 ]`.
    lf: [u32; 4],
    cdef: Option<cdef::CdefParams>,
    tx_select: bool,
    reduced_tx_set: bool,
    tile_cols_log2: u32,
}

/// What a second coding pass replays.
struct Replay {
    /// The first pass's decision log, per tile.
    logs: Vec<Vec<rdo::Decision>>,
    /// Each 64x64 block's CDEF index.
    cdef: Option<Vec<i8>>,
}

/// The coded frame: its state for the in-loop filters, and its tiles.
struct Coded {
    f: crate::decoder::FrameCtx,
    header: Vec<u8>,
    tiles: Vec<Vec<u8>>,
    logs: Vec<Vec<rdo::Decision>>,
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
            enable_cdef: cfg.tools.cdef,
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
        let mut dec = Decoder::new();
        dec.set_threads(cfg.threads.max(1));
        Encoder {
            cfg,
            seq: Arc::new(seq),
            dec,
            frame_num: 0,
            since_key: 0,
            force_key: false,
            last_key: false,
            q,
            recon: None,
            slot_frame: [0; NUM_REF_FRAMES],
            last_slot: 0,
            last2_slot: 1,
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
    /// sequence header and metadata on key frames, frame OBU).
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
        let mut p = self.frame_params(key);
        let seq = self.seq.clone();
        if self.dec.seq.is_none() || key {
            self.dec.seq = Some(seq.clone());
        }
        let (src, stride) = self.source_planes(frame);
        let src = Arc::new(src);
        let tools = self.cfg.tools;
        let mut coded = self.code_frame(&p, &src, &stride, None)?;
        if tools.lf_search || tools.cdef {
            // The in-loop filters' parameters, chosen on the first pass's
            // reconstruction; then the frame again with them, replaying the
            // first pass's decisions.
            let mut f = coded.f;
            if tools.lf_search && self.cfg.loop_filter.is_none() {
                p.lf = search_lf(&mut f, &src, &stride, p.lf);
            }
            f.hdr.loop_filter_level = p.lf.map(|l| l as i32);
            if p.lf[0] != 0 || p.lf[1] != 0 {
                crate::decoder::postfilter::loop_filter_threads(&mut f, self.cfg.threads.max(1));
            }
            let mut cdef_table = None;
            if tools.cdef {
                let lambda = rd_lambda(&self.cfg, p.qidx);
                let (params, table) = cdef::search(&f, &src, &stride, lambda, tools.cdef_thorough);
                p.cdef = Some(params);
                cdef_table = Some(table);
            }
            let replay = Replay {
                logs: std::mem::take(&mut coded.logs),
                cdef: cdef_table,
            };
            coded = self.code_frame(&p, &src, &stride, Some(replay))?;
        }
        let Coded {
            f, header, tiles, ..
        } = coded;
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
        let n = tiles.len();
        if n > 1 {
            // tile_start_and_end_present_flag = 0, then byte alignment.
            payload.push(0);
        }
        for (i, t) in tiles.iter().enumerate() {
            if i + 1 < n {
                payload.extend_from_slice(&((t.len() - 1) as u32).to_le_bytes());
            }
            payload.extend_from_slice(t);
        }
        write_obu(&mut out, OBU_FRAME, &payload);
        // The reference slots now hold this frame.
        for i in 0..NUM_REF_FRAMES {
            if (p.refresh >> i) & 1 != 0 {
                self.slot_frame[i] = self.frame_num;
            }
        }
        if key {
            self.last_slot = 0;
            self.last2_slot = 1;
        } else if tools.multi_ref {
            std::mem::swap(&mut self.last_slot, &mut self.last2_slot);
        }
        self.frame_num += 1;
        self.since_key = if key { 1 } else { self.since_key + 1 };
        self.force_key = false;
        self.last_key = key;
        self.rate_control(out.len() as u64 * 8, key);
        Ok(out)
    }

    /// The header parameters of the next frame (before the in-loop filter
    /// search).
    fn frame_params(&self, key: bool) -> FrameParams {
        let tools = self.cfg.tools;
        let mut qidx = self.quantizer();
        let mut refresh = 0xFFu8;
        let mut ref_slots = [0usize; REFS_PER_FRAME];
        let mut search_refs = Vec::new();
        if !key {
            if tools.multi_ref {
                // This frame replaces the older of the two last frames;
                // every GOLDEN_INTERVAL frames it is also the golden frame,
                // coded finer.
                let golden = self.since_key.is_multiple_of(GOLDEN_INTERVAL);
                refresh = 1 << self.last2_slot;
                if golden {
                    refresh |= 1 << GOLDEN_SLOT;
                    qidx = qidx.saturating_sub(golden_boost(qidx)).max(1);
                }
                ref_slots = [
                    self.last_slot,
                    self.last2_slot,
                    self.last2_slot,
                    GOLDEN_SLOT,
                    GOLDEN_SLOT,
                    GOLDEN_SLOT,
                    GOLDEN_SLOT,
                ];
                search_refs.push(LAST_FRAME);
                let lf = self.slot_frame[self.last_slot];
                let l2 = self.slot_frame[self.last2_slot];
                let g = self.slot_frame[GOLDEN_SLOT];
                if l2 != lf {
                    search_refs.push(LAST2_FRAME);
                }
                if g != lf && g != l2 {
                    search_refs.push(GOLDEN_FRAME);
                }
            } else {
                refresh = 1;
                search_refs.push(LAST_FRAME);
            }
        }
        let lf = self
            .cfg
            .loop_filter
            .unwrap_or_else(|| ((qidx as f64) * 0.18 + 2.0).min(40.0) as u32);
        let chroma_lf = if lf != 0 { lf / 2 + 1 } else { 0 };
        FrameParams {
            key,
            qidx,
            refresh,
            ref_slots,
            search_refs,
            lf: [lf, lf, chroma_lf, chroma_lf],
            cdef: if tools.cdef {
                Some(cdef::CdefParams::off())
            } else {
                None
            },
            tx_select: tools.tx_size && tools.rdo,
            reduced_tx_set: !(tools.full_tx_set && tools.rdo),
            tile_cols_log2: self.cfg.tile_cols_log2,
        }
    }

    /// The source planes, padded to the frame buffers' size by repeating
    /// the last column and row.
    fn source_planes(&self, frame: &Frame) -> (Vec<Vec<u16>>, Vec<usize>) {
        let mi_cols = 2 * ((self.cfg.width as usize + 7) >> 3);
        let mi_rows = 2 * ((self.cfg.height as usize + 7) >> 3);
        let aw = (mi_cols * MI_SIZE + 127) & !127;
        let ah = (mi_rows * MI_SIZE + 127) & !127;
        let mut src = Vec::new();
        let mut stride = Vec::new();
        for p in 0..3 {
            let (sw, sh) = if p == 0 {
                (aw + 32, ah + 32)
            } else {
                ((aw >> 1) + 32, (ah >> 1) + 32)
            };
            let pl = frame.planes[p];
            let (w, h) = (pl.width as usize, pl.height as usize);
            let mut v = vec![0u16; sw * sh];
            for y in 0..sh {
                let sy = y.min(h - 1);
                let row = &mut v[y * sw..(y + 1) * sw];
                for (x, o) in row.iter_mut().enumerate() {
                    *o = frame.sample(p, x.min(w - 1) as u32, sy as u32);
                }
            }
            src.push(v);
            stride.push(sw);
        }
        (src, stride)
    }

    /// Writes the header for `p`, parses it back with the decoder's parser
    /// (the frame state is then exactly what a decoder sets up) and codes
    /// every tile: searching, or replaying an earlier pass.
    fn code_frame(
        &mut self,
        p: &FrameParams,
        src: &Arc<Vec<Vec<u16>>>,
        stride: &[usize],
        replay: Option<Replay>,
    ) -> Result<Coded> {
        let header = self.write_frame_header(p);
        let seq = self.seq.clone();
        let mut r = BitReader::new(&header);
        let hdr = FrameHeader::parse(&mut r, &seq, &mut self.dec.ref_state, 0, 0)?;
        let mut f = self.dec.setup_frame(seq, hdr)?;
        let ti = f.hdr.tile_info.clone();
        let num_tiles = ti.cols * ti.rows;
        let mut replay = replay;
        let mut jobs: Vec<std::sync::Mutex<Option<Box<tile::EncCtx>>>> = Vec::new();
        for t in 0..num_tiles {
            let mut enc = self.enc_ctx(p, src, stride);
            if let Some(rp) = replay.as_mut() {
                enc.rdo.replay = std::mem::take(&mut rp.logs[t]).into();
                enc.cdef_table = rp.cdef.clone();
            }
            jobs.push(std::sync::Mutex::new(Some(enc)));
        }
        // One tile: the bytes, the decision log, the CDFs if it is the tile
        // the frame keeps them from.
        type TileOut = (
            Vec<u8>,
            Vec<rdo::Decision>,
            Option<Box<crate::cdf::CdfContext>>,
        );
        let code_tile = |f: &mut crate::decoder::FrameCtx, t: usize| -> Result<TileOut> {
            let enc = jobs[t].lock().expect("job").take().expect("job");
            let mut td = TileDecoder::new_encoder(f, enc, t / ti.cols, t % ti.cols);
            td.decode_tile()?;
            let saved = if !td.f.hdr.disable_frame_end_update_cdf && t == ti.context_update_tile_id
            {
                Some(td.cdf.clone())
            } else {
                None
            };
            let log = std::mem::take(&mut td.enc.as_mut().expect("encode mode").rdo.log);
            let crate::symbol::Coder::Enc(e) = td.sd else {
                unreachable!("encode mode")
            };
            Ok((e.finish(), log, saved))
        };
        let threads = self.cfg.threads.max(1).min(num_tiles);
        let outs: Vec<Result<TileOut>> = if threads > 1 {
            // Tiles are independent: each is coded into a frame state of its
            // own, then copied in.
            let nums: Vec<usize> = (0..num_tiles).collect();
            crate::decoder::tiles_in_parallel(&mut f, &nums, threads, code_tile)
        } else {
            (0..num_tiles).map(|t| code_tile(&mut f, t)).collect()
        };
        let mut tiles = Vec::with_capacity(num_tiles);
        let mut logs = Vec::with_capacity(num_tiles);
        for o in outs {
            let (bytes, log, saved) = o?;
            if saved.is_some() {
                f.saved_cdfs = saved;
            }
            tiles.push(bytes);
            logs.push(log);
        }
        Ok(Coded {
            f,
            header,
            tiles,
            logs,
        })
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
        p: &FrameParams,
        src: &Arc<Vec<Vec<u16>>>,
        stride: &[usize],
    ) -> Box<tile::EncCtx> {
        let bdi = ((self.cfg.bit_depth - 8) >> 1) as usize;
        let qstep =
            AC_QLOOKUP[bdi][p.qidx as usize] as f64 / (1 << (self.cfg.bit_depth - 8)) as f64;
        let scale = (1 << (self.cfg.bit_depth - 8)) as f64;
        Box::new(tile::EncCtx {
            src: src.clone(),
            stride: stride.to_vec(),
            coefs: Box::new([0; 1024]),
            lambda: 0.4 * qstep * scale,
            rd_lambda: rd_lambda(&self.cfg, p.qidx),
            refs: p.search_refs.clone(),
            search_range: self.cfg.search_range,
            tools: self.cfg.tools,
            rdo: Default::default(),
            res: Vec::new(),
            fc: Vec::new(),
            cdef_table: None,
        })
    }

    /// `uncompressed_header()` for the encoder's frames, byte aligned.
    fn write_frame_header(&self, p: &FrameParams) -> Vec<u8> {
        let seq = &self.seq;
        let key = p.key;
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
            w.f(3, 0); // primary_ref_frame: LAST_FRAME
            w.f(8, p.refresh as u32); // refresh_frame_flags
            w.flag(false); // frame_refs_short_signaling
            for &s in &p.ref_slots {
                w.f(3, s as u32); // ref_frame_idx
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
        // tile_info(): uniform spacing, the tile columns asked for (at
        // least as many as the frame needs).
        let mi_cols = 2 * ((self.cfg.width + 7) >> 3);
        let mi_rows = 2 * ((self.cfg.height + 7) >> 3);
        let sb_cols = (mi_cols + 15) >> 4;
        let sb_rows = (mi_rows + 15) >> 4;
        let tl = crate::header::tile_log2;
        let min_cols = tl(MAX_TILE_WIDTH >> 6, sb_cols);
        let max_cols = tl(1, sb_cols.min(MAX_TILE_COLS));
        let max_rows = tl(1, sb_rows.min(MAX_TILE_ROWS));
        let cols_log2 = p.tile_cols_log2.clamp(min_cols, max_cols);
        w.flag(true); // uniform_tile_spacing_flag
        for _ in min_cols..cols_log2 {
            w.flag(true); // increment_tile_cols_log2
        }
        if cols_log2 < max_cols {
            w.flag(false);
        }
        let min_tiles = min_cols.max(tl(MAX_TILE_AREA >> 12, sb_rows * sb_cols));
        let min_rows = min_tiles.saturating_sub(cols_log2);
        if min_rows < max_rows {
            w.flag(false); // increment_tile_rows_log2
        }
        if cols_log2 + min_rows > 0 {
            w.f(cols_log2 + min_rows, 0); // context_update_tile_id
            w.f(2, 3); // tile_size_bytes_minus_1
        }
        // quantization_params()
        w.f(8, p.qidx);
        w.flag(false); // DeltaQYDc
        w.flag(false); // DeltaQUDc
        w.flag(false); // DeltaQUAc
        w.flag(false); // using_qmatrix
        w.flag(false); // segmentation_enabled
        w.flag(false); // delta_q_present
        // loop_filter_params()
        w.f(6, p.lf[0]);
        w.f(6, p.lf[1]);
        if p.lf[0] != 0 || p.lf[1] != 0 {
            w.f(6, p.lf[2]);
            w.f(6, p.lf[3]);
        }
        w.f(3, 0); // loop_filter_sharpness
        w.flag(false); // loop_filter_delta_enabled
        // cdef_params()
        if let Some(c) = &p.cdef {
            w.f(2, c.damping_minus_3);
            w.f(2, c.bits);
            for i in 0..1usize << c.bits {
                w.f(4, c.y[i].0);
                w.f(2, c.y[i].1);
                w.f(4, c.uv[i].0);
                w.f(2, c.uv[i].1);
            }
        }
        // read_tx_mode(): TX_MODE_SELECT when transform sizes are searched.
        w.flag(p.tx_select);
        if !key {
            w.flag(false); // reference_select
            // skip_mode not allowed without reference_select.
        }
        w.flag(p.reduced_tx_set); // reduced_tx_set
        if !key {
            for _ in LAST_FRAME..=ALTREF_FRAME {
                w.flag(false); // is_global
            }
        }
        w.byte_align();
        w.finish()
    }
}

/// How often a boosted golden frame is coded with several references.
const GOLDEN_INTERVAL: u64 = 16;

/// How much finer the golden frame's quantiser is.
fn golden_boost(qidx: u32) -> u32 {
    (qidx / 6).clamp(4, 24)
}

/// The rate-distortion Lagrange multiplier at quantiser `qidx`.
fn rd_lambda(cfg: &Config, qidx: u32) -> f64 {
    let bdi = ((cfg.bit_depth - 8) >> 1) as usize;
    let qstep = AC_QLOOKUP[bdi][qidx as usize] as f64 / (1 << (cfg.bit_depth - 8)) as f64;
    let scale = (1 << (cfg.bit_depth - 8)) as f64;
    let step = qstep * scale / 8.0;
    rd_lambda_factor() * step * step
}

/// The loop filter levels with the least error: scaled versions of the
/// starting ones, each plane's chosen on its own error. `f.cur` is left
/// unfiltered.
fn search_lf(
    f: &mut crate::decoder::FrameCtx,
    src: &[Vec<u16>],
    stride: &[usize],
    start: [u32; 4],
) -> [u32; 4] {
    let saved = f.cur.clone();
    let mut best = [(u64::MAX, 0u32); 3];
    for scale in [0.0, 0.5, 0.75, 1.0, 1.25, 1.5] {
        let lv = |l: u32| ((l as f64 * scale).round() as u32).min(63);
        let levels = [lv(start[0]), lv(start[1]), lv(start[2]), lv(start[3])];
        f.hdr.loop_filter_level = levels.map(|l| l as i32);
        if levels[0] != 0 || levels[1] != 0 {
            crate::decoder::postfilter::loop_filter(f);
        }
        for (p, b) in best.iter_mut().enumerate() {
            let e = plane_sse(f, src, stride, p);
            if e < b.0 {
                *b = (e, if p == 0 { levels[0] } else { levels[1 + p] });
            }
        }
        f.cur.planes.clone_from(&saved.planes);
    }
    // Chroma is filtered only when luma is.
    if best[0].1 == 0 {
        return [0; 4];
    }
    [best[0].1, best[0].1, best[1].1, best[2].1]
}

/// Squared error of plane `p` of `f.cur` against the source, inside the
/// frame.
fn plane_sse(f: &crate::decoder::FrameCtx, src: &[Vec<u16>], stride: &[usize], p: usize) -> u64 {
    let (ssx, ssy) = f.plane_ss(p);
    let w = (f.hdr.frame_width as usize + ssx) >> ssx;
    let h = (f.hdr.frame_height as usize + ssy) >> ssy;
    let pl = &f.cur.planes[p];
    let mut s = 0u64;
    for y in 0..h {
        let a = &src[p][y * stride[p]..y * stride[p] + w];
        let b = &pl.data[y * pl.stride..y * pl.stride + w];
        for (&u, &v) in a.iter().zip(b) {
            let d = u as i64 - v as i64;
            s += (d * d) as u64;
        }
    }
    s
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

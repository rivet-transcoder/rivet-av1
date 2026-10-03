//! Named constants of the AV1 specification (section 3 and the enumerations
//! of section 6), under the specification's names.

#![allow(dead_code)]

pub(crate) const REFS_PER_FRAME: usize = 7;
pub(crate) const TOTAL_REFS_PER_FRAME: usize = 8;
pub(crate) const NUM_REF_FRAMES: usize = 8;
pub(crate) const MAX_SEGMENTS: usize = 8;
pub(crate) const SEG_LVL_ALT_Q: usize = 0;
pub(crate) const SEG_LVL_ALT_LF_Y_V: usize = 1;
pub(crate) const SEG_LVL_REF_FRAME: usize = 5;
pub(crate) const SEG_LVL_SKIP: usize = 6;
pub(crate) const SEG_LVL_GLOBALMV: usize = 7;
pub(crate) const SEG_LVL_MAX: usize = 8;
pub(crate) const MAX_LOOP_FILTER: i32 = 63;
pub(crate) const PRIMARY_REF_NONE: usize = 7;
pub(crate) const MI_SIZE: usize = 4;
pub(crate) const MI_SIZE_LOG2: usize = 2;
pub(crate) const MAX_TILE_WIDTH: u32 = 4096;
pub(crate) const MAX_TILE_AREA: u32 = 4096 * 2304;
pub(crate) const MAX_TILE_ROWS: u32 = 64;
pub(crate) const MAX_TILE_COLS: u32 = 64;
pub(crate) const SUPERRES_NUM: u32 = 8;
pub(crate) const SUPERRES_DENOM_MIN: u32 = 9;
pub(crate) const SUPERRES_DENOM_BITS: u32 = 3;
pub(crate) const SUPERRES_FILTER_BITS: u32 = 6;
pub(crate) const SUPERRES_FILTER_TAPS: i32 = 8;
pub(crate) const SUPERRES_FILTER_OFFSET: i32 = 3;
pub(crate) const SUPERRES_SCALE_BITS: u32 = 14;
pub(crate) const SUPERRES_SCALE_MASK: i64 = (1 << 14) - 1;
pub(crate) const SUPERRES_EXTRA_BITS: u32 = 8;
pub(crate) const SELECT_SCREEN_CONTENT_TOOLS: u32 = 2;
pub(crate) const SELECT_INTEGER_MV: u32 = 2;
pub(crate) const RESTORATION_TILESIZE_MAX: usize = 256;
pub(crate) const WARPEDMODEL_PREC_BITS: u32 = 16;
pub(crate) const GM_ABS_TRANS_BITS: u32 = 12;
pub(crate) const GM_ABS_TRANS_ONLY_BITS: u32 = 9;
pub(crate) const GM_ABS_ALPHA_BITS: u32 = 12;
pub(crate) const GM_ALPHA_PREC_BITS: u32 = 15;
pub(crate) const GM_TRANS_PREC_BITS: u32 = 6;
pub(crate) const GM_TRANS_ONLY_PREC_BITS: u32 = 3;
pub(crate) const DIV_LUT_PREC_BITS: i32 = 14;
pub(crate) const DIV_LUT_BITS: i32 = 8;
pub(crate) const LEAST_SQUARES_SAMPLES_MAX: usize = 8;
pub(crate) const LS_MV_MAX: i32 = 256;
pub(crate) const WARPEDMODEL_TRANS_CLAMP: i32 = 1 << 23;
pub(crate) const WARPEDMODEL_NONDIAGAFFINE_CLAMP: i32 = 1 << 13;
pub(crate) const WARPEDPIXEL_PREC_SHIFTS: i32 = 1 << 6;
pub(crate) const WARPEDDIFF_PREC_BITS: i32 = 10;
pub(crate) const WARP_PARAM_REDUCE_BITS: i32 = 6;
pub(crate) const MAX_SB_SIZE: i32 = 128;
pub(crate) const MASK_MASTER_SIZE: usize = 64;
pub(crate) const REF_SCALE_SHIFT: i32 = 14;
pub(crate) const SUBPEL_BITS: i32 = 4;
pub(crate) const SUBPEL_MASK: i32 = 15;
pub(crate) const SCALE_SUBPEL_BITS: i32 = 10;
pub(crate) const MV_BORDER: i32 = 128;
pub(crate) const FILTER_BITS: i32 = 7;
pub(crate) const MAX_FRAME_DISTANCE: i32 = 31;
pub(crate) const MAX_OFFSET_WIDTH: i32 = 8;
pub(crate) const MAX_OFFSET_HEIGHT: i32 = 0;
pub(crate) const MFMV_STACK_SIZE: i32 = 3;
pub(crate) const REFMVS_LIMIT: i32 = (1 << 12) - 1;
pub(crate) const MAX_REF_MV_STACK_SIZE: usize = 8;
pub(crate) const REF_CAT_LEVEL: u32 = 640;
pub(crate) const INTRABC_DELAY_PIXELS: i32 = 256;
pub(crate) const INTRABC_DELAY_SB64: i32 = 4;
pub(crate) const MAX_VARTX_DEPTH: usize = 2;
pub(crate) const MAX_ANGLE_DELTA: i32 = 3;
pub(crate) const ANGLE_STEP: i32 = 3;
pub(crate) const PALETTE_COLORS: usize = 8;
pub(crate) const PALETTE_NUM_NEIGHBORS: usize = 3;
pub(crate) const DELTA_Q_SMALL: u32 = 3;
pub(crate) const DELTA_LF_SMALL: u32 = 3;
pub(crate) const FRAME_LF_COUNT: usize = 4;
pub(crate) const NUM_BASE_LEVELS: u32 = 2;
pub(crate) const COEFF_BASE_RANGE: u32 = 12;
pub(crate) const BR_CDF_SIZE: u32 = 4;
pub(crate) const SIG_COEF_CONTEXTS: usize = 42;
pub(crate) const SIG_COEF_CONTEXTS_EOB: usize = 4;
pub(crate) const SIG_COEF_CONTEXTS_2D: usize = 26;
pub(crate) const SIG_REF_DIFF_OFFSET_NUM: usize = 5;
pub(crate) const INTRA_FILTER_SCALE_BITS: u32 = 4;
pub(crate) const SGRPROJ_PARAMS_BITS: u32 = 4;
pub(crate) const SGRPROJ_PRJ_SUBEXP_K: u32 = 4;
pub(crate) const SGRPROJ_PRJ_BITS: i32 = 7;
pub(crate) const SGRPROJ_RST_BITS: i32 = 4;
pub(crate) const SGRPROJ_MTABLE_BITS: i32 = 20;
pub(crate) const SGRPROJ_RECIP_BITS: i32 = 12;
pub(crate) const SGRPROJ_SGR_BITS: i32 = 8;
pub(crate) const EC_PROB_SHIFT: u32 = 6;
pub(crate) const EC_MIN_PROB: u32 = 4;
pub(crate) const CLASS0_SIZE: i32 = 2;
pub(crate) const MV_INTRABC_CONTEXT: usize = 1;
pub(crate) const COMP_NEWMV_CTXS: usize = 5;
pub(crate) const TX_SIZES: usize = 5;

// Frame types (6.8.2).
pub(crate) const KEY_FRAME: u32 = 0;
pub(crate) const INTER_FRAME: u32 = 1;
pub(crate) const INTRA_ONLY_FRAME: u32 = 2;
pub(crate) const SWITCH_FRAME: u32 = 3;

// Reference frames (6.10.24); NONE is -1.
pub(crate) const NONE: i32 = -1;
pub(crate) const INTRA_FRAME: i32 = 0;
pub(crate) const LAST_FRAME: i32 = 1;
pub(crate) const LAST2_FRAME: i32 = 2;
pub(crate) const LAST3_FRAME: i32 = 3;
pub(crate) const GOLDEN_FRAME: i32 = 4;
pub(crate) const BWDREF_FRAME: i32 = 5;
pub(crate) const ALTREF2_FRAME: i32 = 6;
pub(crate) const ALTREF_FRAME: i32 = 7;

// Block sizes (6.10.4).
pub(crate) const BLOCK_4X4: usize = 0;
pub(crate) const BLOCK_4X8: usize = 1;
pub(crate) const BLOCK_8X4: usize = 2;
pub(crate) const BLOCK_8X8: usize = 3;
pub(crate) const BLOCK_8X16: usize = 4;
pub(crate) const BLOCK_16X8: usize = 5;
pub(crate) const BLOCK_16X16: usize = 6;
pub(crate) const BLOCK_16X32: usize = 7;
pub(crate) const BLOCK_32X16: usize = 8;
pub(crate) const BLOCK_32X32: usize = 9;
pub(crate) const BLOCK_32X64: usize = 10;
pub(crate) const BLOCK_64X32: usize = 11;
pub(crate) const BLOCK_64X64: usize = 12;
pub(crate) const BLOCK_64X128: usize = 13;
pub(crate) const BLOCK_128X64: usize = 14;
pub(crate) const BLOCK_128X128: usize = 15;
pub(crate) const BLOCK_4X16: usize = 16;
pub(crate) const BLOCK_16X4: usize = 17;
pub(crate) const BLOCK_8X32: usize = 18;
pub(crate) const BLOCK_32X8: usize = 19;
pub(crate) const BLOCK_16X64: usize = 20;
pub(crate) const BLOCK_64X16: usize = 21;
pub(crate) const BLOCK_SIZES: usize = 22;
pub(crate) const BLOCK_INVALID: usize = 22;

// Partitions (6.10.4).
pub(crate) const PARTITION_NONE: usize = 0;
pub(crate) const PARTITION_HORZ: usize = 1;
pub(crate) const PARTITION_VERT: usize = 2;
pub(crate) const PARTITION_SPLIT: usize = 3;
pub(crate) const PARTITION_HORZ_A: usize = 4;
pub(crate) const PARTITION_HORZ_B: usize = 5;
pub(crate) const PARTITION_VERT_A: usize = 6;
pub(crate) const PARTITION_VERT_B: usize = 7;
pub(crate) const PARTITION_HORZ_4: usize = 8;
pub(crate) const PARTITION_VERT_4: usize = 9;

// Transform sizes (6.10.16).
pub(crate) const TX_4X4: usize = 0;
pub(crate) const TX_8X8: usize = 1;
pub(crate) const TX_16X16: usize = 2;
pub(crate) const TX_32X32: usize = 3;
pub(crate) const TX_64X64: usize = 4;
pub(crate) const TX_4X8: usize = 5;
pub(crate) const TX_8X4: usize = 6;
pub(crate) const TX_8X16: usize = 7;
pub(crate) const TX_16X8: usize = 8;
pub(crate) const TX_16X32: usize = 9;
pub(crate) const TX_32X16: usize = 10;
pub(crate) const TX_32X64: usize = 11;
pub(crate) const TX_64X32: usize = 12;
pub(crate) const TX_4X16: usize = 13;
pub(crate) const TX_16X4: usize = 14;
pub(crate) const TX_8X32: usize = 15;
pub(crate) const TX_32X8: usize = 16;
pub(crate) const TX_16X64: usize = 17;
pub(crate) const TX_64X16: usize = 18;
pub(crate) const TX_SIZES_ALL: usize = 19;

// TxMode (6.8.21).
pub(crate) const ONLY_4X4: u32 = 0;
pub(crate) const TX_MODE_LARGEST: u32 = 1;
pub(crate) const TX_MODE_SELECT: u32 = 2;

// Transform types (section 3).
pub(crate) const DCT_DCT: usize = 0;
pub(crate) const ADST_DCT: usize = 1;
pub(crate) const DCT_ADST: usize = 2;
pub(crate) const ADST_ADST: usize = 3;
pub(crate) const FLIPADST_DCT: usize = 4;
pub(crate) const DCT_FLIPADST: usize = 5;
pub(crate) const FLIPADST_FLIPADST: usize = 6;
pub(crate) const ADST_FLIPADST: usize = 7;
pub(crate) const FLIPADST_ADST: usize = 8;
pub(crate) const IDTX: usize = 9;
pub(crate) const V_DCT: usize = 10;
pub(crate) const H_DCT: usize = 11;
pub(crate) const V_ADST: usize = 12;
pub(crate) const H_ADST: usize = 13;
pub(crate) const V_FLIPADST: usize = 14;
pub(crate) const H_FLIPADST: usize = 15;

// Transform sets (6.10.19).
pub(crate) const TX_SET_DCTONLY: usize = 0;
pub(crate) const TX_SET_INTRA_1: usize = 1;
pub(crate) const TX_SET_INTRA_2: usize = 2;
pub(crate) const TX_SET_INTER_1: usize = 1;
pub(crate) const TX_SET_INTER_2: usize = 2;
pub(crate) const TX_SET_INTER_3: usize = 3;

pub(crate) const TX_CLASS_2D: usize = 0;
pub(crate) const TX_CLASS_HORIZ: usize = 1;
pub(crate) const TX_CLASS_VERT: usize = 2;

// Intra modes (6.10.22) and YMode inter values (6.10.23).
pub(crate) const DC_PRED: usize = 0;
pub(crate) const V_PRED: usize = 1;
pub(crate) const H_PRED: usize = 2;
pub(crate) const D45_PRED: usize = 3;
pub(crate) const D135_PRED: usize = 4;
pub(crate) const D113_PRED: usize = 5;
pub(crate) const D157_PRED: usize = 6;
pub(crate) const D203_PRED: usize = 7;
pub(crate) const D67_PRED: usize = 8;
pub(crate) const SMOOTH_PRED: usize = 9;
pub(crate) const SMOOTH_V_PRED: usize = 10;
pub(crate) const SMOOTH_H_PRED: usize = 11;
pub(crate) const PAETH_PRED: usize = 12;
pub(crate) const UV_CFL_PRED: usize = 13;
pub(crate) const NEARESTMV: usize = 14;
pub(crate) const NEARMV: usize = 15;
pub(crate) const GLOBALMV: usize = 16;
pub(crate) const NEWMV: usize = 17;
pub(crate) const NEAREST_NEARESTMV: usize = 18;
pub(crate) const NEAR_NEARMV: usize = 19;
pub(crate) const NEAREST_NEWMV: usize = 20;
pub(crate) const NEW_NEARESTMV: usize = 21;
pub(crate) const NEAR_NEWMV: usize = 22;
pub(crate) const NEW_NEARMV: usize = 23;
pub(crate) const GLOBAL_GLOBALMV: usize = 24;
pub(crate) const NEW_NEWMV: usize = 25;

// Filter intra modes (6.10.24).
pub(crate) const FILTER_DC_PRED: usize = 0;

// Interintra modes.
pub(crate) const II_DC_PRED: usize = 0;
pub(crate) const II_V_PRED: usize = 1;
pub(crate) const II_H_PRED: usize = 2;
pub(crate) const II_SMOOTH_PRED: usize = 3;

// Interpolation filters (6.8.9).
pub(crate) const EIGHTTAP: u32 = 0;
pub(crate) const EIGHTTAP_SMOOTH: u32 = 1;
pub(crate) const EIGHTTAP_SHARP: u32 = 2;
pub(crate) const BILINEAR: u32 = 3;
pub(crate) const SWITCHABLE: u32 = 4;

// Motion modes.
pub(crate) const SIMPLE: u32 = 0;
pub(crate) const OBMC: u32 = 1;
pub(crate) const LOCALWARP: u32 = 2;

// Compound types (6.10.28).
pub(crate) const COMPOUND_WEDGE: u32 = 0;
pub(crate) const COMPOUND_DIFFWTD: u32 = 1;
pub(crate) const COMPOUND_AVERAGE: u32 = 2;
pub(crate) const COMPOUND_INTRA: u32 = 3;
pub(crate) const COMPOUND_DISTANCE: u32 = 4;

// comp_ref_type.
pub(crate) const UNIDIR_COMP_REFERENCE: u32 = 0;
pub(crate) const BIDIR_COMP_REFERENCE: u32 = 1;

// Global motion types.
pub(crate) const IDENTITY: u32 = 0;
pub(crate) const TRANSLATION: u32 = 1;
pub(crate) const ROTZOOM: u32 = 2;
pub(crate) const AFFINE: u32 = 3;

// mv_joint.
pub(crate) const MV_JOINT_ZERO: u32 = 0;
pub(crate) const MV_JOINT_HNZVZ: u32 = 1;
pub(crate) const MV_JOINT_HZVNZ: u32 = 2;
pub(crate) const MV_JOINT_HNZVNZ: u32 = 3;

// Loop restoration types (6.10.15).
pub(crate) const RESTORE_NONE: u8 = 0;
pub(crate) const RESTORE_WIENER: u8 = 1;
pub(crate) const RESTORE_SGRPROJ: u8 = 2;
pub(crate) const RESTORE_SWITCHABLE: u8 = 3;

// CFL signs.
pub(crate) const CFL_SIGN_ZERO: u32 = 0;
pub(crate) const CFL_SIGN_NEG: u32 = 1;
pub(crate) const CFL_SIGN_POS: u32 = 2;

// OBU types (6.2.2).
pub(crate) const OBU_SEQUENCE_HEADER: u32 = 1;
pub(crate) const OBU_TEMPORAL_DELIMITER: u32 = 2;
pub(crate) const OBU_FRAME_HEADER: u32 = 3;
pub(crate) const OBU_TILE_GROUP: u32 = 4;
pub(crate) const OBU_METADATA: u32 = 5;
pub(crate) const OBU_FRAME: u32 = 6;
pub(crate) const OBU_REDUNDANT_FRAME_HEADER: u32 = 7;
pub(crate) const OBU_TILE_LIST: u32 = 8;
pub(crate) const OBU_PADDING: u32 = 15;

// Color config (6.4.2).
pub(crate) const CP_BT_709: u32 = 1;
pub(crate) const CP_UNSPECIFIED: u32 = 2;
pub(crate) const TC_UNSPECIFIED: u32 = 2;
pub(crate) const TC_SRGB: u32 = 13;
pub(crate) const MC_IDENTITY: u32 = 0;
pub(crate) const MC_UNSPECIFIED: u32 = 2;
pub(crate) const CSP_UNKNOWN: u32 = 0;

/// `Block_Width[ b ]`.
#[inline]
pub(crate) fn block_width(b: usize) -> usize {
    4 * crate::tables::NUM_4X4_BLOCKS_WIDE[b]
}

/// `Block_Height[ b ]`.
#[inline]
pub(crate) fn block_height(b: usize) -> usize {
    4 * crate::tables::NUM_4X4_BLOCKS_HIGH[b]
}

/// `Round2( x, n )`.
#[inline]
pub(crate) fn round2(x: i32, n: u32) -> i32 {
    if n == 0 { x } else { (x + (1 << (n - 1))) >> n }
}

/// `Round2( x, n )` on 64 bits.
#[inline]
pub(crate) fn round2_64(x: i64, n: u32) -> i64 {
    if n == 0 {
        x
    } else {
        (x + (1i64 << (n - 1))) >> n
    }
}

/// `Round2Signed( x, n )`.
#[inline]
pub(crate) fn round2signed(x: i32, n: u32) -> i32 {
    if x >= 0 { round2(x, n) } else { -round2(-x, n) }
}

/// `Round2Signed( x, n )` on 64 bits.
#[inline]
pub(crate) fn round2signed_64(x: i64, n: u32) -> i64 {
    if x >= 0 {
        round2_64(x, n)
    } else {
        -round2_64(-x, n)
    }
}

/// `Clip3( lo, hi, x )`.
#[inline]
pub(crate) fn clip3(lo: i32, hi: i32, x: i32) -> i32 {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

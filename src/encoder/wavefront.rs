//! The decision search of one tile on several threads: superblock rows in
//! a wavefront.
//!
//! Superblock `(r, c)` depends on its left neighbour (same row) and on
//! the row above up to its above-right neighbour `(r - 1, c + 1)`: the
//! mode info, contexts and reconstructed samples it predicts from. Each
//! worker codes whole rows into a frame state of its own; before a
//! superblock it copies in what the row above has finished since (from
//! the shared frame state, where each superblock is published when
//! done), and it starts a row once the row above is two superblocks
//! ahead. The symbol costs come from each row's own adaptation of the
//! CDFs (started from the row above's after its second superblock, as
//! a sequential coder would have them roughly), so the decisions are not
//! exactly the sequential search's: the encoder always codes the frame
//! again, sequentially, replaying them (`Replay`), and that pass alone
//! makes the bitstream.

use std::sync::{Condvar, Mutex};

use crate::Result;
use crate::cdf::CdfContext;
use crate::decoder::FrameCtx;
use crate::decoder::tile::TileDecoder;
use crate::encoder::rdo::Decision;
use crate::encoder::tile::EncCtx;

/// A row's CDFs (with the key frames' luma mode CDFs, kept apart).
type RowCdfs = (Box<CdfContext>, [[[u16; 14]; 5]; 5]);

/// What the rows share.
struct Shared<'a> {
    f: &'a mut FrameCtx,
    /// Superblocks finished per row.
    done: Vec<usize>,
    /// The above contexts as the finished superblocks left them (tile
    /// columns, in 4x4 units of each plane, like `TileDecoder`'s).
    above_level: [Vec<u8>; 3],
    above_dc: [Vec<u8>; 3],
    above_seg: Vec<u8>,
    /// Each row's CDFs after its second superblock.
    cdfs: Vec<Option<RowCdfs>>,
    /// Each superblock's decisions, in raster order.
    logs: Vec<Vec<Decision>>,
    failed: Option<crate::Error>,
}

/// Searches the decisions of tile `(tile_row, tile_col)` of `f` with
/// `threads` workers; returns its decision log (what a sequential search
/// would log, superblock by superblock) and leaves the reconstruction in
/// `f`.
pub(crate) fn search_tile(
    f: &mut FrameCtx,
    tile_row: usize,
    tile_col: usize,
    threads: usize,
    make_enc: &(dyn Fn() -> Box<EncCtx> + Sync),
) -> Result<Vec<Decision>> {
    let ti = f.hdr.tile_info.clone();
    let (r0, r1) = (ti.mi_row_starts[tile_row], ti.mi_row_starts[tile_row + 1]);
    let (c0, c1) = (ti.mi_col_starts[tile_col], ti.mi_col_starts[tile_col + 1]);
    let sb4 = if f.seq.use_128x128_superblock { 32 } else { 16 };
    let rows = (r1 - r0).div_ceil(sb4);
    let cols = (c1 - c0).div_ceil(sb4);
    let last_tile_row = tile_row + 1 == ti.rows;
    let last_tile_col = tile_col + 1 == ti.cols;
    let workers = threads.min(rows).max(1);
    let shards: Vec<FrameCtx> = (0..workers).map(|_| f.shard()).collect();
    let actx = |n: usize| -> [Vec<u8>; 3] { [vec![0; n], vec![0; n], vec![0; n]] };
    let width4 = f.mi_cols + 64;
    let shared = Mutex::new(Shared {
        f,
        done: vec![0; rows],
        above_level: actx(width4),
        above_dc: actx(width4),
        above_seg: vec![0; width4],
        cdfs: (0..rows).map(|_| None).collect(),
        logs: (0..rows * cols).map(|_| Vec::new()).collect(),
        failed: None,
    });
    let progress = Condvar::new();
    let next_row = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|s| {
        for mut shard in shards {
            let (shared, progress, next_row) = (&shared, &progress, &next_row);
            s.spawn(move || {
                loop {
                    let row = next_row.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if row >= rows {
                        break;
                    }
                    if let Err(e) = search_row(
                        &mut shard,
                        shared,
                        progress,
                        RowJob {
                            row,
                            rows,
                            cols,
                            sb4,
                            r0,
                            r1,
                            c0,
                            c1,
                            tile_row,
                            tile_col,
                            last_tile_row,
                            last_tile_col,
                        },
                        make_enc,
                    ) {
                        let mut sh = shared.lock().expect("wavefront state");
                        sh.failed.get_or_insert(e);
                        // Unblock every waiting row.
                        for d in sh.done.iter_mut() {
                            *d = usize::MAX;
                        }
                        progress.notify_all();
                        break;
                    }
                }
            });
        }
    });
    let sh = shared.into_inner().expect("wavefront state");
    if let Some(e) = sh.failed {
        return Err(e);
    }
    Ok(sh.logs.into_iter().flatten().collect())
}

/// One superblock row's place in the tile.
#[derive(Clone, Copy)]
struct RowJob {
    row: usize,
    rows: usize,
    cols: usize,
    sb4: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
    tile_row: usize,
    tile_col: usize,
    last_tile_row: bool,
    last_tile_col: bool,
}

fn search_row(
    shard: &mut FrameCtx,
    shared: &Mutex<Shared>,
    progress: &Condvar,
    j: RowJob,
    make_enc: &(dyn Fn() -> Box<EncCtx> + Sync),
) -> Result<()> {
    let mut td = TileDecoder::new_encoder(shard, make_enc(), j.tile_row, j.tile_col);
    td.begin_tile();
    td.begin_sb_row();
    let r = j.r0 + j.row * j.sb4;
    // The superblocks of the row above copied in so far.
    let mut pulled = 0;
    for k in 0..j.cols {
        let c = j.c0 + k * j.sb4;
        // Wait for the row above to reach the above-right superblock (and
        // for its CDFs, on the first superblock).
        let need = (k + 2).min(j.cols);
        {
            let mut sh = shared.lock().expect("wavefront state");
            if j.row > 0 {
                while sh.done[j.row - 1] < need {
                    sh = progress.wait(sh).expect("wavefront state");
                }
                if let Some(e) = sh.failed.as_ref() {
                    return Err(crate::Error::Unsupported(e.to_string()));
                }
                if k == 0
                    && let Some((cdf, ymode)) = sh.cdfs[j.row - 1].as_ref()
                {
                    td.cdf.clone_from(cdf);
                    td.intra_frame_y_mode_cdf = *ymode;
                }
                let ru = r - j.sb4;
                while pulled < need {
                    let cu = j.c0 + pulled * j.sb4;
                    let (cend, last_col) = sb_cols(&j, pulled);
                    td.f.merge_rect(&*sh.f, ru, r, cu, cend, false, last_col);
                    let ssx = td.f.ssx;
                    for p in 0..3 {
                        let sx = if p > 0 { ssx } else { 0 };
                        let (a, b) = (cu >> sx, (cend + sx) >> sx);
                        td.above_level_ctx[p][a..b].copy_from_slice(&sh.above_level[p][a..b]);
                        td.above_dc_ctx[p][a..b].copy_from_slice(&sh.above_dc[p][a..b]);
                    }
                    td.above_seg_pred_ctx[cu..cend].copy_from_slice(&sh.above_seg[cu..cend]);
                    pulled += 1;
                }
            }
        }
        let log_start = td.enc.as_ref().expect("encode mode").rdo.log.len();
        td.decode_sb(r, c)?;
        let seg = td
            .enc
            .as_mut()
            .expect("encode mode")
            .rdo
            .log
            .split_off(log_start);
        // Publish the superblock.
        let mut sh = shared.lock().expect("wavefront state");
        let (cend, last_col) = sb_cols(&j, k);
        let rend = (r + j.sb4).min(j.r1);
        let last_row = j.last_tile_row && j.row + 1 == j.rows;
        sh.f.merge_rect(td.f, r, rend, c, cend, last_row, last_col);
        for p in 0..3 {
            let sx = if p > 0 { td.f.ssx } else { 0 };
            let (a, b) = (c >> sx, (cend + sx) >> sx);
            sh.above_level[p][a..b].copy_from_slice(&td.above_level_ctx[p][a..b]);
            sh.above_dc[p][a..b].copy_from_slice(&td.above_dc_ctx[p][a..b]);
        }
        sh.above_seg[c..cend].copy_from_slice(&td.above_seg_pred_ctx[c..cend]);
        sh.logs[j.row * j.cols + k] = seg;
        if k + 1 == 2.min(j.cols) {
            sh.cdfs[j.row] = Some((td.cdf.clone(), td.intra_frame_y_mode_cdf));
        }
        if sh.done[j.row] != usize::MAX {
            sh.done[j.row] = k + 1;
        }
        progress.notify_all();
    }
    Ok(())
}

/// The end column (4x4 units) of superblock `k` of the row, and whether
/// it is the frame's last.
fn sb_cols(j: &RowJob, k: usize) -> (usize, bool) {
    let cend = (j.c0 + (k + 1) * j.sb4).min(j.c1);
    (cend, j.last_tile_col && k + 1 == j.cols)
}

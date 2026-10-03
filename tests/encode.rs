//! The encoder: every temporal unit it writes must decode, with a fresh
//! decoder, to exactly the encoder's reconstruction; quality must track the
//! quantiser. The natural-content source is the decoded frames of a
//! committed test vector (tests/data).

mod common;

use av1::{ChromaFormat, Config, Decoder, Encoder, Frame};

fn psnr(a: &Frame, b: &Frame, plane: usize) -> f64 {
    let maxv = ((1u32 << a.bit_depth) - 1) as f64;
    let pl = a.planes[plane];
    let mut se = 0f64;
    for y in 0..pl.height {
        for x in 0..pl.width {
            let d = a.sample(plane, x, y) as f64 - b.sample(plane, x, y) as f64;
            se += d * d;
        }
    }
    let mse = se / (pl.width * pl.height) as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (maxv * maxv / mse).log10()
    }
}

/// `n` frames of natural video: the 4 frames of av1-1-b8-05-mv.ivf
/// (352x288), decoded, played forwards then backwards as needed.
fn natural(n: usize) -> Vec<Frame> {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/av1-1-b8-05-mv.ivf");
    let mut d = Decoder::new();
    let src: Vec<Frame> = common::packets(&p)
        .iter()
        .filter_map(|pk| d.decode(pk).unwrap())
        .collect();
    let period = 2 * src.len() - 2;
    (0..n)
        .map(|i| {
            let k = i % period;
            src[if k < src.len() { k } else { period - k }].clone()
        })
        .collect()
}

fn synthetic(w: u32, h: u32, t: u32, bit_depth: u32) -> Frame {
    let mut f = Frame::new(w, h, bit_depth, ChromaFormat::Yuv420);
    let shift = bit_depth - 8;
    for p in 0..3 {
        let pl = f.planes[p];
        for y in 0..pl.height {
            for x in 0..pl.width {
                // Luma pans 2 samples a frame, chroma (half resolution) 1.
                let xs = if p == 0 { x + 2 * t } else { x + t };
                let v = if p == 0 {
                    let ring = ((xs as f64 - 20.0).hypot(y as f64 - 14.0) / 3.0).sin() * 50.0;
                    (110.0 + ring + ((xs * 7 + y * 3) % 23) as f64) as u16
                } else {
                    (90 + (xs * 2 + y + p as u32 * 17) % 60) as u16
                };
                f.set_sample(p, x, y, v.min(255) << shift);
            }
        }
    }
    f
}

/// Encodes `frames`; decodes every packet with a fresh decoder and checks
/// it against the encoder's reconstruction. Returns the decoded frames and
/// the packet sizes.
fn round_trip(cfg: Config, frames: &[Frame]) -> (Vec<Frame>, Vec<usize>) {
    let mut enc = Encoder::new(cfg);
    let mut dec = Decoder::new();
    dec.set_strict(true);
    let mut out = Vec::new();
    let mut sizes = Vec::new();
    for (i, f) in frames.iter().enumerate() {
        let pkt = enc.encode(f).unwrap();
        sizes.push(pkt.len());
        let d = dec
            .decode(&pkt)
            .unwrap()
            .expect("every packet shows a frame");
        assert_eq!(
            &d,
            enc.reconstruction().unwrap(),
            "frame {i}: decoded frame differs from the encoder's reconstruction"
        );
        out.push(d);
    }
    (out, sizes)
}

#[test]
fn key_frames_decode_to_the_reconstruction() {
    let frames: Vec<Frame> = (0..3).map(|t| synthetic(64, 48, t, 8)).collect();
    for q in [20, 100, 200] {
        let mut cfg = Config::new(64, 48);
        cfg.quantizer = q;
        cfg.keyframe_interval = 1;
        let (dec, _) = round_trip(cfg, &frames);
        assert!(psnr(&frames[0], &dec[0], 0) > 20.0);
    }
}

#[test]
fn odd_sizes() {
    for (w, h) in [(1, 1), (7, 5), (66, 66), (130, 34), (16, 200)] {
        let frames: Vec<Frame> = (0..3).map(|t| synthetic(w, h, t, 8)).collect();
        let mut cfg = Config::new(w, h);
        cfg.quantizer = 60;
        round_trip(cfg, &frames);
    }
}

#[test]
fn inter_frames_decode_to_the_reconstruction() {
    let frames: Vec<Frame> = (0..6).map(|t| synthetic(96, 64, t, 8)).collect();
    let mut cfg = Config::new(96, 64);
    cfg.quantizer = 80;
    let (dec, sizes) = round_trip(cfg, &frames);
    // Panning content: inter frames are much cheaper than the key frame.
    assert!(sizes[1..].iter().all(|&s| s < sizes[0]), "{sizes:?}");
    for (s, d) in frames.iter().zip(&dec) {
        assert!(psnr(s, d, 0) > 30.0);
    }
}

#[test]
fn ten_bit() {
    let frames: Vec<Frame> = (0..3).map(|t| synthetic(64, 64, t, 10)).collect();
    let mut cfg = Config::new(64, 64);
    cfg.bit_depth = 10;
    cfg.quantizer = 90;
    let (dec, _) = round_trip(cfg, &frames);
    assert!(psnr(&frames[2], &dec[2], 0) > 30.0);
}

#[test]
fn natural_video_quality_tracks_the_quantiser() {
    let src = natural(6);
    let mut last_psnr = f64::INFINITY;
    let mut last_size = 0usize;
    for q in [30, 90, 160] {
        let mut cfg = Config::new(src[0].width, src[0].height);
        cfg.quantizer = q;
        let (dec, sizes) = round_trip(cfg, &src);
        let p: f64 = src
            .iter()
            .zip(&dec)
            .map(|(a, b)| psnr(a, b, 0))
            .sum::<f64>()
            / src.len() as f64;
        let size: usize = sizes.iter().sum();
        eprintln!("q {q}: {size} bytes, luma PSNR {p:.2} dB");
        assert!(p < last_psnr, "PSNR must fall as the quantiser rises");
        assert!(
            last_size == 0 || size < last_size,
            "size must fall as the quantiser rises"
        );
        last_psnr = p;
        last_size = size;
    }
}

#[test]
fn rate_control_tracks_the_target() {
    let src = natural(8);
    let mut cfg = Config::new(src[0].width, src[0].height);
    cfg.quantizer = 60;
    cfg.keyframe_interval = 100;
    let target = 20_000u64;
    cfg.target_bits_per_frame = Some(target);
    let (_, sizes) = round_trip(cfg, &src);
    let inter_bits: u64 =
        sizes[4..].iter().map(|&s| s as u64 * 8).sum::<u64>() / (sizes.len() - 4) as u64;
    eprintln!("rate control: sizes {sizes:?}, inter average {inter_bits} bits for target {target}");
    assert!(inter_bits < target * 3 && inter_bits > target / 3);
}

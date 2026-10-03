//! Encodes raw 8-bit 4:2:0 planar video (I420) to an AV1 IVF file.
//!
//! cargo run --release --example ivfenc -- input.yuv WIDTH HEIGHT output.ivf [QUANTIZER]

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!("usage: ivfenc input.yuv WIDTH HEIGHT output.ivf [QUANTIZER]");
        std::process::exit(2);
    }
    let data = std::fs::read(&args[1])?;
    let w: u32 = args[2].parse()?;
    let h: u32 = args[3].parse()?;
    let mut cfg = av1::Config::new(w, h);
    if let Some(q) = args.get(5) {
        cfg.quantizer = q.parse()?;
    }
    let mut enc = av1::Encoder::new(cfg);
    let frame_len = av1::Frame::new(w, h, 8, av1::ChromaFormat::Yuv420).data.len();
    let mut out = av1::ivf::IvfWriter::new(w as u16, h as u16, 30, 1);
    for (i, chunk) in data.chunks_exact(frame_len).enumerate() {
        let mut f = av1::Frame::new(w, h, 8, av1::ChromaFormat::Yuv420);
        f.data.copy_from_slice(chunk);
        let pkt = enc.encode(&f)?;
        out.frame(i as u64, &pkt);
    }
    std::fs::write(&args[4], out.finish())?;
    Ok(())
}

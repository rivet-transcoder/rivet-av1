//! Decodes an AV1 stream (IVF, or the first AV1 track of an MP4) to a Y4M
//! file: the sources of the encoder's rate-distortion measurements
//! (`examples/av1_rdcurve.rs`) come from here.
//!
//! cargo run --release --example av1toy4m -- input.{ivf,mp4} output.y4m [max_frames] [crop WxH]
//!
//! With `crop`, the centre `W`x`H` of every frame is kept.

use std::io::Write;

#[path = "common/mp4.rs"]
mod mp4;
use mp4::mp4_samples;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: av1toy4m input.{{ivf,mp4}} output.y4m [max_frames] [crop WxH]");
        std::process::exit(2);
    }
    let data = std::fs::read(&args[1])?;
    let max: usize = args.get(3).map_or(usize::MAX, |s| s.parse().unwrap());
    let crop: Option<(u32, u32)> = args.get(4).map(|s| {
        let (w, h) = s.split_once('x').expect("crop is WxH");
        (w.parse().unwrap(), h.parse().unwrap())
    });
    let packets: Vec<&[u8]> = if data.starts_with(b"DKIF") {
        av1::ivf::IvfReader::new(&data)?
            .map(|p| p.map(|p| p.data))
            .collect::<Result<_, _>>()?
    } else {
        mp4_samples(&data).ok_or("no av01 track")?
    };
    let mut dec = av1::Decoder::new();
    let mut out = std::io::BufWriter::new(std::fs::File::create(&args[2])?);
    let mut n = 0;
    'outer: for p in packets {
        for f in dec.decode_all(p)? {
            if f.chroma != av1::ChromaFormat::Yuv420 {
                return Err("only 4:2:0".into());
            }
            let (w, h) = crop.unwrap_or((f.width, f.height));
            let (x0, y0) = (((f.width - w) / 2) & !1, ((f.height - h) / 2) & !1);
            if n == 0 {
                let c = if f.bit_depth > 8 {
                    "C420p10"
                } else {
                    "C420jpeg"
                };
                writeln!(out, "YUV4MPEG2 W{w} H{h} F30:1 Ip A1:1 {c}")?;
            }
            out.write_all(b"FRAME\n")?;
            for p in 0..3 {
                let s = if p == 0 { 0 } else { 1 };
                for y in 0..(h + s) >> s {
                    for x in 0..(w + s) >> s {
                        let v = f.sample(p, (x0 >> s) + x, (y0 >> s) + y);
                        if f.bit_depth > 8 {
                            out.write_all(&v.to_le_bytes())?;
                        } else {
                            out.write_all(&[v as u8])?;
                        }
                    }
                }
            }
            n += 1;
            if n >= max {
                break 'outer;
            }
        }
    }
    eprintln!("{n} frames");
    Ok(())
}

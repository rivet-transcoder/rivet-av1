//! Decodes an IVF file and prints each shown frame's size and MD5.
//!
//! cargo run --release --example ivfdec -- input.ivf

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: ivfdec input.ivf");
        std::process::exit(2);
    }
    let data = std::fs::read(&args[1])?;
    let mut dec = av1::Decoder::new();
    let mut n = 0;
    for (i, pkt) in av1::ivf::IvfReader::new(&data)?.enumerate() {
        for f in dec.decode_all(pkt?.data)? {
            println!(
                "tu {i:4} frame {n:5} {}x{} {}-bit {:?}",
                f.width, f.height, f.bit_depth, f.chroma
            );
            n += 1;
        }
    }
    eprintln!("{n} frames");
    Ok(())
}

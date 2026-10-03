//! The Argon conformance streams (AOMedia's "Argon Streams AV1", v2.1):
//! for every stream with a reference MD5, the MD5 of all output frames,
//! concatenated as raw planar video, must match.
//!
//! The suite (7 GB, from the AOMedia "AV1 video decoder verification
//! tool" page) is not downloaded by the tests: set `ARGON_DIR` to its
//! unpacked `argon_coveragetool_av1_base_and_extended_profiles_v2.1`
//! directory. `ARGON_FILTER=substr` restricts the run to matching paths;
//! `ARGON_REPORT=file` writes one line per stream; `ARGON_LAYERS=1` adds
//! the per-operating-point (`layers/N`) variants; `ARGON_VERBOSE=1` prints
//! each result as it comes. Without `ARGON_DIR`
//! the test reports that it skipped.
//!
//! Each stream's `ref_cmd` script gives the options the reference output
//! was made with: the bitstream format (`--annexb`), the operating point
//! (`--oppoint=N`) and whether every layer's frames are output
//! (`--all-layers`). Large-scale-tile and error-resilience directories
//! (no MD5s, or tile list OBUs) are not run.

use std::path::{Path, PathBuf};

struct Case {
    stream: PathBuf,
    md5: String,
    annexb: bool,
    oppoint: usize,
    all_layers: bool,
    name: String,
}

fn cases(root: &Path) -> Vec<Case> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(root) else {
        return out;
    };
    let mut dirs: Vec<PathBuf> = rd
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    for dir in dirs {
        let dname = dir.file_name().unwrap().to_string_lossy().to_string();
        if dname.contains("large_scale_tile") || dname.contains("error") {
            continue;
        }
        let md5_dir = dir.join("md5_ref");
        let cmd_dir = dir.join("ref_cmd");
        let mut push = |md5_path: PathBuf, cmd_path: PathBuf, label: String| {
            let Ok(md5) = std::fs::read_to_string(&md5_path) else {
                return;
            };
            let Ok(cmd) = std::fs::read_to_string(&cmd_path) else {
                return;
            };
            let Some(line) = cmd.lines().find(|l| l.contains("aomdec")) else {
                return;
            };
            let Some(input) = line
                .split_whitespace()
                .find(|w| w.contains("ARGON_STREAMS_INPUT"))
            else {
                return;
            };
            let file = input.rsplit('/').next().unwrap().trim_matches('"');
            let oppoint = line
                .split_whitespace()
                .find_map(|w| w.strip_prefix("--oppoint="))
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            out.push(Case {
                stream: dir.join("streams").join(file),
                md5: md5.split_whitespace().next().unwrap_or("").to_string(),
                annexb: line.contains("--annexb"),
                oppoint,
                all_layers: line.contains("--all-layers"),
                name: label,
            });
        };
        if let Ok(rd) = std::fs::read_dir(&md5_dir) {
            let mut v: Vec<PathBuf> = rd
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.is_file())
                .collect();
            v.sort();
            for p in v {
                let stem = p.file_stem().unwrap().to_string_lossy().to_string();
                let label = format!("{dname}/{stem}");
                push(p, cmd_dir.join(format!("{stem}.sh")), label);
            }
        }
        let layers = std::env::var("ARGON_LAYERS").is_ok();
        if let Ok(rd) = std::fs::read_dir(md5_dir.join("layers")).map_err(|_| ()).and_then(|r| if layers { Ok(r) } else { Err(()) }) {
            let mut layers: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
            layers.sort();
            for l in layers {
                let n = l.file_name().unwrap().to_string_lossy().to_string();
                let Ok(rd) = std::fs::read_dir(&l) else {
                    continue;
                };
                let mut v: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
                v.sort();
                for p in v {
                    let stem = p.file_stem().unwrap().to_string_lossy().to_string();
                    let label = format!("{dname}/layers/{n}/{stem}");
                    push(
                        p,
                        cmd_dir
                            .join("layers")
                            .join(&n)
                            .join(format!("{stem}.sh")),
                        label,
                    );
                }
            }
        }
    }
    out
}

/// Decodes a case: `Ok(true)` when the output MD5 matches.
fn run(c: &Case) -> Result<bool, String> {
    let data = std::fs::read(&c.stream).map_err(|e| e.to_string())?;
    let mut dec = av1::Decoder::new();
    dec.set_operating_point(c.oppoint);
    let mut out = md5::Context::new();
    // Monochrome output as aomdec writes it with --rawvideo: we hash with
    // and without neutral chroma planes and accept either.
    let mut out_mono = md5::Context::new();
    let mut emit = |frames: Vec<av1::Frame>| {
        let frames = if c.all_layers {
            frames
        } else {
            frames.into_iter().last().into_iter().collect()
        };
        for f in frames {
            out.consume(f.packed());
            out_mono.consume(f.packed());
            if f.chroma == av1::ChromaFormat::Mono {
                let cw = f.width.div_ceil(2) as usize;
                let ch = f.height.div_ceil(2) as usize;
                let mid = 1u16 << (f.bit_depth - 1);
                let plane: Vec<u8> = if f.bit_depth > 8 {
                    std::iter::repeat_n(mid.to_le_bytes(), cw * ch)
                        .flatten()
                        .collect()
                } else {
                    vec![mid as u8; cw * ch]
                };
                out_mono.consume(&plane);
                out_mono.consume(&plane);
            }
        }
    };
    if c.annexb {
        for tu in av1::annexb_temporal_units(&data).map_err(|e| e.to_string())? {
            emit(dec.decode_annexb(tu).map_err(|e| e.to_string())?);
        }
    } else {
        // Section 5's low-overhead format: temporal units split at the
        // temporal delimiters.
        for tu in split_at_delimiters(&data) {
            emit(dec.decode_all(tu).map_err(|e| e.to_string())?);
        }
    }
    let a = format!("{:x}", out.compute());
    let b = format!("{:x}", out_mono.compute());
    Ok(a == c.md5 || b == c.md5)
}

/// Splits a low-overhead OBU stream into temporal units at its temporal
/// delimiter OBUs.
fn split_at_delimiters(data: &[u8]) -> Vec<&[u8]> {
    let mut starts = Vec::new();
    let mut pos = 0;
    while pos < data.len() {
        let h = data[pos];
        let ty = (h >> 3) & 15;
        let ext = (h >> 2) & 1;
        let has_size = (h >> 1) & 1;
        if ty == 2 {
            starts.push(pos);
        }
        let mut p = pos + 1 + ext as usize;
        if has_size == 0 {
            break;
        }
        let mut size = 0usize;
        for i in 0..8 {
            let Some(&b) = data.get(p) else {
                return vec![data];
            };
            p += 1;
            size |= ((b & 0x7f) as usize) << (7 * i);
            if b & 0x80 == 0 {
                break;
            }
        }
        pos = p + size;
    }
    if starts.is_empty() || starts[0] != 0 {
        starts.insert(0, 0);
    }
    let mut out = Vec::new();
    for (i, &s) in starts.iter().enumerate() {
        let e = starts.get(i + 1).copied().unwrap_or(data.len());
        out.push(&data[s..e]);
    }
    out
}

#[test]
fn argon() {
    let Ok(root) = std::env::var("ARGON_DIR") else {
        eprintln!("ARGON_DIR not set; Argon conformance streams skipped");
        return;
    };
    let mut cs = cases(Path::new(&root));
    if let Ok(f) = std::env::var("ARGON_FILTER") {
        cs.retain(|c| c.name.contains(&f));
    }
    let n = std::thread::available_parallelism().map_or(4, |n| n.get());
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..n {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= cs.len() {
                        break;
                    }
                    let c = &cs[i];
                    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(c)));
                    let r = match r {
                        Ok(r) => r,
                        Err(_) => Err("panic".to_string()),
                    };
                    if std::env::var("ARGON_VERBOSE").is_ok() {
                        eprintln!(
                            "{} {}",
                            if matches!(r, Ok(true)) {
                                "PASS"
                            } else {
                                "FAIL"
                            },
                            c.name
                        );
                    }
                    results.lock().unwrap().push((c.name.clone(), r));
                }
            });
        }
    });
    let mut results = results.into_inner().unwrap();
    results.sort_by(|a, b| a.0.cmp(&b.0));
    let mut report = String::new();
    let mut by_dir: std::collections::BTreeMap<String, (usize, usize)> = Default::default();
    for (name, r) in &results {
        let dir = name.split('/').next().unwrap().to_string()
            + if name.contains("/layers/") {
                "/layers"
            } else {
                ""
            };
        let e = by_dir.entry(dir).or_default();
        e.1 += 1;
        let line = match r {
            Ok(true) => {
                e.0 += 1;
                format!("PASS {name}\n")
            }
            Ok(false) => format!("FAIL {name}: md5 mismatch\n"),
            Err(m) => format!("FAIL {name}: {m}\n"),
        };
        report.push_str(&line);
    }
    let total_pass: usize = by_dir.values().map(|v| v.0).sum();
    for (d, (p, t)) in &by_dir {
        eprintln!("{d}: {p}/{t}");
    }
    eprintln!("Argon: {total_pass}/{} pass", results.len());
    if let Ok(f) = std::env::var("ARGON_REPORT") {
        std::fs::write(f, report).unwrap();
    }
}

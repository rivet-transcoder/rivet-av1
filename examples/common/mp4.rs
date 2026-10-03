//! Just enough ISO BMFF to take the AV1 samples out of an MP4, for the
//! examples.

/// The samples of the first `av01` track of an MP4: just enough ISO BMFF
/// (moov/trak/mdia/minf/stbl with stsz, stsc and stco or co64).
pub fn mp4_samples(data: &[u8]) -> Option<Vec<&[u8]>> {
    fn boxes(d: &[u8]) -> Vec<(&[u8], &[u8])> {
        let mut out = Vec::new();
        let mut p = 0;
        while p + 8 <= d.len() {
            let mut size = u32::from_be_bytes(d[p..p + 4].try_into().unwrap()) as usize;
            let typ = &d[p + 4..p + 8];
            let mut hdr = 8;
            if size == 1 && p + 16 <= d.len() {
                size = u64::from_be_bytes(d[p + 8..p + 16].try_into().unwrap()) as usize;
                hdr = 16;
            } else if size == 0 {
                size = d.len() - p;
            }
            if size < hdr || p + size > d.len() {
                break;
            }
            out.push((typ, &d[p + hdr..p + size]));
            p += size;
        }
        out
    }
    fn find<'a>(d: &'a [u8], t: &[u8]) -> Option<&'a [u8]> {
        boxes(d).into_iter().find(|(k, _)| *k == t).map(|(_, v)| v)
    }
    let be32 = |d: &[u8], p: usize| u32::from_be_bytes(d[p..p + 4].try_into().unwrap()) as usize;
    let moov = find(data, b"moov")?;
    for (t, trak) in boxes(moov) {
        if t != b"trak" {
            continue;
        }
        let stbl = find(find(find(trak, b"mdia")?, b"minf")?, b"stbl")?;
        let stsd = find(stbl, b"stsd")?;
        if stsd.len() < 16 || &stsd[12..16] != b"av01" {
            continue;
        }
        let stsz = find(stbl, b"stsz")?;
        let fixed = be32(stsz, 4);
        let count = be32(stsz, 8);
        let sizes: Vec<usize> = (0..count)
            .map(|i| {
                if fixed != 0 {
                    fixed
                } else {
                    be32(stsz, 12 + 4 * i)
                }
            })
            .collect();
        let offsets: Vec<usize> = if let Some(stco) = find(stbl, b"stco") {
            (0..be32(stco, 4)).map(|i| be32(stco, 8 + 4 * i)).collect()
        } else {
            let co64 = find(stbl, b"co64")?;
            (0..be32(co64, 4))
                .map(|i| {
                    u64::from_be_bytes(co64[8 + 8 * i..16 + 8 * i].try_into().unwrap()) as usize
                })
                .collect()
        };
        let stsc = find(stbl, b"stsc")?;
        let entries: Vec<(usize, usize)> = (0..be32(stsc, 4))
            .map(|i| (be32(stsc, 8 + 12 * i), be32(stsc, 12 + 12 * i)))
            .collect();
        let mut out = Vec::new();
        let mut s = 0;
        for (ci, &off) in offsets.iter().enumerate() {
            let chunk = ci + 1;
            let per = entries
                .iter()
                .rev()
                .find(|(first, _)| *first <= chunk)
                .map_or(1, |e| e.1);
            let mut o = off;
            for _ in 0..per {
                if s >= sizes.len() {
                    break;
                }
                out.push(data.get(o..o + sizes[s])?);
                o += sizes[s];
                s += 1;
            }
        }
        return Some(out);
    }
    None
}

//! A reader and writer for IVF, the minimal container the AOM tools use for
//! AV1: a 32-byte file header (FourCC `AV01`), then per frame a 12-byte
//! header (size, timestamp) and the temporal unit.

use crate::{Error, Result};

/// The IVF file header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IvfHeader {
    /// Codec FourCC, `AV01` for AV1.
    pub fourcc: [u8; 4],
    /// Width in pixels.
    pub width: u16,
    /// Height in pixels.
    pub height: u16,
    /// Time base denominator (frame rate numerator).
    pub rate: u32,
    /// Time base numerator.
    pub scale: u32,
    /// Number of frames, as the writer recorded it.
    pub frames: u32,
}

/// An IVF frame: its timestamp and data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IvfFrame<'a> {
    /// Presentation timestamp in time-base units.
    pub pts: u64,
    /// The temporal unit.
    pub data: &'a [u8],
}

/// Reads an IVF file held in memory.
pub struct IvfReader<'a> {
    data: &'a [u8],
    pos: usize,
    header: IvfHeader,
}

impl<'a> IvfReader<'a> {
    /// Parses the file header.
    pub fn new(data: &'a [u8]) -> Result<Self> {
        if data.len() < 32 || &data[0..4] != b"DKIF" {
            return Err(Error::InvalidInput("not an IVF file".into()));
        }
        let le16 = |o: usize| u16::from_le_bytes([data[o], data[o + 1]]);
        let le32 = |o: usize| u32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);
        let header_len = le16(6) as usize;
        let header = IvfHeader {
            fourcc: [data[8], data[9], data[10], data[11]],
            width: le16(12),
            height: le16(14),
            rate: le32(16),
            scale: le32(20),
            frames: le32(24),
        };
        Ok(IvfReader {
            data,
            pos: header_len.max(32),
            header,
        })
    }

    /// The file header.
    pub fn header(&self) -> &IvfHeader {
        &self.header
    }
}

impl<'a> Iterator for IvfReader<'a> {
    type Item = Result<IvfFrame<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos + 12 > self.data.len() {
            return None;
        }
        let d = self.data;
        let p = self.pos;
        let size = u32::from_le_bytes([d[p], d[p + 1], d[p + 2], d[p + 3]]) as usize;
        let pts = u64::from_le_bytes(d[p + 4..p + 12].try_into().unwrap());
        if p + 12 + size > d.len() {
            self.pos = d.len();
            return Some(Err(Error::InvalidInput(
                "IVF frame runs past the end of the file".into(),
            )));
        }
        self.pos = p + 12 + size;
        Some(Ok(IvfFrame {
            pts,
            data: &d[p + 12..p + 12 + size],
        }))
    }
}

/// Writes an IVF file to memory.
pub struct IvfWriter {
    out: Vec<u8>,
    frames: u32,
}

impl IvfWriter {
    /// Starts an AV1 IVF file of the given size and time base (`rate` /
    /// `scale` frames per second).
    pub fn new(width: u16, height: u16, rate: u32, scale: u32) -> Self {
        let mut out = Vec::with_capacity(4096);
        out.extend_from_slice(b"DKIF");
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&32u16.to_le_bytes());
        out.extend_from_slice(b"AV01");
        out.extend_from_slice(&width.to_le_bytes());
        out.extend_from_slice(&height.to_le_bytes());
        out.extend_from_slice(&rate.to_le_bytes());
        out.extend_from_slice(&scale.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        IvfWriter { out, frames: 0 }
    }

    /// Appends a frame.
    pub fn frame(&mut self, pts: u64, data: &[u8]) {
        self.out
            .extend_from_slice(&(data.len() as u32).to_le_bytes());
        self.out.extend_from_slice(&pts.to_le_bytes());
        self.out.extend_from_slice(data);
        self.frames += 1;
    }

    /// The finished file.
    pub fn finish(mut self) -> Vec<u8> {
        self.out[24..28].copy_from_slice(&self.frames.to_le_bytes());
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let mut w = IvfWriter::new(64, 48, 30, 1);
        w.frame(0, &[1, 2, 3]);
        w.frame(1, &[4]);
        let file = w.finish();
        let r = IvfReader::new(&file).unwrap();
        assert_eq!(r.header().frames, 2);
        assert_eq!(r.header().width, 64);
        assert_eq!(&r.header().fourcc, b"AV01");
        let frames: Vec<_> = r.map(|f| f.unwrap()).collect();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].data, &[1, 2, 3]);
        assert_eq!(frames[1].pts, 1);
    }
}

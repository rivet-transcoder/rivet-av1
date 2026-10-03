//! Bit-level reading and writing for the parts of AV1 that are not
//! arithmetic coded: OBU headers, sequence and frame headers (4.10, 8.1).

use crate::{Error, Result};

/// Reads bits most significant first (8.1). Reading past the end is an
/// error rather than a zero, so a truncated header is reported.
#[derive(Debug, Clone)]
pub(crate) struct BitReader<'a> {
    data: &'a [u8],
    /// Position in bits from the start of `data`.
    pos: usize,
}

impl<'a> BitReader<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        BitReader { data, pos: 0 }
    }

    /// `get_position()`: bits consumed so far.
    pub(crate) fn position(&self) -> usize {
        self.pos
    }

    pub(crate) fn bit(&mut self) -> Result<u32> {
        let byte = self.pos >> 3;
        if byte >= self.data.len() {
            return Err(Error::bitstream("header runs past the end of its OBU"));
        }
        let b = (self.data[byte] >> (7 - (self.pos & 7))) & 1;
        self.pos += 1;
        Ok(b as u32)
    }

    /// `f(n)` for n up to 32.
    pub(crate) fn f(&mut self, n: u32) -> Result<u32> {
        let mut x: u64 = 0;
        for _ in 0..n {
            x = (x << 1) | self.bit()? as u64;
        }
        Ok(x as u32)
    }

    pub(crate) fn flag(&mut self) -> Result<bool> {
        Ok(self.bit()? != 0)
    }

    /// `su(n)`.
    pub(crate) fn su(&mut self, n: u32) -> Result<i32> {
        let v = self.f(n)? as i64;
        let sign = 1i64 << (n - 1);
        Ok(if v & sign != 0 { v - 2 * sign } else { v } as i32)
    }

    /// `ns(n)`.
    pub(crate) fn ns(&mut self, n: u32) -> Result<u32> {
        if n <= 1 {
            return Ok(0);
        }
        let w = floor_log2(n) + 1;
        let m = (1u32 << w) - n;
        let v = self.f(w - 1)?;
        if v < m {
            return Ok(v);
        }
        let extra = self.f(1)?;
        Ok((v << 1) - m + extra)
    }

    /// `uvlc()`.
    pub(crate) fn uvlc(&mut self) -> Result<u32> {
        let mut lz = 0;
        loop {
            if self.bit()? != 0 {
                break;
            }
            lz += 1;
            if lz > 40 {
                return Err(Error::bitstream("uvlc too long"));
            }
        }
        if lz >= 32 {
            return Ok(u32::MAX);
        }
        let v = self.f(lz)? as u64;
        Ok((v + (1u64 << lz) - 1) as u32)
    }

    /// `leb128()`.
    pub(crate) fn leb128(&mut self) -> Result<u64> {
        let mut value: u64 = 0;
        for i in 0..8 {
            let b = self.f(8)? as u64;
            value |= (b & 0x7f) << (i * 7);
            if b & 0x80 == 0 {
                break;
            }
        }
        Ok(value)
    }

    /// `byte_alignment()`.
    pub(crate) fn byte_align(&mut self) -> Result<()> {
        while self.pos & 7 != 0 {
            self.bit()?;
        }
        Ok(())
    }
}

/// `FloorLog2(x)` for x >= 1.
pub(crate) fn floor_log2(x: u32) -> u32 {
    31 - x.leading_zeros()
}

/// `CeilLog2(x)`.
pub(crate) fn ceil_log2(x: u32) -> u32 {
    if x < 2 {
        return 0;
    }
    let mut i = 1;
    let mut p = 2u64;
    while p < x as u64 {
        i += 1;
        p <<= 1;
    }
    i
}

/// Writes bits most significant first: the inverse of [`BitReader`].
#[derive(Debug, Clone, Default)]
pub(crate) struct BitWriter {
    pub(crate) out: Vec<u8>,
    bits: u32,
    nbits: u32,
}

impl BitWriter {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn bit(&mut self, b: u32) {
        self.bits = (self.bits << 1) | (b & 1);
        self.nbits += 1;
        if self.nbits == 8 {
            self.out.push(self.bits as u8);
            self.bits = 0;
            self.nbits = 0;
        }
    }

    /// `f(n)`.
    pub(crate) fn f(&mut self, n: u32, v: u32) {
        for i in (0..n).rev() {
            self.bit((v >> i) & 1);
        }
    }

    pub(crate) fn flag(&mut self, b: bool) {
        self.bit(b as u32);
    }

    /// `trailing_bits()`: a one, then zeros to the byte boundary.
    pub(crate) fn trailing_bits(&mut self) {
        self.bit(1);
        while self.nbits != 0 {
            self.bit(0);
        }
    }

    /// `byte_alignment()`.
    pub(crate) fn byte_align(&mut self) {
        while self.nbits != 0 {
            self.bit(0);
        }
    }

    pub(crate) fn finish(mut self) -> Vec<u8> {
        self.byte_align();
        self.out
    }
}

/// Appends `v` as `leb128()`, in the fewest bytes.
pub(crate) fn write_leb128(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let mut w = BitWriter::new();
        w.f(3, 5);
        w.f(32, 0xdead_beef);
        w.trailing_bits();
        let d = w.finish();
        let mut r = BitReader::new(&d);
        assert_eq!(r.f(3).unwrap(), 5);
        assert_eq!(r.f(32).unwrap(), 0xdead_beef);
    }

    #[test]
    fn leb128() {
        for v in [0u64, 1, 127, 128, 300, 1 << 20, (1 << 32) - 1] {
            let mut out = Vec::new();
            write_leb128(&mut out, v);
            assert_eq!(BitReader::new(&out).leb128().unwrap(), v);
        }
    }

    #[test]
    fn logs() {
        assert_eq!(floor_log2(1), 0);
        assert_eq!(floor_log2(255), 7);
        assert_eq!(ceil_log2(0), 0);
        assert_eq!(ceil_log2(5), 3);
        assert_eq!(ceil_log2(8), 3);
    }
}

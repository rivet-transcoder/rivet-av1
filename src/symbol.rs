//! The symbol decoder (8.2): AV1's multi-symbol arithmetic decoder with
//! per-context CDF adaptation, and `read_literal`, `NS(n)` on top of it.

use crate::bits::floor_log2;
use crate::consts::{EC_MIN_PROB, EC_PROB_SHIFT};

/// Decodes the arithmetic-coded data of one tile.
pub(crate) struct SymbolDecoder<'a> {
    data: &'a [u8],
    /// Next bit to read, from the start of `data`.
    pos: usize,
    value: u32,
    range: u32,
    max_bits: i64,
    /// `disable_cdf_update`.
    disable_update: bool,
}

impl<'a> SymbolDecoder<'a> {
    /// `init_symbol( sz )` over the `sz = data.len()` bytes of a tile.
    pub(crate) fn new(data: &'a [u8], disable_update: bool) -> Self {
        let sz = data.len() as i64;
        let mut d = SymbolDecoder {
            data,
            pos: 0,
            value: 0,
            range: 1 << 15,
            max_bits: 8 * sz - 15,
            disable_update,
        };
        let num_bits = (sz * 8).min(15) as u32;
        let buf = d.read_bits(num_bits);
        let padded = buf << (15 - num_bits);
        d.value = ((1 << 15) - 1) ^ padded;
        d
    }

    fn read_bits(&mut self, n: u32) -> u32 {
        let mut x = 0u32;
        for _ in 0..n {
            let byte = self.pos >> 3;
            let b = if byte < self.data.len() {
                (self.data[byte] >> (7 - (self.pos & 7))) & 1
            } else {
                0
            };
            x = (x << 1) | b as u32;
            self.pos += 1;
        }
        x
    }

    /// `read_symbol( cdf )`: `cdf` holds N + 1 entries, the last being the
    /// adaptation counter.
    pub(crate) fn read_symbol(&mut self, cdf: &mut [u16]) -> usize {
        let n = cdf.len() - 1;
        let mut cur = self.range;
        let mut symbol: usize = 0;
        let mut prev;
        loop {
            prev = cur;
            let f = (1u32 << 15) - cdf[symbol] as u32;
            cur = ((self.range >> 8) * (f >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT);
            cur += EC_MIN_PROB * (n - symbol - 1) as u32;
            if self.value >= cur {
                break;
            }
            symbol += 1;
        }
        self.range = prev - cur;
        self.value -= cur;
        self.renormalize();
        if !self.disable_update {
            update_cdf(cdf, symbol);
        }
        symbol
    }

    fn renormalize(&mut self) {
        let bits = 15 - floor_log2(self.range);
        self.range <<= bits;
        let num_bits = (bits as i64).min(self.max_bits.max(0)) as u32;
        let new_data = self.read_bits(num_bits);
        let padded = new_data << (bits - num_bits);
        self.value = padded ^ (((self.value + 1) << bits) - 1);
        self.max_bits -= bits as i64;
    }

    /// `read_bool()`: an equiprobable bit.
    pub(crate) fn read_bool(&mut self) -> u32 {
        let cdf = [1u16 << 14, 1 << 15, 0];
        // The adaptation of this throwaway CDF has no effect.
        let n = 2;
        let mut cur = self.range;
        let mut symbol = 0;
        let mut prev;
        loop {
            prev = cur;
            let f = (1u32 << 15) - cdf[symbol] as u32;
            cur = ((self.range >> 8) * (f >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT);
            cur += EC_MIN_PROB * (n - symbol - 1) as u32;
            if self.value >= cur {
                break;
            }
            symbol += 1;
        }
        self.range = prev - cur;
        self.value -= cur;
        self.renormalize();
        symbol as u32
    }

    /// `read_literal( n )` / `L(n)`.
    pub(crate) fn read_literal(&mut self, n: u32) -> u32 {
        let mut x = 0;
        for _ in 0..n {
            x = 2 * x + self.read_bool();
        }
        x
    }

    /// `NS(n)`.
    pub(crate) fn read_ns(&mut self, n: u32) -> u32 {
        if n <= 1 {
            return 0;
        }
        let w = floor_log2(n) + 1;
        let m = (1 << w) - n;
        let v = self.read_literal(w - 1);
        if v < m {
            return v;
        }
        let extra = self.read_literal(1);
        (v << 1) - m + extra
    }

    /// Whether the tile's padding is as `exit_symbol()` requires: the
    /// trailing one bit where decoding ended, zeros after it. A cheap check
    /// that the whole tile was parsed with the right syntax.
    pub(crate) fn trailing_ok(&self) -> bool {
        let pos = self.pos as i64;
        let trailing = pos - (self.max_bits + 15).min(15);
        let end = pos + self.max_bits.max(0);
        if trailing < 0 || trailing as usize >= self.data.len() * 8 || end as usize > self.data.len() * 8 {
            return false;
        }
        let bit = |p: i64| (self.data[(p >> 3) as usize] >> (7 - (p & 7))) & 1;
        if bit(trailing) != 1 {
            return false;
        }
        ((trailing + 1)..end).all(|p| bit(p) == 0)
    }
}

/// The CDF adaptation of 8.2.6.
#[inline]
pub(crate) fn update_cdf(cdf: &mut [u16], symbol: usize) {
    let n = cdf.len() - 1;
    let cnt = cdf[n];
    let rate = 3 + (cnt > 15) as u32 + (cnt > 31) as u32 + floor_log2(n as u32).min(2);
    let mut tmp = 0u32;
    for (i, c) in cdf.iter_mut().enumerate().take(n - 1) {
        if i == symbol {
            tmp = 1 << 15;
        }
        let v = *c as u32;
        if tmp < v {
            *c -= ((v - tmp) >> rate) as u16;
        } else {
            *c += ((tmp - v) >> rate) as u16;
        }
    }
    if cnt < 32 {
        cdf[n] += 1;
    }
}

/// The inverse of [`SymbolDecoder`]: an arithmetic encoder whose output
/// the symbol decoding process (8.2.6) decodes to the symbols given.
///
/// Derivation from 8.2.6: writing `off = R - 1 - SymbolValue` for the
/// decoder's position in the current interval of size `R = SymbolRange`,
/// symbol `s` is decoded when `off` lies in `[R - prev, R - cur)`, where
/// `prev` and `cur` are the decoder's thresholds for `s - 1` and `s`; the
/// interval then becomes `prev - cur` wide, and renormalisation by `d` bits
/// appends `d` code bits to `off`. So the encoder keeps the interval's low
/// end, adds `R - prev`, narrows `R` to `prev - cur`, and shifts both by
/// `d`. At the end it picks the code value in the final interval whose
/// window starts with the one bit `exit_symbol()` expects, followed by
/// zeros.
pub(crate) struct SymbolEncoder {
    out: Vec<u8>,
    /// The low end of the interval: the bits not yet flushed to `out`.
    low: u64,
    /// Number of bits held in `low`.
    cnt: u32,
    range: u32,
    disable_update: bool,
}

impl SymbolEncoder {
    pub(crate) fn new(disable_update: bool) -> Self {
        SymbolEncoder {
            out: Vec::new(),
            low: 0,
            cnt: 15,
            range: 1 << 15,
            disable_update,
        }
    }

    fn carry(&mut self) {
        let mut i = self.out.len();
        while i > 0 {
            i -= 1;
            self.out[i] = self.out[i].wrapping_add(1);
            if self.out[i] != 0 {
                return;
            }
        }
    }

    /// Codes `symbol` with `cdf` (N + 1 entries) and adapts it.
    pub(crate) fn write_symbol(&mut self, cdf: &mut [u16], symbol: usize) {
        let n = cdf.len() - 1;
        let r = self.range;
        let thresh = |s: isize| -> u32 {
            if s < 0 {
                return r;
            }
            let s = s as usize;
            let f = (1u32 << 15) - cdf[s] as u32;
            (((r >> 8) * (f >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT)) + EC_MIN_PROB * (n - s - 1) as u32
        };
        let prev = thresh(symbol as isize - 1);
        let cur = thresh(symbol as isize);
        self.low += (r - prev) as u64;
        self.range = prev - cur;
        if self.low >> self.cnt != 0 {
            self.low &= (1u64 << self.cnt) - 1;
            self.carry();
        }
        let d = 15 - floor_log2(self.range);
        self.range <<= d;
        self.low <<= d;
        self.cnt += d;
        while self.cnt >= 24 {
            let byte = (self.low >> (self.cnt - 8)) as u8;
            self.out.push(byte);
            self.low &= (1u64 << (self.cnt - 8)) - 1;
            self.cnt -= 8;
        }
        if !self.disable_update {
            update_cdf(cdf, symbol);
        }
    }

    /// An equiprobable bit (`read_bool()`).
    pub(crate) fn write_bool(&mut self, bit: u32) {
        let mut cdf = [1u16 << 14, 1 << 15, 0];
        let save = self.disable_update;
        self.disable_update = true;
        self.write_symbol(&mut cdf, bit as usize);
        self.disable_update = save;
    }

    /// `L(n)`.
    pub(crate) fn write_literal(&mut self, n: u32, v: u32) {
        for i in (0..n).rev() {
            self.write_bool((v >> i) & 1);
        }
    }

    /// `NS(n)`.
    pub(crate) fn write_ns(&mut self, n: u32, v: u32) {
        if n <= 1 {
            return;
        }
        let w = floor_log2(n) + 1;
        let m = (1 << w) - n;
        if v < m {
            self.write_literal(w - 1, v);
        } else {
            let t = v + m;
            self.write_literal(w - 1, t >> 1);
            self.write_literal(1, t & 1);
        }
    }

    /// Ends the tile: the code value with the trailing one bit, padded to a
    /// whole byte.
    pub(crate) fn finish(mut self) -> Vec<u8> {
        let half = 1u64 << 14;
        let mut c = if self.low <= half {
            half
        } else {
            (((self.low - half) + (1 << 15) - 1) >> 15 << 15) + half
        };
        if c >> self.cnt != 0 {
            c &= (1u64 << self.cnt) - 1;
            self.carry();
        }
        // The bits of c from the top down to bit 14 (the trailing one).
        let nbits = self.cnt - 14;
        let mut v = c >> 14;
        let mut left = nbits;
        while left >= 8 {
            self.out.push((v >> (left - 8)) as u8);
            left -= 8;
        }
        if left > 0 {
            v &= (1 << left) - 1;
            self.out.push((v << (8 - left)) as u8);
        }
        self.out
    }
}

/// The tile's symbol coder: decoding, or encoding the values the encoder
/// planted.
pub(crate) enum Coder<'a> {
    Dec(SymbolDecoder<'a>),
    Enc(SymbolEncoder),
}

impl Coder<'_> {
    pub(crate) fn encoding(&self) -> bool {
        matches!(self, Coder::Enc(_))
    }

    /// `read_symbol( cdf )` when decoding; codes `v` when encoding.
    #[inline]
    pub(crate) fn symbol(&mut self, cdf: &mut [u16], v: usize) -> usize {
        match self {
            Coder::Dec(d) => d.read_symbol(cdf),
            Coder::Enc(e) => {
                e.write_symbol(cdf, v);
                v
            }
        }
    }

    /// A syntax element the encoder never codes.
    #[inline]
    pub(crate) fn read_symbol(&mut self, cdf: &mut [u16]) -> usize {
        match self {
            Coder::Dec(d) => d.read_symbol(cdf),
            Coder::Enc(_) => panic!("the encoder does not code this syntax element"),
        }
    }

    /// `L(n)` when decoding; codes `v` when encoding.
    #[inline]
    pub(crate) fn literal(&mut self, n: u32, v: u32) -> u32 {
        match self {
            Coder::Dec(d) => d.read_literal(n),
            Coder::Enc(e) => {
                e.write_literal(n, v);
                v
            }
        }
    }

    /// A literal the encoder never codes (zero-length ones are fine).
    #[inline]
    pub(crate) fn read_literal(&mut self, n: u32) -> u32 {
        match self {
            Coder::Dec(d) => d.read_literal(n),
            Coder::Enc(_) if n == 0 => 0,
            Coder::Enc(_) => panic!("the encoder does not code this literal"),
        }
    }

    /// `NS(n)` when decoding; codes `v` when encoding.
    pub(crate) fn ns(&mut self, n: u32, v: u32) -> u32 {
        match self {
            Coder::Dec(d) => d.read_ns(n),
            Coder::Enc(e) => {
                e.write_ns(n, v);
                v
            }
        }
    }

    /// A non-symmetric value the encoder never codes.
    pub(crate) fn read_ns(&mut self, n: u32) -> u32 {
        match self {
            Coder::Dec(d) => d.read_ns(n),
            Coder::Enc(_) => panic!("the encoder does not code this value"),
        }
    }

    pub(crate) fn trailing_ok(&self) -> bool {
        match self {
            Coder::Dec(d) => d.trailing_ok(),
            Coder::Enc(_) => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Symbols written with adapting CDFs decode back, padding and all.
    #[test]
    fn round_trip() {
        let mut seed = 12345u32;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        for trial in 0..50 {
            let n_syms = 1 + (rnd() % 2000) as usize;
            let mut syms = Vec::new();
            for _ in 0..n_syms {
                let kind = rnd() % 3;
                let v = match kind {
                    0 => rnd() % 4,
                    1 => (rnd() % 16).min(rnd() % 16),
                    _ => rnd() & 1,
                };
                syms.push((kind, v));
            }
            let base4: [u16; 5] = [8000, 16000, 24000, 32768, 0];
            let mut base16 = [0u16; 17];
            for (i, b) in base16.iter_mut().enumerate().take(16) {
                *b = ((i as u32 + 1) * 32768 / 16) as u16;
            }
            let (mut c4, mut c16) = (base4, base16);
            let mut e = SymbolEncoder::new(trial % 7 == 0);
            for &(k, v) in &syms {
                match k {
                    0 => e.write_symbol(&mut c4, v as usize),
                    1 => e.write_symbol(&mut c16, v as usize),
                    _ => e.write_bool(v),
                }
            }
            let data = e.finish();
            let (mut c4, mut c16) = (base4, base16);
            let mut d = SymbolDecoder::new(&data, trial % 7 == 0);
            for (i, &(k, v)) in syms.iter().enumerate() {
                let got = match k {
                    0 => d.read_symbol(&mut c4) as u32,
                    1 => d.read_symbol(&mut c16) as u32,
                    _ => d.read_bool(),
                };
                assert_eq!(got, v, "trial {trial} symbol {i}");
            }
            assert!(d.trailing_ok(), "trial {trial}: padding");
        }
    }
}

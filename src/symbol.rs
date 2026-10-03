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
        if trailing < 0 || end as usize > self.data.len() * 8 {
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

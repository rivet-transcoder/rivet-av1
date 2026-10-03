# rivet-av1

[![CI](https://github.com/rivet-transcoder/rivet-av1/actions/workflows/ci.yml/badge.svg)](https://github.com/rivet-transcoder/rivet-av1/actions/workflows/ci.yml)

An **AV1** decoder and encoder in Rust: no C, no system libraries, no build
script, nothing to install on a build host. Written from the *AV1
Bitstream & Decoding Process Specification* (AOMedia), not translated from
any other implementation. The decoder is bit-exact on **all 244** of
AOMedia's public AV1 test vectors and on all 3 015 of the Argon
conformance streams (the numbers are
[below](#how-it-is-checked)); the encoder writes key and inter frames that
decode to exactly what it reconstructed.

Written for the **[rivet](https://github.com/rivet-transcoder/rivet)**
transcoder, whose default output codec is AV1 (today through the
third-party rav1d and rav1e crates). Usable on its own by anything that has
AV1 temporal units (from IVF, WebM / Matroska, MP4, or an Annex B stream)
and wants planar pictures back, or planar pictures and wants AV1.

This is the first milestone of a longer effort: the decoder implements the
whole specification but is single-threaded and scalar; the encoder is
deliberately simple. What is and is not there is listed precisely below.

Published as `rivet-av1`; **imported as `av1`** (`use av1::…`). One
dependency (`thiserror`), no features, no build script.

```toml
[dependencies]
av1 = { package = "rivet-av1", git = "https://github.com/rivet-transcoder/rivet-av1", branch = "develop" }
```

## What it decodes

Everything in the specification's general decoding process:

| | |
|---|---|
| **Profiles** | 0, 1 and 2 (Main, High, Professional): 8-, 10- and 12-bit; monochrome, 4:2:0, 4:2:2 and 4:4:4; every colour description |
| **Bitstream** | OBUs in the low-overhead format (section 5) and the length-delimited format of Annex B; sequence headers, temporal delimiters, frame headers and redundant copies, tile groups, frame OBUs; HDR metadata OBUs (content light level, mastering display) reported on every frame, other metadata and padding skipped; operating point selection and layer dropping (scalable streams); frame ids |
| **Frames** | key, inter, intra-only and switch frames; hidden frames and `show_existing_frame` (including key frames shown that way); error-resilient mode; frame size changes, `frame_size_with_refs`, render sizes; reference frame scaling (2:1 down to 1:16) |
| **Syntax** | uniform and explicit tiles (any count), the symbol decoder with CDF adaptation, saving and loading CDFs through the reference slots, `disable_cdf_update` / `disable_frame_end_update_cdf`; segmentation (map, temporal prediction, every feature); delta quantiser and delta loop filter (single and multi); quantiser matrices; lossless |
| **Partitions** | every partition (none, horizontal, vertical, split, the A/B three-way splits, 4-way) from 128x128 superblocks down to 4x4; transform sizes 4x4 to 64x64 including all rectangular sizes, `TX_MODE_SELECT` and inter transform trees |
| **Intra** | all thirteen modes, angle deltas, the intra edge filter and edge upsampling, smooth modes, Paeth, filter intra (recursive), chroma from luma, palette (with the colour cache), intra block copy |
| **Inter** | single and compound prediction from seven references; motion vector prediction with the temporal motion field (projection); NEAREST/NEAR/GLOBAL/NEW and every compound mode, dynamic reference list; regular, smooth, sharp and bilinear filters with dual filters; global motion (translation, rotation-zoom, affine) and local warped motion; OBMC; inter-intra (smooth and wedge); wedge, difference-weighted and distance-weighted compound; skip mode |
| **Residual** | coefficient coding at every size; all sixteen transform types (DCT, ADST, flipped ADST, identity in every combination), the 64-point DCT, the Walsh-Hadamard transform; dequantisation at all bit depths |
| **Filters** | the deblocking loop filter (4-, 6-, 8- and 16-wide), CDEF, super-resolution upscaling, loop restoration (Wiener and self-guided) |
| **Output** | film grain synthesis (the specification's reference process, bit-exact) |

Output is a [`Frame`](src/frame.rs): the visible picture (`UpscaledWidth`
by `FrameHeight`), planes Y, U, V packed one after another, one byte per
sample at 8 bits and little-endian `u16` above — the shape of rivet-vp9's
`Frame`. `Decoder::decode(&temporal_unit)` takes one temporal unit (an IVF
frame or container block) and returns the frame it shows; with spatial
layers that is the last (highest) one, as the specification's output
process recommends, and `Decoder::decode_all` returns every one.
`Decoder::decode_annexb` takes an Annex B temporal unit
(`annexb_temporal_units` splits a stream into them).

Not there yet:

- **Large-scale tile decoding** (tile list OBUs, 7.3). Optional for
  conformance; reported as `Error::Unsupported`.
- **Speed.** Decoding is single-threaded scalar Rust that follows the
  specification's processes closely: about 6 megapixels a second on one
  core (some 7 frames/s at 1280x720) on the machine it was written on. No
  tile or frame threading, no SIMD, no frame-buffer pool.
- **Error recovery.** A corrupt frame returns `Error::Bitstream`; there is
  no concealment, so decoding should resume at the next key frame.
  Malformed input has not made it panic under the property tests below,
  and frames larger than 2^26 pixels (adjustable, `set_max_pixels`) are
  refused before allocation.
- Conformance requirements that do not change the output (frame id
  consistency, level limits) are not checked; `set_strict(true)` checks
  each tile's trailing padding.

## What it encodes

Profile 0 (8- or 10-bit 4:2:0), one tile, one temporal unit per frame, any
size from 1x1 up to 4096 wide:

- **Key frames**: 64x64 superblocks split down to 8x8 by a variance test
  against the quantiser (partition search lite; 4x4 where the frame edge
  forces it); each block takes the best of ten intra modes (DC, V, H,
  smooth, smooth-V, smooth-H, Paeth, D45, D135, D67) by SATD on the actual
  prediction, chroma chosen separately; the intra edge filter on.
- **Transforms**: the largest that fits each block; for intra luma blocks up
  to 16x16 the best of DCT, ADST and the two mixed DCT/ADST types by
  distortion plus estimated rate on the quantised reconstruction; chroma
  as the mode implies. A dead-zone quantiser.
- **Inter frames**: single-reference prediction from the previous frame:
  a full-sample square search then half- and quarter-sample refinement
  with the normative 8-tap predictor; NEWMV (coded against the
  specification's motion vector prediction), NEARESTMV or GLOBALMV, with
  intra as the alternative per block; skip when the residual quantises to
  nothing. CDFs carried from frame to frame through the reference slot.
- **Quantiser**: fixed (`Config::quantizer`, the `base_q_idx` 1–255), or
  **rate control** to a target number of bits per frame
  (`Config::target_bits_per_frame`): the quantiser moves with the log of
  the ratio of the bits a frame took to its target.
- **Loop filter** at a level derived from the quantiser (or set).
- **Colour and HDR signalling**: `Config::color` is written into the
  sequence header's `color_config()` (primaries, transfer, matrix, range,
  chroma sample position; `color_description_present_flag` when any code
  point is specified), and `Config::hdr` as `METADATA_TYPE_HDR_CLL` and
  `METADATA_TYPE_HDR_MDCV` metadata OBUs after the sequence header of every
  key frame — 10-bit PQ (HDR10) and HLG. The decoder reports both on every
  `Frame` (`color`, `hdr`) and through `Decoder::color_info` /
  `Decoder::hdr_metadata`.
- **Key frames on demand**: `Encoder::force_keyframe()` makes the next frame
  a key frame (with its sequence header, so a decoder can start there)
  without resetting the encoder; the interval restarts from it.
- The arithmetic encoder is the exact inverse of the decoder's symbol
  decoder (derived from 8.2.6).

How it works: the encoder runs the decoder's own tile walker in encode
mode. Before each syntax element the walker asks the encoder for its
decision (partition, modes, motion vector, the quantised levels of a
transform block — taken with the prediction already in place), then codes
it with exactly the contexts and CDF adaptation it decodes with, and
reconstructs exactly as it decodes. The frame header is written, then
parsed back by the decoder's header parser, and the in-loop filters and
reference update are the decoder's. So the encoder's reconstruction is,
by construction, what a decoder outputs — and `tests/encode.rs` checks
that every temporal unit decoded by a fresh decoder equals it, sample for
sample.

Not there yet (in rough order of value):

- **Rate-distortion search** — modes are chosen by SATD, partitions by a
  variance test; no trial coding of partitions, no `TX_MODE_SELECT`, no
  rectangular partitions.
- **More of the toolbox** — CDEF and loop restoration (the decoder's are
  ready; the encoder writes neither), palette, intra block copy, filter
  intra, chroma from luma, compound and multiple references, golden /
  alt-ref structures, OBMC and warped motion, segmentation and adaptive
  quantisation, film grain parameters.
- **Profiles 1 and 2**, 12-bit, monochrome; lossless (quantiser 0 needs the
  forward Walsh-Hadamard transform).
- Speed: about 10 frames/s at 352x288, single-threaded.

Measured on 8 frames of natural video at 352x288 (the decoded frames of
`av1-1-b8-05-mv.ivf`, played back and forth; `tests/encode.rs`,
`quality_table`), a key frame then seven inter frames:

| quantiser | bytes (8 frames) | key frame | per inter frame | PSNR Y | PSNR U | PSNR V |
|---|---|---|---|---|---|---|
| 20 | 230 307 | 43 793 | 26 644 | 47.75 dB | 49.82 dB | 50.50 dB |
| 50 | 141 411 | 29 427 | 15 997 | 42.31 dB | 46.46 dB | 46.79 dB |
| 90 | 93 210 | 20 812 | 10 342 | 38.51 dB | 44.06 dB | 44.09 dB |
| 130 | 53 295 | 12 820 | 5 782 | 34.17 dB | 41.40 dB | 41.12 dB |
| 170 | 25 707 | 6 424 | 2 754 | 29.69 dB | 38.86 dB | 38.09 dB |
| 210 | 10 151 | 2 380 | 1 110 | 25.58 dB | 36.63 dB | 35.74 dB |
| 250 | 3 821 | 744 | 439 | 22.53 dB | 34.63 dB | 34.08 dB |

At quantiser 90 the same eight frames take 161 503 bytes coded all-intra.

## How it is checked

- **AOMedia's AV1 test vectors**: bitstreams published with the MD5 of
  every frame the reference decoder shows. `tools/fetch-vectors.sh`
  downloads the 244 `av1-1-b8-*` and `av1-1-b10-*` streams (about 7 MB)
  and `tests/vectors.rs` decodes each and compares every shown frame.
  **244 of 244 pass**, every frame bit-exact:

  | group | what it exercises | pass |
  |---|---|---|
  | av1-1-b8-00, av1-1-b10-00 | quantiser 0–63, 8- and 10-bit (lossless included) | 128 / 128 |
  | av1-1-b8-01 | frame sizes 16 to 66 and 196 to 226, every combination | 100 / 100 |
  | av1-1-b8-02 | all-intra coding | 1 / 1 |
  | av1-1-b8-03 | frame size up and down, scaled references (Matroska) | 2 / 2 |
  | av1-1-b8-04, 05, 06 | CDF update, motion vectors, motion field projection | 3 / 3 |
  | av1-1-b8-16 | intra-only frames, intra block copy | 1 / 1 |
  | av1-1-b8-22 | temporal and spatial scalability (L1T2, L2T1, L2T2) | 5 / 5 |
  | av1-1-b8-23, av1-1-b10-23 | film grain | 2 / 2 |
  | av1-1-b8-24, av1-1-b10-24 | monochrome | 2 / 2 |

  Twelve small vectors are committed in [`tests/data`](tests/data/README.md)
  so `cargo test` checks real streams without the download.
- **The Argon conformance streams** (Argon Streams AV1 v2.1, AOMedia's
  decoder verification suite: streams designed for coverage of every
  syntax element and decoding process, in Annex B and section 5 formats,
  with reference MD5s of all output). `tests/argon.rs` runs them from a
  local copy (`ARGON_DIR`; 7 GB, not downloaded by the tests), taking each
  stream's options (format, operating point, all layers) from its
  reference command. **All 3 015
  base streams pass**, every output byte matching:

  | group | what it exercises | pass |
  |---|---|---|
  | profile0_core, profile1_core, profile2_core | the coverage streams, Annex B, per profile (0: 8/10-bit 4:2:0 and monochrome; 1: 4:4:4; 2: 4:2:2, 12-bit) | 764 / 764, 731 / 731, 894 / 894 |
  | profileN_core_special, profileN_not_annexb_special | the suite's "special" streams, Annex B and section 5 | 330 / 330 |
  | profileN_not_annexb | section 5 low-overhead streams | 37 / 37 |
  | profileN_stress | the suite's stress streams | 252 / 252 |
  | profile_switching | the suite's profile-switching streams | 7 / 7 |

  The suite also carries, per stream, reference output for each operating
  point (`layers/N`, about 75 000 more decodes of the same streams with
  layers dropped): the operating point 1 set was run, **2 763 / 2 763 pass**
  (`ARGON_LAYERS=1`); the others were not run for time. The large-scale-tile directories (tile
  list OBUs, unsupported) and the error-resilience directories (no
  reference output) are not run.
- **The encoder** (`tests/encode.rs`): every temporal unit, decoded by a
  fresh decoder with the padding check on, equals the encoder's
  reconstruction exactly — key frames at several quantisers, inter frames,
  10-bit, sizes from 1x1; PSNR falls and size falls as the quantiser rises;
  inter frames cost less than the key frame; rate control lands near its
  target.
- **Malformed input** (`tests/fuzz.rs`, proptest): arbitrary bytes, and the
  committed vectors with bits flipped, bytes cut and garbage spliced in,
  decoded in debug builds (overflow checks on) — errors, never a panic.
- **Units**: the arithmetic encoder against the decoder over random symbol
  sequences with adapting CDFs (and the decoder's padding check); the
  forward transforms round-tripping through the normative inverse; the
  inverse DCT on DC-only blocks; the Walsh-Hadamard transform; bit and
  IVF round trips.

## Provenance and licensing

Written from the specification's text; **no AV1 implementation's source was
read** — not libaom, not dav1d or rav1d, not rav1e, not SVT-AV1, not
FFmpeg — and none was run: the tests use only the specification-derived
checks above, round trips through this crate, and published conformance
bitstreams with their MD5s (data: the AOM test vectors from AOMedia's
public test-data bucket, the committed ones listed with their source in
[`tests/data/README.md`](tests/data/README.md), and the Argon streams). The
specification's tables (default CDFs, scans, quantiser lookups and
matrices, filter taps) are data and were transcribed by
[`tools/gen_tables.py`](tools/gen_tables.py) from the specification's own
source text into [`src/tables.rs`](src/tables.rs). See [NOTICE](NOTICE).

Licensed under the Open Encoding Attribution License 1.0
([LICENSE.md](LICENSE.md)): source-available, not open source.

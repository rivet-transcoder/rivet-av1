# Committed test vectors

A few of AOMedia's public AV1 test vectors, small enough to keep in the
repository so `cargo test` checks real streams without a download. Each
`name.md5` holds the MD5 of every frame the stream shows, as published next
to the stream. The full set (244 streams, about 7 MB) is fetched by
`tools/fetch-vectors.sh` into `tests/vectors/`.

Source: `https://storage.googleapis.com/aom-test-data/<name>` and
`<name>.md5`, downloaded 2026-10-02. They are bitstream data and expected
checksums, published by AOMedia for decoder testing; no implementation's
code.

| stream | what it covers |
|---|---|
| av1-1-b8-01-size-66x66.ivf, av1-1-b8-01-size-226x226.ivf | frame sizes that are not multiples of 8 / 64 |
| av1-1-b8-00-quantizer-20.ivf, av1-1-b10-00-quantizer-10.ivf | 8- and 10-bit coding at a fixed quantiser |
| av1-1-b8-03-sizeup.mkv | frame size changes, scaled references (Matroska container) |
| av1-1-b8-04-cdfupdate.ivf | CDF update modes |
| av1-1-b8-05-mv.ivf, av1-1-b8-06-mfmv.ivf | motion vectors, motion field projection |
| av1-1-b8-22-svc-L1T2.ivf | temporal scalability |
| av1-1-b8-23-film_grain-50.ivf, av1-1-b10-23-film_grain-50.ivf | film grain synthesis |
| av1-1-b8-24-monochrome.ivf | monochrome (hashed with neutral chroma, as the .md5 files were made) |

av1-1-b8-05-mv.ivf's decoded frames are also the natural-content source of
the encoder tests.

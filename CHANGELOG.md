# Changelog

All notable changes to E8ZIP will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2026-07-14

### Added — more borrowed from the Glass Network (glass_windows branch)

- **Real-weight validation** (`benchmarks/benchmark_real_weights.py`):
  the TENSOR regime tested against actual `sentence-transformers/
  all-MiniLM-L6-v2` weights (not synthetic Gaussians), matching the
  Glass Network's own methodology. Result: **23.85 dB @ 4.34 bits/
  weight**, beating both the Glass Network's own published Leech-VQ
  reference (19.85 dB @ 4.81 bpw) and entropy-coded scalar INT5
  (20.38 dB @ 3.74 bpw) at a comparable rate. This corrects and
  properly earns the claim v2.0.0's synthetic-Gaussian benchmark made
  prematurely.
- **`core/transforms.py`**: incoherence transforms factored out and
  extended with a dense random-orthogonal rotation (ported from
  `model_cookbook/turbo_leech.py`) alongside randomized Hadamard, both
  exposed as a `transform=` option on `compress_tensor`.
- **`core/klc.py`**: gain/shape progressive lattice codec, ported and
  extended from the Glass Network's KLC2 (`packages/kdot_v2/
  lattice_codec.py`). Separates per-block magnitude from direction,
  uses the exact Leech CVP for the base layer (KLC2 used a cheaper
  sign-only coset), and entropy-codes every stream including the
  progressive residual layers (KLC2 stored raw bytes). Exposed as
  `KGCCompressor.compress_tensor_progressive` /
  `decompress_tensor_progressive(blob, up_to_layer=...)`.
  **Honest result**: hypothesized to also win on rate-distortion:
  it does not. Measured Pareto-dominated by the flat TENSOR regime at
  every tested rate, even after hyperparameter tuning. Its real,
  verified value is truncatable/progressive decode (byte-exact
  truncation points, strictly monotonic per-layer fidelity), a
  capability the flat regime doesn't have at all - documented as such
  rather than oversold.
- **Golay-protected KGC2 header**: the archive control block (version/
  regime/truth/flags) is now itself a Golay codeword - heals up to 3
  flipped bits transparently, detects 4+ instead of silently
  misparsing. 30/30 fault-injection trials healed in testing.
- 18 new tests (`tests/test_klc.py`): transform exactness/orthogonality,
  KLC round-trip across shapes, progressive-fidelity monotonicity,
  gain recovery, zero-block handling, header self-healing. 64 total
  now pass.

### Fixed

- `core/leech_lattice.py`: `shells` was cast to `int32`, which could
  silently overflow (undefined cast) for large-magnitude input via the
  legacy v1 LEECH-mode pathway; widened to `int64` with NaN/Inf
  sanitization at the `LeechLattice` OO-wrapper boundary.

## [2.0.0] - 2026-07-12

### Added — KGC: the Kaleidoscope Geometric Codec

- **Real entropy coding** (`core/entropy.py`): carry-less range coder with
  adaptive Fenwick-tree models (order-1, order-2, and experimental
  E8-trajectory context). Replaces zlib as the bit-emitting engine.
- **Exact Golay code** (`core/golay.py`): [24,12,8] extended binary Golay
  with Pless arithmetic syndrome decoding (heals up to 3 bit errors),
  ported from the Glass Network KMind (glass_windows branch).
- **Exact Leech quantizer** (`core/leech_lattice.py`, rewritten): the
  Conway–Sloane Construction-A CVP decoder, vectorized, with the new
  bijective index decomposition `y = 2g + i·1 + 4z` → `(g_idx, case, z)`
  that makes lattice points entropy-codeable. Verified optimal against
  exhaustive coset search.
- **Unified codec** (`core/kgc.py`): one `KGC2` container implementing
  the Law of Compression Debt — every archive is (address, program,
  residual) with a sha256 source address and a truth_status rung
  (exact_recovery / reconstruction / interpretation / confabulation).
  - Regime BYTES: lossless, verified round-trip, raw fallback (never
    expands random data).
  - Regime TENSOR: QuIP#-family weight compression — randomized Hadamard
    incoherence → Leech VQ → entropy-coded indices. Pareto-dominates
    scalar INT4/INT5 at equal rate in benchmarks.
  - Regime CONSOLIDATE: RG collapse of row sets to canonical Leech sites
    with bitwise-exact residual mode, conserved foreground mass (0.000%
    drift), and member addresses for auditability.
- Flat-layout self-alias in `__init__.py` so the repo imports as `e8zip`
  from any checkout directory; pytest now runs from a plain clone.
- Honest benchmark suite `benchmarks/benchmark_kgc.py` (every number is
  a verified round-trip; losing baselines are printed).
- 26 new falsification-style tests (`tests/test_kgc.py`).

### Fixed

- Leech quantization is now exact and ~1000x faster than the v1
  sampled-minimal-vector approximation.
- Corrected the upstream KMind `turbo_leech_packer` arity mismatch by
  having the decoder return the full `(point, g_idx, case, z)` index.

### Honest notes

- v1 "lossless" modes (FAST/NORMAL/LEECH/QUIP) stored a zlib copy of the
  original and ignored the geometric data on decode. They remain for
  `.e8z` v1 compatibility but are superseded by KGC.
- On match-heavy byte data, mature LZ codecs still win; the KGC byte
  regime's contribution is the verified debt container and the geometric
  context model, not LZ replacement.

## [Unreleased]

### Planned

- LLM-predictor entropy coding (arithmetic coding at model cross-entropy)
- GPU path for Leech CVP (BVH / RT-core nearest-neighbor search)
- Leech theta-series shell priors for the index models
- Multi-threaded compression
- Archive encryption support

## [1.0.0] - 2025-12-08

### Added

- Initial release of E8ZIP
- E8 lattice vector quantization
- Leech lattice (Lambda_24) experimental compressor
- Hyperbolic geodesic trajectory compression
- Golden ratio weighted geometric structures
- Hadamard transform preprocessing
- Black hole style holographic encoding
- Command-line interface (CLI)
- Python API for compression/decompression
- E8Z archive format
- Support for file and directory compression
- Multiple compression modes (fast, standard, ultra, mythic)
- Cross-platform support (Windows, Linux, macOS)
- Comprehensive test suite
- Benchmark tools for performance testing

### Features

- `e8zip` CLI tool
- `e8zip-gui` interactive interface
- Python package installable via pip
- Rich terminal output with progress bars
- Archive information and listing commands
- Streaming compression support
- GPU backend (optional, with PyTorch)

### Documentation

- README with installation and usage instructions
- API documentation in code
- Contributing guidelines
- MIT License

## [0.1.0] - 2025-11-15

### Added

- Initial proof of concept
- Basic E8 lattice implementation
- Simple compression/decompression

---

**Note**: Dates follow YYYY-MM-DD format. Version numbers follow semantic versioning.

[Unreleased]: https://github.com/skyemalone/e8zip/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/skyemalone/e8zip/releases/tag/v1.0.0
[0.1.0]: https://github.com/skyemalone/e8zip/releases/tag/v0.1.0

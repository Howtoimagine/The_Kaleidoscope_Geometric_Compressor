# Changelog

All notable changes to E8ZIP will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2026-07-12

### Added — theta prior, LLM predictor socket, recall-as-rays

- **Exact Leech theta series** (`core/leech_lattice.py`): `theta_series(n)`
  computes N(2n) for arbitrary n via Θ = E₁₂ − (65520/691)Δ with the Euler
  product expanded by the pentagonal number theorem — exact Python
  integers, verified against the tabulated coefficients. `shell_prior()`
  turns it into the max-entropy shell distribution
  P(shell n) ∝ N(2n)·exp(−n/σ²) for a Gaussian source.
- **Prior-seeded entropy models** (`core/entropy.py`): `AdaptiveModel`
  accepts an initial `prior`, with a raised rescale ceiling so good
  priors keep resolution while data can still override them.
- **Tensor regime priors** (`core/kgc.py`, header flags, backward
  compatible with 2.0 archives):
  - `FLAG_Z_PRIOR` (new default): z-translation model seeded with the
    discretized Gaussian implied by `scale`. Rate improvement measured
    at every operating point (4.20 → 4.13 bits/weight @ 23.9 dB).
  - `FLAG_SHELL_PRIOR` (opt-in): shell-indexed coding against the theta
    prior with shell-conditioned z models. Measured a NET rate loss
    (~+0.24 bpw at scale 4 — the shell is deterministic given (g,i,z),
    and conditioning recovers only ~2 of its ~8 bits/block); kept,
    documented as such, for progressive decode / shell auditing.
- **Predictor front-end** (`core/predictor.py`): the LLM socket for the
  BYTES regime. Any deterministic causal model emitting next-byte
  probabilities drives the range coder (`mode="predictor:<name>"`);
  archives store only the registry name. Ships `NGramMixPredictor`
  (dependency-free reference) and `CallablePredictor` (adapter for
  torch / llama.cpp / ONNX logits functions).
- **Recall-as-rays prototype** (`core/recall.py`, `docs/RT_RECALL.md`,
  `benchmarks/benchmark_recall.py`): feasibility study + three-way
  bake-off for GPU memory recall on NVIDIA RT cores — GEMM brute force,
  Leech-cell hashing (exact CVP as LSH), and the RT-style
  multi-projection (24→3) radius-filter + vote + exact re-rank pipeline
  (CPU reference semantics matching an OptiX BVH/any-hit port).
  Measured: projection filter reaches recall@10 = 1.00 at r=1.5, votes
  2/4 on clustered banks; lattice-cell hash trails at 0.58.
- **GPU recall backends** (`core/recall_gpu.py`, torch/CUDA): exact GEMM
  brute force and the full RT-style pipeline with every stage on device
  (grid probe via sorted-cell-key `searchsorted` — the one stage an
  OptiX port swaps for RT-core BVH traversal). Decision-gate numbers on
  RTX 3070 Ti at N=10⁶ rows: projection filter 0.119 ms/query at
  recall@10 = 0.993 vs exact GPU GEMM 0.285 ms/query — the filter beats
  exact search 2.4x on CUDA cores alone, passing the >2x adoption gate
  before RT cores are even used.
- **Online GRU predictor** (`core/predictor.py`, `"gru-online"`):
  NNCP-style byte-level GRU trained online during BOTH compression and
  decompression from a fixed seed — no weights in the archive, the
  learned model is program, not payload. On 8 KB of source code:
  gru-online 2.68x, ngram-mix 2.87x, order-2 baseline 2.28x (all
  verified round-trips).
- **OptiX RT-core recall backend** (`core/rt_optix.py`): the ray-tracing
  hardware port, built and running. NVRTC-compiled device programs
  (degenerate rays from each query, a custom origin-in-sphere
  `__intersection__`, an atomic-append `__anyhit__` that keeps
  traversing), a GAS of one AABB per stored row per projection, votes +
  exact 24-D re-rank in CuPy. Bounded-memory query chunking on both GPU
  backends. Measured on RTX 3070 Ti (full numbers in docs/RT_RECALL.md):
  at a saturating N=100k/4096-query batch the RT-core path is the fastest
  method (0.009 ms/query, 25% under exact GEMM) and the only one reaching
  recall@10 = 1.000; at N=2M the tuned CUDA grid probe is faster on this
  Ampere card while RT cores still give strictly the best recall. Live
  install verified: pyoptix 9.1, CuPy 14.1, CUDA 12.9/13.0, OptiX
  headers, VS2022 Build Tools + Win11 SDK.
- 29 new falsification-style tests (`tests/test_upgrades.py`), including
  live OptiX RT-core tests that auto-skip when the stack is absent.

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

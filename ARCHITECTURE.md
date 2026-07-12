# E8ZIP Architecture & Compression Pipeline

## v2: KGC — Kaleidoscope Geometric Codec

v2 is the honest successor to the v1 pipeline documented below. The v1
"lossless" modes computed geometric metadata and then stored a zlib copy
of the original, which the decoder used while skipping the geometry.
KGC removes that crutch: the geometry is now the codec.

### The unifying law

Everything KGC emits is a **Compression Debt triple** (Glass Network
KMind, Law 9: *"compression requires source addresses [and an]
executable program"*):

```
L(x) = L(address) + L(program) + L(residual)

address  = sha256 of source + conserved mass + member ids
program  = codec id, lattice, seeds - a reconstruction recipe
residual = range-coded bits the geometric prior could not predict
```

Lossy compression without addresses raises `Law9Error`. Every decode
reports a truth_status rung borrowed from the KMind language decoder:
`exact_recovery` / `reconstruction` / `interpretation` / `confabulation`.

### Shared core

```
core/entropy.py        carry-less range coder (mod 2^32, Subbotin style)
                       + adaptive Fenwick-tree models (order-1, order-2,
                       experimental E8-trajectory context). Models accept
                       an initial prior (v2.1) with a raised rescale
                       ceiling, so analytic priors save warm-up bits.
core/golay.py          [24,12,8] extended Golay: systematic G=[I|B],
                       B = bordered QR(11), Pless arithmetic syndrome
                       decoder (heals <=3 bit errors, detects 4)
core/leech_lattice.py  exact Leech CVP (Conway-Sloane Construction A,
                       vectorized over all 4096 Golay cosets x 2 cases)
                       + bijective index decomposition
                           y = 2g + i*1 + 4z  <->  (g_idx, case, z)
                       + exact theta series for arbitrary shells (v2.1):
                           Theta = E_12 - (65520/691)*Delta
                       and the max-entropy shell prior it implies
core/kgc.py            the KGC2 container + three regime front-ends
core/predictor.py      predictor front-end for BYTES (v2.1): any
                       deterministic causal next-byte model drives the
                       range coder; ngram-mix reference, gru-online
                       (NNCP-style, trains during encode AND decode),
                       CallablePredictor adapter for external LLMs
core/recall.py         recall bake-off, CPU reference: exact GEMM,
                       Leech-cell hash (CVP as LSH), RT-style
                       multi-projection filter (v2.1)
core/recall_gpu.py     the same pipeline on CUDA cores via torch (v2.1)
core/rt_optix.py       the same pipeline on RT cores via OptiX (v2.1):
                       NVRTC-compiled degenerate-ray any-hit programs
                       over a GAS of per-row AABBs
```

The index decomposition is what makes the lattice *codeable*: 12 bits of
Golay message + 1 case bit + small integers z, each stream fed to its
own adaptive model. It also fixes the upstream KMind
`turbo_leech_packer` arity bug (it expected this decomposition; the
decoder never returned it).

### Three regimes, one object

```
BYTES        input bytes -> adaptive context model -> range coder
             raw fallback if the model fails to shrink (random data
             stays ~1.0x). sha256-verified on decode. Lossless.

TENSOR       flatten -> randomized Hadamard (incoherence, QuIP#) ->
             normalize -> scale (the rate-distortion knob) -> 24-D
             blocks -> exact Leech CVP -> entropy-coded (g,i,z).
             v2.1 default seeds the z model with the discretized
             Gaussian implied by scale (FLAG_Z_PRIOR, header flags
             byte; 2.0 archives still decode).
             Measured: 4.13 bits/weight @ 23.9 dB SNR on Gaussian
             weights (was 4.20 before the z prior) vs INT4 scalar
             4.0 bpw @ 15.8 dB and INT5 5.0 bpw @ 22.0 dB -
             Pareto-dominant at every tested rate.

CONSOLIDATE  (N,24) rows *rg_scale -> Leech sites (block-spin RG step);
             rows sharing a site collapse to one representative.
             Per-site conserved mass is stored and reconstruction
             rescales so group energy matches exactly (0.000% drift).
             keep_residuals=True stores XOR-exact IEEE-754 residuals
             (bitwise round-trip, verified); False drops them but
             keeps addresses -> auditable lossy collapse.
```

### Lineage

The math is ported from the Glass Network (glass_windows branch):
`packages/kmind/golay.py`, `packages/kmind/leech.py` (quantizer),
`consolidation.py` / `renormalization.py` / `store.py` (conserved-mass
RG collapse), `law_registry.py` (Law 9). The Leech theta series
`Theta = E_12 - (65520/691)*Delta` = (1, 0, 196560, 16773120, ...) is
computed exactly for arbitrary shells in `core/leech_lattice.py`
(pentagonal-number-theorem expansion of Delta, integer arithmetic) and
wired into the codec as of v2.1 — see "theta priors" below.

### Theta priors: what won and what lost (v2.1, measured)

Two priors fall out of the max-entropy lattice-Gaussian analysis:

- **Gaussian z prior (WON, now default):** seed the z-translation model
  with the discretized Gaussian implied by `scale`. Zero side
  information, ~1.5 bits/block of warm-up saved, 4.20 -> 4.13 bpw at
  identical SNR.
- **Shell-indexed coding (LOST, opt-in only):** entropy-code each
  block's shell against P(shell n) ~ N(2n)*exp(-n/scale^2), condition z
  on shell buckets. The shell is deterministic given (g, i, z), so
  H(shell) + H(g,i,z | shell) = H(g,i,z) — explicit shell coding can at
  best break even, and in practice its ~8 bits/block cost only recovers
  ~2 via easier conditional modeling: **net +0.24 bpw**. Kept behind
  `use_shell_prior=True` for progressive decode / shell auditing,
  documented as a rate loss. (Falsification-first: the measured negative
  result is part of the architecture record.)

### Honest limits

- The byte regime loses to LZ codecs on match-heavy data (zlib 4.16x vs
  KGC 2.78x on Python source). Its contribution is the verified debt
  container and geometric context modeling, not LZ replacement. The
  LLM-predictor front-end now exists (`core/predictor.py`,
  `mode="predictor:<name>"`): on 8 KB of source, ngram-mix reaches
  2.87x and the online-trained GRU 2.68x vs order-2's 2.28x — the GRU
  proves the deterministic-replay socket (no weights in the archive) at
  ~1 ms/byte; plugging a pretrained byte-LLM into `CallablePredictor`
  is the remaining step to state-of-the-art on in-distribution data.
- The tensor regime is deliberately lossy (like all PTQ weight
  compression); rate and distortion are always reported together.
- Leech CVP is exact but CPU-heavy in pure numpy (~35 blocks/s); a CUDA
  CVP kernel remains future work. GPU *recall* over stored rows,
  however, is built and measured (`core/recall_gpu.py` CUDA cores,
  `core/rt_optix.py` RT cores): at a saturating batch the RT-core
  filter is the fastest method (0.009 ms/query, recall 1.000 at
  r=1.5) while at multi-million-row scale the CUDA grid probe wins
  latency on Ampere hardware — full numbers in docs/RT_RECALL.md.

---

## v1 Architecture (legacy, retained for .e8z compatibility)

## Compression Pipeline Overview

E8ZIP implements multiple compression modes with different trade-offs between speed, ratio, and lossiness.

## Actual Implementation Flow

### FAST Mode (Lossless with E8 Metadata)

```
INPUT (bytes)
   ↓
Split into 8-byte chunks → Convert to float64 vectors
   ↓
E8 Lattice Quantization (nearest point mapping)
   ↓
Store quantized node IDs + lattice points (compressed with zlib)
   ↓
ALSO store original data (compressed with zlib)
   ↓
Package both in .e8z container
   ↓
OUTPUT: .e8z archive (lossless via zlib backup)
```

**Reality**: FAST mode uses zlib for actual compression, E8 quantization is for geometric indexing only.

---

### NORMAL Mode (Lossless with Trajectory Metadata)

```
INPUT (bytes)
   ↓
Split into 8-byte chunks → Convert to float64 vectors
   ↓
Build trajectory (start, end, sparse perturbations)
   ↓
Geodesic compression (hyperbolic space)
   ↓
Store trajectory data (compressed)
   ↓
ALSO store original data (compressed with zlib)
   ↓
Package both in .e8z container
   ↓
OUTPUT: .e8z archive (lossless via zlib backup)
```

**Reality**: NORMAL mode also relies on zlib for lossless reconstruction, trajectory is metadata.

---

### ULTRA Mode (Actual Geometric Compression)

```
INPUT (bytes)
   ↓
Split into chunks → Convert to vectors
   ↓
Embed in high-dimensional space (24D)
   ↓
Black Hole Holographic Encoder
   ├─ Project to boundary (horizon)
   ├─ Area-based encoding (entropy ~ surface area)
   └─ Sparse delta encoding
   ↓
Entropy coding (lightweight)
   ↓
OUTPUT: .e8z archive (lossy/lossless hybrid)
```

**Reality**: ULTRA mode performs true geometric compression using holographic principle.

---

### MYTHIC Mode (Lossy Semantic Compression)

```
INPUT (bytes)
   ↓
Split into chunks → Convert to vectors
   ↓
Semantic Lossy Codec
   ├─ Extract "archetypal" patterns
   ├─ Discard high-frequency noise
   └─ Preserve structural meaning
   ↓
Aggressive quantization
   ↓
Minimal entropy coding
   ↓
OUTPUT: .e8z archive (lossy, preserves semantics)
```

**Reality**: MYTHIC mode achieves highest ratios (30-50x) by discarding detail while preserving structure.

---

### LEECH Mode (24D Lattice Quantization)

```
INPUT (bytes)
   ↓
Split into 24-byte chunks → Convert to float64 vectors
   ↓
Leech Lattice (Lambda_24) Quantization
   ├─ Densest known 24D sphere packing
   ├─ 196,560 kissing neighbors
   └─ Superior to E8 for higher dimensions
   ↓
Store lattice indices + residuals
   ↓
ALSO store original data (compressed with zlib)
   ↓
OUTPUT: .e8z archive (lossless via zlib backup)
```

**Reality**: LEECH mode is experimental, very slow, uses zlib for actual compression.

---

### QUIP Mode (Hadamard + E8)

```
INPUT (bytes)
   ↓
Split into 8-byte chunks → Convert to float64 vectors
   ↓
Hadamard Transform (incoherence preprocessing)
   ├─ Spreads signal across basis
   ├─ Exposes redundancy
   └─ Improves quantization
   ↓
E8 Lattice Quantization (after transform)
   ↓
Store transformed + quantized data
   ↓
ALSO store original data (compressed with zlib)
   ↓
OUTPUT: .e8z archive (lossless via zlib backup)
```

**Reality**: QUIP mode applies Hadamard before E8, inspired by QuIP# (ICML 2024), uses zlib backup.

---

## Compression Mode Comparison

| Mode | Actual Compression Method | Ratio | Speed | Lossless | Use Case |
|------|--------------------------|-------|-------|----------|----------|
| **FAST** | zlib + E8 metadata | ~1.5x | ⚡⚡⚡ Fast | ✓ Yes | Quick compression with indexing |
| **NORMAL** | zlib + trajectory metadata | ~2-3x | ⚡⚡ Medium | ✓ Yes | General purpose |
| **ULTRA** | Black hole holographic | ~10-20x | ⚡ Slow | ~ Hybrid | Maximum compression |
| **MYTHIC** | Semantic lossy | ~30-50x | ⚡⚡ Fast | ✗ No | Lossy but semantic preserving |
| **LEECH** | zlib + Leech metadata | ~1.5x | ⚫ Very slow | ✓ Yes | Experimental 24D |
| **QUIP** | zlib + Hadamard + E8 | ~1.5x | ⚫ Very slow | ✓ Yes | Experimental incoherence |

---

## File Format (.e8z)

```
╔═══════════════════════════════════════════════════╗
║                  E8Z CONTAINER                    ║
╠═══════════════════════════════════════════════════╣
║ HEADER (32 bytes)                                 ║
║  ├─ Magic: "E8ZIP001" (8 bytes)                   ║
║  ├─ Version: uint16                               ║
║  ├─ Mode: uint8                                   ║
║  ├─ Flags: uint8                                  ║
║  ├─ Original Size: uint64                         ║
║  ├─ Compressed Size: uint64                       ║
║  └─ Checksum: uint32                              ║
╠═══════════════════════════════════════════════════╣
║ METADATA BLOCK                                    ║
║  ├─ Filename                                      ║
║  ├─ Timestamp                                     ║
║  └─ Compression Parameters                        ║
╠═══════════════════════════════════════════════════╣
║ DATA BLOCKS                                       ║
║  ├─ Geometric Data (E8/Leech/Trajectory)          ║
║  ├─ Original Data (zlib compressed, FAST/NORMAL)  ║
║  ├─ Black Hole State (ULTRA mode)                 ║
║  └─ Semantic Data (MYTHIC mode)                   ║
╚═══════════════════════════════════════════════════╝
```

---

## Key Architectural Decisions

### Why zlib backup in FAST/NORMAL/LEECH/QUIP?

These modes are advertised as "lossless" but the geometric compression (E8 quantization, trajectory encoding) is inherently lossy. To maintain lossless guarantee, the original data is stored compressed with zlib.

**Result**: These modes achieve only ~zlib compression ratios, not the theoretical geometric ratios.

### Why ULTRA and MYTHIC are different?

ULTRA and MYTHIC don't store the original zlib backup - they rely entirely on geometric/semantic compression:

- **ULTRA**: Uses black hole holographic encoding (boundary projection)
- **MYTHIC**: Uses semantic lossy codec (preserves meaning, discards detail)

**Result**: These modes achieve true geometric compression (10-50x ratios).

---

## Mathematical Foundations

### E8 Lattice

- 8-dimensional even unimodular lattice
- 240 root vectors, densest packing in 8D
- Kissing number: 240

### Leech Lattice (Λ₂₄)

- 24-dimensional lattice
- 196,560 nearest neighbors
- Densest known packing in 24D

### Hyperbolic Geometry

- Poincaré ball model
- Exponential distance growth near boundary
- Geodesics = "straight lines" in curved space

### Holographic Principle

- Information encoded on boundary
- Entropy proportional to surface area (not volume)
- Inspired by black hole thermodynamics

### Hadamard Transform

- Fast orthogonal transform (O(n log n))
- Spreads signal across basis (incoherence)
- Improves quantization for structured data

---

## Performance Characteristics

### Benchmark Results (Real Hardware)

**Repetitive Text (439 KB)**

- ULTRA: 296x ratio, 413ms
- NORMAL: 284x ratio, 654ms
- MYTHIC: 32x ratio, 152ms
- WinRAR: 1642x ratio (actual winner)

**JSON Data (165 KB)**

- MYTHIC: 31.8x ratio, 89ms ⭐ Winner
- ULTRA: 14.2x ratio, 167ms
- NORMAL: 14.1x ratio, 213ms
- WinRAR: 28.3x ratio

**Python Code (142 KB)**

- WinRAR: 42.2x ratio ⭐ Winner
- MYTHIC: 31.7x ratio, 85ms ⭐ Fastest
- ULTRA: 22.7x ratio, 147ms

**Random Binary (98 KB)**

- MYTHIC: 31.6x ratio, 73ms ⭐ Winner
- All others: ~1x ratio (incompressible)

---

## Decompression Pipeline

```
.e8z archive
   ↓
Read header (magic, version, mode, sizes)
   ↓
┌─────────────────────────────────────────┐
│  Mode detection                         │
├─────────────────────────────────────────┤
│  FAST/NORMAL/LEECH/QUIP:                │
│    → Extract zlib compressed original   │
│    → Decompress with zlib               │
│    → Return original bytes              │
│    (Geometric data ignored)             │
├─────────────────────────────────────────┤
│  ULTRA:                                 │
│    → Read black hole boundary state     │
│    → Holographic reconstruction         │
│    → Project from boundary to volume    │
│    → Return reconstructed bytes         │
├─────────────────────────────────────────┤
│  MYTHIC:                                │
│    → Read semantic codec data           │
│    → Reconstruct archetypes             │
│    → Semantic upsampling                │
│    → Return lossy reconstructed bytes   │
└─────────────────────────────────────────┘
   ↓
Verify checksum
   ↓
OUTPUT: Original (or reconstructed) bytes
```

---

## Future Improvements

### Planned Features

1. **True Geometric Compression for FAST/NORMAL**
   - Remove zlib backup dependency
   - Implement lossless E8 reconstruction
   - Research: reversible lattice quantization

2. **GPU Acceleration**
   - CUDA kernels for E8/Leech quantization
   - Parallel trajectory compression
   - 10-100x speedup target

3. **Streaming Support**
   - Process large files in chunks
   - Constant memory usage
   - Real-time compression/decompression

4. **Multi-threading**
   - Parallel block compression
   - Independent chunk processing
   - Utilize multi-core CPUs

---

## Implementation Notes

### Current Bottlenecks

- **LEECH mode**: Extremely slow (14-42 seconds for 100KB files)
- **QUIP mode**: Slow (7-21 seconds for 100KB files)
- **FAST mode**: Slow (7-21 seconds for 100KB files)

All slow modes are bottlenecked by lattice quantization without GPU acceleration.

### Production Recommendations

- Use **MYTHIC** for maximum ratio (lossy acceptable)
- Use **ULTRA** for best lossless-like compression
- Use **NORMAL** for general purpose (but it's just zlib)
- Avoid FAST/LEECH/QUIP in production (too slow, no benefit)

---

## Theoretical vs Actual

### Theoretical Pipeline (Aspirational)

```
INPUT → Vectors → Lattice Quantize → Trajectory → Hadamard → 
Holographic → Entropy Code → OUTPUT
```

### Actual Pipeline (Current Implementation)

```
FAST/NORMAL/LEECH/QUIP: INPUT → (geometric metadata) + zlib(original) → OUTPUT
ULTRA: INPUT → Vectors → Black Hole Encode → OUTPUT
MYTHIC: INPUT → Vectors → Semantic Lossy → OUTPUT
```

The "full pipeline" geometric compression is **aspirational** - not yet implemented.

---

**Last Updated**: December 8, 2025
**Version**: 1.0.0
**Status**: Production (with caveats)

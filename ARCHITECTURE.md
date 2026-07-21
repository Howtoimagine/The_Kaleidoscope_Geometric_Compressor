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
core/transforms.py     incoherence transforms: randomized Hadamard
                       (O(n log n), QuIP#) and dense random orthogonal
                       rotation (O(n^2), stronger decorrelation,
                       ported from the Glass Network's turbo_leech.py)
core/kgc.py            the KGC2 container + three regime front-ends,
                       Golay-protected header (see below)
core/klc.py            gain/shape progressive lattice codec (KLC),
                       ported and extended from the Glass Network's
                       KLC2 - see the honest comparison below
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

The KGC2 archive header's control block (version/regime/truth/flags) is
itself a Golay codeword: up to 3 flipped bits anywhere in those fields
are healed transparently on decode, 4+ are detected and rejected rather
than silently misparsed. Verified in `tests/test_klc.py::TestHeaderHealing`
(30/30 random 1-3-bit-flip trials healed).

### Three regimes, one object

```
BYTES        input bytes -> adaptive context model -> range coder
             raw fallback if the model fails to shrink (random data
             stays ~1.0x). sha256-verified on decode. Lossless.

TENSOR       flatten -> incoherence transform (Hadamard or rotation) ->
             normalize -> scale (the rate-distortion knob) -> 24-D
             blocks -> exact Leech CVP -> entropy-coded (g,i,z).
             v2.1 default seeds the z model with the discretized
             Gaussian implied by scale (FLAG_Z_PRIOR, header flags
             byte; 2.0 archives still decode).
             Measured on synthetic Gaussian weights: 4.13 bits/weight
             @ 23.9 dB SNR (was 4.20 before the z prior) vs INT4
             scalar 4.0 bpw @ 15.8 dB and INT5 5.0 bpw @ 22.0 dB.
             Measured on REAL weights (all-MiniLM-L6-v2): 23.85 dB @
             4.34 bpw, beating the Glass Network's own Leech-VQ
             reference (19.85 dB @ 4.81 bpw) - see the full table
             below.

CONSOLIDATE  (N,24) rows *rg_scale -> Leech sites (block-spin RG step);
             rows sharing a site collapse to one representative.
             Per-site conserved mass is stored and reconstruction
             rescales so group energy matches exactly (0.000% drift).
             keep_residuals=True stores XOR-exact IEEE-754 residuals
             (bitwise round-trip, verified); False drops them but
             keeps addresses -> auditable lossy collapse.
```

A fourth path sits alongside TENSOR rather than replacing it:

```
PROGRESSIVE  (core/klc.py) per-24-D-block gain/shape separation ->
             exact Leech CVP of the (scaled) unit-direction -> Golay
             coset base layer -> progressive int8 residual layers,
             each its own entropy-coded, length-prefixed, truncatable
             segment. Decode may stop at any layer for a smaller,
             lower-fidelity result - the flat TENSOR regime cannot do
             this at all (all-or-nothing).
```

### Measured on real weights (all-MiniLM-L6-v2, not synthetic)

Every number below is a verified round-trip against actual
`sentence-transformers/all-MiniLM-L6-v2` weight tensors (fetched via
`huggingface_hub`), matching the Glass Network's own benchmark
methodology (`benchmarks/turbo_leech_rate_distortion.py`,
glass_windows branch) so the comparison is apples-to-apples:

| Method | dB (SNR) | bits/weight |
|---|---|---|
| **KGC TENSOR regime** (this repo) | **23.85** | **4.34** |
| Glass Network reference Leech VQ (no entropy coding of indices) | 19.85 | 4.81 |
| Scalar INT5 (entropy-coded) | 20.38 | 3.74 |
| Scalar INT6 (entropy-coded) | 26.71 | 4.76 |

The TENSOR regime beats the Glass Network's own published Leech-VQ
number outright - more fidelity, fewer bits - which the exact-CVP
decoder plus real adaptive entropy coding (rather than a marginal-
entropy estimate) accounts for. It's ahead of INT5 at a comparable
rate and in the same neighborhood as INT6 while still exposing a
continuous rate-distortion knob (`scale`) that discrete bit-widths
don't have.

**PROGRESSIVE (KLC) - an honest negative result.** The working
hypothesis going in was that separating magnitude from direction
per-block would *also* win on rate-distortion, on top of adding
truncatable decode. Measured on the same real tensors, after tuning
(`shape_scale` 16->48, `gain_k` 32->128 - a real improvement over the
first guess): **it does not beat TENSOR at any tested rate.**

| Operating point | KGC TENSOR | KLC (tuned) |
|---|---|---|
| low rate | 23.91 dB @ 4.57 bpw | 20.31-30.38 dB @ 4.11-6.09 bpw (defaults vs tuned) |
| matched high rate | 30.93 dB @ 5.85 bpw | 30.38 dB @ 6.09 bpw |

At matched fidelity (~30.4-30.9 dB) TENSOR needs *fewer* bits. KLC's
real, verified value is the truncatable/progressive property itself
(byte-exact truncation points, strictly monotonic fidelity per layer -
`tests/test_klc.py::test_progressive_fidelity_is_monotonic`), not
compression ratio. Use it for partial-fidelity streaming or bandwidth-
adaptive serving; use TENSOR for the best ratio. `benchmarks/
benchmark_real_weights.py` reproduces both tables.

### Lineage

The math is ported from the Glass Network (glass_windows branch):
`packages/kmind/golay.py`, `packages/kmind/leech.py` (quantizer),
`packages/kdot_v2/lattice_codec.py` (KLC2 - gain/shape + progressive
layers), `packages/kmind/model_cookbook/turbo_leech.py` (dense
rotation transform, real-weight benchmark harness),
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

### Measured on a real generative LLM (v2.2, perplexity - not weight SNR)

Every number above this point is weight-space SNR/MSE - a proxy for
what actually matters. `benchmarks/benchmark_llm_perplexity.py` closes
that gap: it dequantizes into an actual model and measures WikiText-2
perplexity after real inference, matching how GPTQ/AWQ papers report
results, on `Qwen/Qwen3.5-2B-Base` (a real, current hybrid linear/full-
attention generative model - not a small encoder).

**Honest scope**: exact Leech CVP is too slow to quantize a full 1.37B-
parameter model in reasonable time (see below), so this measures 3
representative Linear layers (12.58M params, 0.92% of the model,
spanning both the gated-linear-attention and full-GQA-attention layer
types) with everything else held at fp32 - not a full-model result.

| Method | WikiText-2 ppl | bits/weight | ppl delta vs fp32 |
|---|---|---|---|
| fp32 (unquantized) | 8.887 | 32.00 | - |
| RTN INT4 (naive baseline) | 9.242 | 4.00 | +0.355 |
| **GPTQ INT4** (Frantar et al. 2022, own implementation) | **8.922** | 4.00 | **+0.035** |
| **KGC TENSOR** (scale=4.0) | **8.957** | 4.08 | **+0.070** |

KGC clearly beats naive RTN at the same bit-width (+0.070 vs +0.355 ppl
- about 5x less degradation) but does not match GPTQ (+0.070 vs
+0.035 - GPTQ is about 2x closer to fp32). This is the first real
signal on the question "is this a good LLM quantizer": genuinely
competitive with a real PTQ baseline, not yet state-of-the-art against
the specific method (Hessian-based error compensation) it was modeled
after. Full-model, full-eval-set numbers remain future work pending a
faster CVP path - see below. **The activation-aware upgrade in the next
section (Resonant) closes this gap on the same 3 layers - to +0.002,
past GPTQ.**

### Resonant Quantization: MEASURED - the gap is closed (v2.3)

`core/resonant.py` builds the activation-aware form of the idea. On the
same 3 layers and the same WikiText-2 eval as the v2.2 table, it reduces
the quantization penalty to essentially zero - past GPTQ, not just up to
it (`res_eval`, seq_len 1024, 40 windows; fp32 and KGC reproduce the v2.2
numbers exactly, confirming the harness is identical):

| Method (3 layers, 1024-token calib) | WikiText-2 ppl | ppl delta vs fp32 |
|---|---|---|
| fp32 (unquantized) | 8.887 | - |
| KGC TENSOR (plain Leech CVP, scale=4.0) | 8.957 | +0.070 |
| GPTQ INT4 (own impl, matched calib) | 8.944 | +0.057 |
| **Resonant (AWQ-in-the-lattice, alpha=0.5)** | **8.889** | **+0.002** |

Resonant lands ~35x closer to fp32 than plain KGC (+0.002 vs +0.070). Two
honesty notes, because the result is strong enough to invite suspicion:

1. **GPTQ is calibration-starved in this run.** With 1024 calibration
   tokens for 2048-dim inputs its input Hessian is rank-deficient; GPTQ's
   error feedback needs a full-rank Hessian, so it degrades to +0.057
   here (`gptq_quantize_` now escalates its damping to stay numerically
   sound in exactly this regime - a fix, not a fudge; the unobserved
   null space falls back to plain rounding, which is correct). The v2.2
   table above, run with a full-rank Hessian, is GPTQ at proper strength:
   +0.035. Resonant needs only per-channel activation RMS - a first
   moment well-estimated from 1024 tokens - so it is far more robust to
   thin calibration. It beats GPTQ in BOTH regimes: +0.002 vs +0.057
   matched-calibration, and vs +0.035 full-Hessian. (Needing ~10x less
   calibration data than GPTQ is the known AWQ advantage, reproduced here
   inside the lattice.)
2. **This is 3 layers, not the model.** 0.92% of the weights, held-out
   perplexity, a single global alpha=0.5. Not a full-model claim.

This is a DISTORTION win, not a rate win. `benchmarks/benchmark_resonant.py`
first ruled out the rate hypothesis (the founding "compress geodesic PATHS
not nodes" slogan) on these real weights: the block-to-block deltas sit on
~2x HIGHER Leech shells than the absolute points, g_idx is already near-
uniform (11.4 / 12 bits), and there are zero duplicate blocks - at ~4
bits/weight the blocks are essentially independent high-shell points, so
there is no free lunch in path / codebook / predictive index coding. The
only lever left is distortion, and Resonant pulls it.

Why it works - the DISTORTION lever, now measured per layer. The Leech
CVP minimizes weight error `||W - What||`; the model experiences output
error `||(W - What) X||`. Scaling each input channel by `s^alpha` (s =
activation RMS) before snapping spends the lattice's resolution on the
channels the model actually reads through, at the identical bit-rate:

| layer | KGC out-MSE | Resonant out-MSE | GPTQ out-MSE |
|---|---|---|---|
| L0.out_proj  | 0.700% | **0.120%** | 0.285% |
| L0.in_proj_z | 0.194% | 0.183% | 0.056% |
| L3.o_proj    | 0.798% | **0.330%** | 0.273% |
| **mean**     | **0.564%** | **0.211%** | 0.205% |

(output-MSE as % of signal power on the calibration activations). Resonant
cuts KGC's layer output error 2.7x - to parity with a full-strength GPTQ -
by accepting ~60% more WEIGHT error (the exact AWQ/GPTQ trade), and that
2.7x output-error reduction is what collapses the perplexity penalty from
+0.070 to +0.002. Identity used: `y = W x = (W diag(s^a))(diag(s^-a) x)`;
store `s` per input channel (side info, amortized over all output rows)
plus the lattice archive. `alpha` is searched to minimise real layer
output error; alpha=0 recovers plain KGC, so it can only match or beat it.

**The recall-reranked form - "GPTQ on the Leech lattice" (v2.4).**
`resonant_rerank_quantize` (core/resonant.py) keeps the objective - minimise
disagreement, not distance - but instead of the single Euclidean-nearest
Leech point per 24-D block, it enumerates a candidate set (Hessian-whitened
dither / list-decode - the **recall engine** move) and snaps to the
candidate that minimises the model's OUTPUT error. Getting the objective
right turned out to be the whole game, in two steps:

1. **Block-diagonal proxy** - rank candidates by `(x-c) H_bb (x-c)` using
   only the block's local 24x24 Hessian. This is a WEAK, non-robust top-up
   (+1..+9%, and it goes *negative* at the coarsest rate) - it ignores how
   one block's rounding error propagates into every other block.

2. **Coupling (the real lever)** - the exact output error is `r^T H_r r`
   (r = x - chosen, `H_r` the full transformed Hessian; exact because the
   Hadamard is orthogonal). Minimise it by GAUSS-SEIDEL coordinate descent:
   update blocks in sequence, maintain the gradient `G = r @ H_r`
   incrementally, and snap each block to
   `argmin (x-c) H_bb (x-c) + 2(x-c).g_b` with the fresh coupling
   `g_b = (G - r_b H_bb)_b` from the blocks already updated. This is
   *exactly* GPTQ's error feedback - but across LATTICE blocks, snapping to
   the best of K candidates instead of scalar rounding. Since c0 is always a
   candidate the true error is monotone non-increasing; a Jacobi (parallel)
   update instead DIVERGES and must be sequential.

Measured (L3.o_proj, output-MSE reduction vs the Euclidean snap = scalar
Resonant, `benchmark_resonant_rerank.py`, RATE-MATCHED - same bits):

| scale / bits | block-diagonal | **coupling (3 sweeps)** |
|---|---|---|
| ~4.11 bits | +7.6% | **+62.9%** |
| ~3.70 bits | +9.6% | **+63.7%** |
| ~3.44 bits | -2.7% | **+59.6%** |

The coupling objective delivers ~+60% output-MSE reduction at every rate -
an order of magnitude past the block-diagonal proxy, holding exactly where
the proxy collapses, and at the SAME bit-rate (the chosen points' Leech
index entropy is unchanged; it picks *better* candidates, not costlier
ones). `sweeps=1` already reaches ~+58%, and it generalizes across layer
types (L0.out_proj, linear-attention: +49% at ~4 bits). This is the point the whole arc
was aiming at: the lattice's coding gain (denser than scalar in 24-D) AND
GPTQ's Hessian error feedback, in one quantizer - candidate enumeration in
the densest lattice in 24 dimensions in place of a scalar Cholesky feed.
The **Golay** coset label keeps every candidate self-healing; the **theta
series** prices each shell in closed form. Quantization and associative
recall were the same operation all along (snap a noisy vector to the
nearest stored attractor); Resonant makes the attractors the model's own
behaviour.

**But the end-to-end 3-bit perplexity says the coupling win is mostly
OVERFITTING - and it is the most important result in this section.** Run
on the 3 layers at ~3.07 bits (matched to GPTQ INT3's 3.0), held-out
WikiText-2:

| method (3-bit) | calibration out-MSE (mean) | held-out ppl | ppl delta |
|---|---|---|---|
| fp32 | - | 8.887 | - |
| KGC (alpha=0) | 2.46% | 9.013 | +0.125 |
| **scalar-Resonant (alpha=0.5)** | 0.94% | **8.912** | **+0.025** |
| coupling (K=4, 3 sweeps) | **0.44%** | 8.944 | +0.056 |
| coupling + damp=2.0 | - | 8.915 | +0.028 |
| GPTQ INT3 | 1.10% | 9.199 | +0.311 |

The ordering FLIPS between the calibration proxy and the real metric.
Coupling has the LOWEST calibration output-error (0.44%, ~2x better than
scalar) yet a WORSE held-out perplexity than scalar (+0.056 vs +0.025);
GPTQ likewise fits calibration better than KGC but generalizes far worse
(+0.311). The aggressive Hessian-fitting methods (coupling, GPTQ) overfit
the 1024-token calibration Hessian - which is rank-deficient (1024 signal
directions, 1024 null), so their coordinate/error-feedback descent spends
effort fitting noise the null space, and it does not transfer. The robust
first-moment method - scalar AWQ per-channel scaling, which only reads
activation RMS - wins, and its margin over both KGC and GPTQ actually
GROWS at 3 bits (scalar +0.025 vs GPTQ +0.311: 12x). This is the textbook
AWQ-beats-GPTQ-under-thin-calibration result, reproduced inside the
lattice, and a caution the whole project earns the hard way: a 60%
calibration-proxy win was 60% overfitting.

**Hessian damping was tried, and it closes the case.** Adding
`damp * mean(diag) * I` to the coupling Hessian (GPTQ's percdamp), tuned
on HELD-OUT activations (not calibration - the mistake is not repeated),
lands at damp=2.0: it beats scalar by ~12-20% on held-out output-MSE and
pulls the perplexity back from +0.056 (overfit) to +0.028 - exactly
undoing the overfitting the diagnosis predicted. But +0.028 only TIES
scalar's +0.025 (within eval noise); it does not beat it. So even
correctly regularised, the error-feedback machinery (8x the CVP cost, a
sequential coordinate descent) buys NOTHING over plain scalar AWQ scaling
at 3 bits - robust per-channel scaling already captures essentially all
the achievable gain in this regime. The honest 3-bit winner is
**scalar-Resonant** (`rerank=False`): lattice coding gain + robust
scaling, no feedback, cheapest of all, and it beats GPTQ 12x. Coupling
(damped or not) is retained as a correct, documented tool and a clean
negative - feedback may still pay at other bit-rates or with full-rank
calibration, but it does not here.

### Standard-benchmark positioning (v2.3, industry tools)

Two benchmarks against the recognised competitors, run with their own
standard tooling and metrics:

**Lossless bytes vs general-purpose codecs** (`benchmark_lossless_std.py`,
2 MB of enwik8, bits/byte):

| codec | bits/byte |
|---|---|
| brotli -11 | **2.24** |
| xz / LZMA -9 | 2.29 |
| bzip2 -9 | 2.31 |
| gzip -9 | 2.90 |
| KGC bytes (order-2) | 3.10 |

The byte regime loses to every standard codec (1.38x brotli's size,
worse than gzip) and is ~1000x slower - as expected; it is a verified
debt container, not an LZ codec. No surprise, now quantified.

**Vector quantization vs FAISS** (`benchmark_faiss_vq.py`, real 384-D
MiniLM embeddings, recall@10 vs exact top-10) - this is the fair test of
the system's actual strength, against Product Quantization, the industry-
standard vector quantizer that does the same job:

| method | bits/vec | recall@10 |
|---|---|---|
| OPQ M=64 (learned) | 512 | 0.841 |
| KGC Leech s=0.75 | 733 | 0.848 |
| KGC Leech s=1.5 | 1049 | 0.924 |
| SQ4 (scalar) | 1536 | 0.942 |
| **KGC Leech s=3.0** | 1413 | **0.960** |
| SQ8 (scalar) | 3072 | 0.995 |

KGC's fixed-lattice VQ sits *between* scalar and learned quantization: it
**beats scalar quantization** (0.960 recall @ 1413 bits vs SQ4's 0.942 @
1536 - better recall, fewer bits, from the Leech lattice's superior 24-D
packing) but **loses to PQ/OPQ in the aggressive-compression regime**
(OPQ 0.841 @ 512 bits; KGC needs ~730+ to match). The reason is exactly
the theme of this whole document: PQ/OPQ *learn* a data-adaptive codebook
(k-means centroids where the data lives, plus a learned rotation); KGC
uses the *fixed* lattice and cannot concentrate its bits on the data
distribution. The gap PQ opens is the measured value of a learned
codebook - the same thing the consolidation / Resonant machinery above is
designed to add. Honest current standing: a strong high-fidelity
quantizer, not yet a PQ replacement for billion-scale ANN.

### A falsified fix: learned rotation does not deform the lattice (v2.3)

The obvious way to close the PQ gap is OPQ's trick - learn an orthogonal
rotation that aligns the data to the quantizer's grid (OPQ's whole edge
over plain PQ). `core/adaptive.py` implements exactly that for the Leech
lattice (non-parametric OPQ: alternate lattice-quantize / orthogonal
Procrustes). **Measured: it does nothing** - plain 3.044e-4 vs learned
3.041e-4 MSE (0.1%) on real embeddings, and 0.0% even on deliberately
anisotropic data built for the rotation to exploit.

The reason is fundamental and worth stating: the Leech lattice is near-
ISOTROPIC (its Voronoi cell is close to spherical), so rotating the data
does not change how well the lattice tiles it. OPQ's rotation helps PQ
*because PQ is anisotropic* - axis-aligned subspace codebooks whose
variance imbalance the rotation fixes; the lattice has no preferred axes.
The lattice's isotropy is its strength (why it beats scalar quantization)
and simultaneously why the OPQ trick cannot transfer. The PQ gap is
therefore STRUCTURAL - it requires a data-adaptive codebook (learned
centroids), i.e. leaving the lattice for PQ/AQLM. The lever that *does*
move a lattice quantizer is anisotropic SCALING (per-axis resolution),
not rotation (which it is invariant to) - which is exactly why the
activation-aware Resonant kernel worked (+4% on weights) and this did
not. Kept as a committed negative, in the house style of the glass-
network's own preregistered falsifiers.

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
- The progressive codec (KLC) is rate-distortion-dominated by the flat
  TENSOR regime - see the measured comparison above. Its residual-layer
  step schedule is a fixed halving sequence (not adaptive to measured
  residual variance), which is the likely fixable cause; untried in
  this pass.
- Leech CVP is exact but CPU-heavy in pure numpy: ~300-350 blocks/s in
  isolation (steady from N=200 to N=20,000 - no degradation with
  scale, ~1.2 GB peak after the v2.2 chunking fix below), but measured
  at ~22 blocks/s (~15x slower) quantizing real layers *inside the
  same process as a loaded PyTorch model* - torch claims a thread per
  core (`torch.get_num_threads()` == core count) while numpy's OpenBLAS
  backend independently tries to do the same for every matmul in the
  CVP search, and on a 4-core box those two uncoordinated thread pools
  oversubscribe every physical core. This is a real deployment
  consideration, not a correctness bug: pin `OPENBLAS_NUM_THREADS=1` or
  quantize as a separate process from inference/calibration until a
  CUDA CVP kernel exists. At the isolated rate, the full 1.37B-param
  model would take ~45 hours; at the measured in-process rate, closer
  to a month - both are why the v2.2 LLM perplexity benchmark (above)
  is honestly scoped to 3 layers, not the full model.
- A v2.2 fix: `batch_nearest_leech_point` computed its per-case
  `dist_k`/`c_sum_k`/`parity_match` arrays for the *entire* input
  before chunking anything - ~5.7 GB each at N=175k (one real
  2048x2048 LLM layer), which OOM-killed the process outright on a
  15 GB box. Fixed by moving the chunk boundary earlier so every
  intermediate array is bounded regardless of input size; verified via
  the full test suite plus direct throughput/memory checks from
  N=200 to N=20,000.
- GPU *recall* over stored rows, though distinct from CVP quantization
  itself, is built and measured (`core/recall_gpu.py` CUDA cores,
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

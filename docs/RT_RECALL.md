# GPU Recall-as-Rays: RT-Core-Accelerated Recall for the KGC Codec

Feasibility study (2026-07-12) for mapping KGC memory recall — k-nearest
stored rows over Leech/E8-quantized 24-D sites — onto NVIDIA RT cores
(OptiX). CPU reference: `core/recall.py`; CUDA-cores backend:
`core/recall_gpu.py`; **working OptiX RT-core backend:
`core/rt_optix.py`**; decision-gate benchmark:
`benchmarks/benchmark_recall.py`.

> **Status: BUILT AND MEASURED.** The OptiX port is implemented and runs
> — NVRTC-compiled device programs (degenerate rays + a custom
> intersection/any-hit pair), a GAS of one AABB per stored row per
> projection, votes and exact re-rank in CuPy. Stack installed on this
> machine: pyoptix 9.1, CuPy 14.1 (cuda12x), CUDA 12.9/13.0, OptiX
> headers, VS2022 Build Tools + Win11 SDK, on an RTX 3070 Ti. Measured
> results are in "Measured results" below; the short version: **RT cores
> deliver the best recall of any method (1.000, exact BVH containment
> beats grid-cell approximation) and win on latency at small saturating
> batches, but a tuned CUDA-cores grid probe is competitive-to-faster at
> multi-million-row scale on this 2nd-gen-RT-core (Ampere) card.**

## Executive summary

**Verdict: worth a small prototype, with tempered expectations.** Mapping
neighbor search onto RT cores is a well-established technique with ~8
peer-reviewed systems since 2021 (RTNN, TrueKNN, Arkade, RT-DBSCAN,
RTIndeX, JUNO, RTScan, plus 2025–26 follow-ups). Reported wins are real
but concentrated in **low-dimensional (≤3-D), fixed-radius, batch-query**
workloads — exactly the shape of a "project 24-D sites to 3-D,
radius-filter, exact re-rank" pipeline. The one system that tackles
genuinely high-dimensional ANN on RT cores (JUNO, ASPLOS 2024) does it
via product-quantization subspaces mapped to low-D ray queries and
reports 2.2–8.5x throughput over GPU baselines on 1M–100M point
datasets — meaningful but not the 100x headline numbers from pure 3-D
workloads.

**Do not use RT cores for lattice decode.** Leech/E8 CVP via the
Golay-code decoder is a few hundred fixed ops per vector and is
embarrassingly parallel on CUDA/CPU; no RT paper even attempts
structured-lattice decode because there is nothing to search. The
interesting target is **content recall over millions of stored 24-D
rows**: 3-D projection as a coarse candidate filter feeding exact 24-D
re-rank. That is precisely the "filter-refine" reduction Arkade
formalizes (exact when the filter distance lower-bounds the true
distance — a property scaled random projections give as a bound in
expectation).

The honest caveat: a well-tuned CUDA brute-force re-rank over ~10⁶ 24-D
rows is already fast (~ms), and lattice structure gives a *free*
competing index — bucket rows by their coarse-lattice cell ID (lattice
hashing / spherical LSH), which needs no BVH at all. The RT prototype is
justified when N grows past ~10⁷ rows or for high-throughput query
batches; below that, lattice-cell hashing will likely win on simplicity.

## Measured results (RTX 3070 Ti, 8 GB, driver 610.74)

Clustered memory bank (archetypes + noise — the realistic recall
workload; uniform-random rows are the worst case for every method).
recall@k is against exact GEMM top-k. `benchmarks/benchmark_recall.py`.

**Small saturating batch — N=100k, 4096 queries:**

| method | recall@10 | query ms/q |
|---|---|---|
| exact GEMM (GPU) | 1.000 | 0.012 |
| CUDA grid probe, r=1.0, v=2/4 | 0.893 | 0.011 |
| CUDA grid probe, r=1.5, v=2/4 | 0.995 | 0.023 |
| **OptiX BVH, r=1.0, v=2/4** | 0.934 | **0.009** |
| **OptiX BVH, r=1.5, v=2/4** | **1.000** | 0.019 |

At saturation the RT-core path is the fastest method at r=1.0 (0.009
ms/q, 25% under exact GEMM) and the only method reaching recall 1.000.

**Multi-million-row — N=2M, 4096 queries:**

| method | recall@10 | build s | query ms/q |
|---|---|---|---|
| exact GEMM (GPU) | 1.000 | 0.06 | 0.198 |
| CUDA grid probe, r=1.0, v=2/4 | 0.891 | 0.08 | **0.072** |
| CUDA grid probe, r=1.5, v=2/4 | 0.995 | 0.06 | 0.230 |
| OptiX BVH, r=1.0, v=2/4 | 0.932 | 2.96 | 0.196 |
| **OptiX BVH, r=1.5, v=2/4** | **1.000** | 2.11 | 0.929 |

**Honest reading.** At 2M rows the CUDA-cores grid probe is the fastest
(0.072 ms/q) and OptiX no longer wins on latency — the clustered
workload yields few hits per query, so RT-core traversal never saturates
enough to amortize its per-ray shader-call and BVH-build overhead
(2–3 s/build at 2M boxes), and our vote + re-rank + host-transfer tail
dominates. RT cores still deliver strictly the **best recall (1.000)**
because BVH containment is exact where the grid probe only inspects 27
neighbor cells. This matches the literature's caveat exactly: RT wins are
workload- and hardware-dependent, and a well-tuned CUDA baseline is
strong. On a 3rd/4th-gen-RT-core card (Ada/Blackwell, ~2× BVH throughput
per generation) and with hit-heavier workloads the balance shifts toward
RT; on this Ampere card, the honest recommendation is **CUDA grid probe
for latency at scale, OptiX BVH when recall must be exact.**

Reproduce: `python benchmarks/benchmark_recall.py 100000 4096 --gpu-saturate`
and `... 2000000 4096 --gpu-saturate --repeat 3`.

## Prior-art table

| System | Venue/Year | Technique | Reported numbers |
|---|---|---|---|
| Fast Radius Search (Evangelou et al.) | JCGT 2021 | Inverse mapping: sphere of radius r at each point, degenerate short ray at query; kNN + FRNN in 3-D | up to 5x vs CPU, 2.3x vs GPU state of the art |
| RTNN (Zhu) | PPoPP 2022 | Fixed-radius NN as ray-sphere/AABB intersection; query scheduling + partitioning; 3-D point clouds | 2.2x–65x over existing GPU neighbor-search libraries |
| TrueKNN / RT-kNNS Unbound | ICS 2023 | First *unbounded* kNN on RT cores: iteratively grows search radius, retires satisfied queries per round | 1.5x–8x vs RTNN; up to ~200x vs baselines in best cases |
| RT-DBSCAN | IPDPS 2023 | DBSCAN epsilon-neighborhood queries as ray queries | 1.3x–4x over SOTA GPU DBSCAN |
| RTIndeX / RX | VLDB 2023 | DB point/range lookups: keys as triangles, lookups as rays | Competitive with GPU hash/tree indexes; weak: memory/key, ranges, updates |
| JUNO (Liu, Zhu et al.) | ASPLOS 2024 | High-dim ANN: PQ decomposition; codebook centroids as spheres in low-D subspaces, query subvectors fire rays; sparse accumulation on tensor cores | 2.2x–8.5x throughput on 4 datasets, 1M–100M points; algorithm alone up to 2.6x without RT cores |
| Arkade (Mandarapu et al.) | ICS 2024 | Filter-Refine and Monotone-Transformation reductions: non-Euclidean kNN via Euclidean RT queries | 1.6x–200x vs shader-core baselines; 1.3x–33.1x vs RT-core baselines |
| RTScan | VLDB 2024 | Table scans on RT cores | (database workload) |
| Advancing RT-Core FRNN | arXiv 2026 (2601.15633) | Improved traversal/memory-access over RTNN/Evangelou | incremental; confirms BVH build cost as standing overhead |
| RT-RkNN | arXiv 2026 (2605.26671) | Reverse-kNN as ray casting | new query variant, same encoding pattern |
| RT-cores literature review | arXiv 2026 (2603.28771) | Survey | ANN "is the only problem in which an arbitrary dimension has been mapped to RT cores"; RT cores improve ~2x per hardware generation |

## Core technique (consensus across papers)

- **Encoding:** each stored point becomes a sphere (or its AABB) in the
  BVH; the query becomes a ray with origin at the query point.
- **Degenerate rays:** t_min≈0, t_max≈epsilon — only the "which volumes
  contain my origin" test; all handling in an any-hit/intersection
  shader that records hits without terminating traversal.
- **Fixed radius is the native operation.** True kNN needs
  TrueKNN-style iterative radius growth, converging in a few rounds.
- **Known limits:** strictly 3-D FP32 coordinates; BVH is a black box;
  build/refit cost amortizes only over many queries or static data;
  hit-heavy queries (radius too large) collapse to CUDA-like
  performance. Radius selection is the main tuning knob everywhere.
- **Hardware trend:** ~2x BVH throughput per RT-core generation
  (Turing → Ampere → Ada → Blackwell), so the technique ages well.

## High-dimensional data on 3-D hardware

1. **Single random projection 24→3:** far below JL-safe; usable only as
   a filter with generous radius + exact re-rank.
2. **Multi-projection voting (recommended):** m ≈ 4–8 independent 3-D
   projections, one BVH each; candidate = within radius in ≥ v of m
   projections. Drives false positives down exponentially in v. No
   published RT-core paper does exactly this — a small novelty gap.
3. **PQ-subspace decomposition (JUNO's route):** split 24-D into eight
   3-D subspaces; centroids as spheres, query subvectors fire rays,
   sparse accumulation scores candidates. The only *proven* high-dim
   RT-core recipe. Pleasant coincidence: 24 = 8 × 3 exactly, and
   Leech/E8 coordinates give natural well-conditioned subspace splits.
4. **Space-filling curves:** abandons RT hardware; CPU fallback only.

## Recommended prototype design

```
Stage 0 (offline, per memory bank):
  - Stored sites: N x 24-D rows (already Leech/E8-quantized)
  - m=4 fixed random orthonormal 24->3 projections P_1..P_4 (seeded,
    stored in the codec header)
  - Per projection: one OptiX GAS over N spheres (center = P_i @ row,
    radius r_i calibrated so true neighbors within 24-D distance R fall
    inside with prob >= ~0.98; scale projections by sqrt(24/3))

Stage 1 (query, RT cores):
  - Batch queries; per projection, launch degenerate rays from P_i @ q
  - Any-hit shader appends (query_id, site_id) pairs (atomics)

Stage 2 (vote + re-rank, CUDA/CuPy):
  - Histogram site_ids per query; keep votes >= v (start v=2 of 4)
  - Exact 24-D re-rank of survivors, top-k out

Stays OFF the RT path:
  - Leech/E8 CVP decode: exact Golay decoder, O(1) per vector.
  - Baseline #1: CUDA/NumPy GEMM brute force.
  - Baseline #2: lattice-cell hashing (coarse-Leech cell id as hash key)
    — exploits the exact-CVP asset directly, no GPU at all.
```

**Decision gate:** run all three on N = 10⁶ and 10⁷ rows
(`benchmarks/benchmark_recall.py` is the harness; the RT numbers come
after an OptiX port). Adopt RT only if it wins at the real corpus size
and batch shape by >2x.

## Practical stack: Windows 11 + RTX + Python

- **Primary: OptiX via `NVIDIA/otk-pyoptix`** — official Python bindings
  for the OptiX host API. **Verified working install on this machine
  (Windows 11, RTX 3070 Ti):**
  1. `pip install cupy-cuda12x` (CuPy 14.1) and `cuda-python` (nvrtc).
  2. VS2022 Build Tools with the "Desktop development with C++" workload
     **and a Windows 11 SDK component** (`winget install
     Microsoft.VisualStudio.2022.BuildTools --override "--add
     Microsoft.VisualStudio.Workload.VCTools --includeRecommended"`, then
     add `Microsoft.VisualStudio.Component.Windows11SDK.26100` via an
     **elevated** `setup.exe modify` — a non-elevated `--quiet` modify
     fails with exit 5007, and `--wait` is not a valid modify flag).
  3. `pip install pyoptix` — it fetches OptiX headers automatically and
     builds the `optix` module from source via CMake/scikit-build. If it
     picks a generator whose compiler it can't find, force the installed
     one: `CMAKE_GENERATOR="Visual Studio 17 2022" pip install pyoptix`.
  Device programs (intersection/any-hit) are CUDA C compiled to PTX by
  NVRTC at first use (`core/rt_optix.py::_compile_ptx`) — hit shaders
  cannot be written in Python. Buffers interop with **CuPy**, so vote +
  re-rank stay on device without host round-trips.
- **Alternative: Vulkan `VK_KHR_ray_query`** from a compute shader —
  vendor-neutral, but heavier Python tooling and no CUDA interop.
  Choose only if AMD/Intel portability matters.
- **Fallback baselines:** CuPy GEMM and FAISS-GPU for reference curves.
- Minimum: RTX 20-series+ (Ada/Blackwell recommended), driver ≥ R560,
  CUDA 12.6+, OptiX SDK 9.x, Python 3.10+.

## Key risks

1. **The filter may not beat GEMM.** At 24-D, fused GEMM + top-k over
   10⁶ rows is a few ms on any modern RTX card. Mitigated by the
   three-way bake-off gate.
2. **3-D projection recall.** 24→3 is far below JL-safe; the
   multi-projection vote is load-bearing and unproven in the literature.
3. **Radius tuning / hit explosion.** Too-large r floods the any-hit
   shader (atomic contention) — the dominant failure mode in RTNN and
   the 2026 FRNN paper.
4. **BVH rebuild on ingest.** Batch ingest and refit, don't rebuild per
   insert.
5. **Engineering surface.** OptiX device code adds a native build step
   to a pip-friendly codec.
6. **FP32 only.** Fine for a coarse filter; condition projection
   matrices to avoid cancellation.

## References

- RTNN (PPoPP 2022): https://arxiv.org/abs/2201.01366 — code: https://github.com/horizon-research/rtnn
- TrueKNN / RT-kNNS Unbound (ICS 2023): https://arxiv.org/abs/2305.18356 — code: https://github.com/vani-nag/OWLRayTracing
- Arkade (ICS 2024): https://arxiv.org/abs/2311.09168 — code: https://github.com/MDurgaKeerthi/Arkade
- Fast Radius Search (JCGT 2021): https://jcgt.org/published/0010/01/02/paper-lowres.pdf
- RT-DBSCAN (IPDPS 2023): https://arxiv.org/abs/2303.09655
- RTIndeX (VLDB 2023): https://arxiv.org/abs/2303.01139
- JUNO (ASPLOS 2024): https://arxiv.org/abs/2312.01712 — pdf: https://horizon-lab.org/pubs/asplos24-juno.pdf
- RTScan (VLDB 2024): https://dl.acm.org/doi/abs/10.14778/3648160.3648183
- RT Cores for General-Purpose Computing: A Literature Review (2026): https://arxiv.org/html/2603.28771v1
- Advancing RT Core-Accelerated FRNN (2026): https://arxiv.org/pdf/2601.15633
- RT-RkNN (2026): https://arxiv.org/pdf/2605.26671
- otk-pyoptix: https://github.com/NVIDIA/otk-pyoptix
- Vulkan ray query guide: https://docs.vulkan.org/guide/latest/extensions/ray_tracing.html
- Coarse-granular RT indexing (2025): https://ieeexplore.ieee.org/document/11112926/
- QPAD dimension reduction: https://arxiv.org/pdf/2504.16335

"""
Recall bake-off: GEMM brute force vs lattice-cell hash vs RT-style
projection filter (CPU reference). This is the decision gate from
docs/RT_RECALL.md: the OptiX port is only justified if the projection-
filter pipeline wins on recall/latency at the target corpus size.

Run:  python benchmarks/benchmark_recall.py [n_rows]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# flat-layout bootstrap
import importlib.util

_root = Path(__file__).resolve().parent.parent
if "e8zip" not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        "e8zip", _root / "__init__.py", submodule_search_locations=[str(_root)]
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["e8zip"] = _mod
    _spec.loader.exec_module(_mod)

from e8zip.core.recall import (  # noqa: E402
    GemmRecall,
    LatticeCellRecall,
    ProjectionFilterRecall,
    measure_recall,
)

try:
    import torch

    from e8zip.core.recall_gpu import TorchGemmRecall, TorchProjectionFilterRecall

    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

try:
    from e8zip.core.rt_optix import OptixProjectionFilterRecall
    import optix
    import cupy
    import cuda.bindings.nvrtc
    HAS_OPTIX = True
except ImportError:
    HAS_OPTIX = False


def _query_in_batches(engine, queries, k, batch_size=None):
    """Keep exact GEMM memory bounded without shrinking the RT batch."""
    if not batch_size or batch_size >= len(queries):
        return engine.query(queries, k)
    idx, d2 = [], []
    for start in range(0, len(queries), batch_size):
        i, d = engine.query(queries[start : start + batch_size], k)
        idx.append(i)
        d2.append(d)
    return np.concatenate(idx), np.concatenate(d2)


def _bench(
    name,
    cls,
    rows,
    queries,
    k,
    exact_idx,
    sync=None,
    query_batch=None,
    repeats=1,
    **kw,
):
    t0 = time.time()
    engine = cls(rows, **kw)
    if sync:
        sync()
    build = time.time() - t0
    _query_in_batches(engine, queries[:2], k, query_batch)  # warm-up
    if sync:
        sync()
    t0 = time.time()
    for _ in range(repeats):
        idx, _ = _query_in_batches(engine, queries, k, query_batch)
    if sync:
        sync()
    q_ms = (time.time() - t0) * 1000 / (len(queries) * repeats)
    r = measure_recall(idx, exact_idx) if exact_idx is not None else 1.0
    print(f"{name:38s} {r:9.3f} {build:9.2f} {q_ms:11.3f}")
    return idx


def main(
    n_rows: int = 50_000,
    n_queries: int = 64,
    k: int = 10,
    gpu_only: bool = False,
    gpu_exact_batch: int | None = None,
    repeats: int = 1,
) -> None:
    rng = np.random.default_rng(0)
    # clustered memory bank: archetypes + noise (recall workloads are
    # clustered; uniform-random rows would be the worst case for every
    # method including RT cores)
    arche = rng.normal(0, 1.5, (max(n_rows // 250, 8), 24))
    rows = arche[rng.integers(0, len(arche), n_rows)] + rng.normal(
        0, 0.15, (n_rows, 24)
    )
    queries = rows[rng.integers(0, n_rows, n_queries)] + rng.normal(
        0, 0.1, (n_queries, 24)
    )

    print(f"recall bake-off: N={n_rows} rows, {n_queries} queries, k={k}\n")
    print(f"{'method':38s} {'recall@k':>9s} {'build s':>9s} {'query ms/q':>11s}")

    if gpu_only:
        if not HAS_CUDA:
            raise RuntimeError("--gpu-only requires a CUDA-enabled PyTorch install")
        # The full RT batch stays large. Only exact GEMM is chunked, because
        # its Q x N distance matrix is intentionally much denser than a ray
        # traversal result.
        exact_idx = _bench(
            "GEMM brute force GPU (exact)",
            TorchGemmRecall,
            rows,
            queries,
            k,
            None,
            sync=torch.cuda.synchronize,
            query_batch=gpu_exact_batch,
            repeats=repeats,
        )
    else:
        exact_idx = _bench(
            "GEMM brute force CPU (exact)",
            GemmRecall,
            rows,
            queries,
            k,
            None,
            repeats=repeats,
        )

    if not gpu_only and n_rows <= 100_000:  # CPU reference paths get slow
        _bench(
            "Leech cell hash (CVP as LSH)",
            LatticeCellRecall,
            rows,
            queries,
            k,
            exact_idx,
            repeats=repeats,
        )
        for radius, votes in ((1.0, 2), (1.5, 2), (1.5, 3)):
            _bench(
                f"RT-style proj filter CPU r={radius} v={votes}/4",
                ProjectionFilterRecall,
                rows,
                queries,
                k,
                exact_idx,
                n_projections=4,
                votes=votes,
                radius=radius,
                repeats=repeats,
            )

    if HAS_CUDA:
        sync = torch.cuda.synchronize
        print(f"-- CUDA ({torch.cuda.get_device_name(0)}) --")
        if not gpu_only:
            _bench(
                "GEMM brute force GPU (exact)",
                TorchGemmRecall,
                rows,
                queries,
                k,
                exact_idx,
                sync=sync,
                query_batch=gpu_exact_batch,
                repeats=repeats,
            )
        for radius, votes in ((1.0, 2), (1.5, 2)):
            _bench(
                f"RT-style proj filter GPU r={radius} v={votes}/4",
                TorchProjectionFilterRecall,
                rows,
                queries,
                k,
                exact_idx,
                sync=sync,
                n_projections=4,
                votes=votes,
                radius=radius,
                repeats=repeats,
            )
    else:
        print("(no CUDA torch: GPU backends skipped)")

    if HAS_OPTIX:
        import cupy as cp

        sync = cp.cuda.Device().synchronize
        print(f"-- OptiX RT Cores ({cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}) --")
        for radius, votes in ((1.0, 2), (1.5, 2)):
            _bench(
                f"OptiX BVH proj filter r={radius} v={votes}/4",
                OptixProjectionFilterRecall,
                rows,
                queries,
                k,
                exact_idx,
                sync=sync,
                n_projections=4,
                votes=votes,
                radius=radius,
                repeats=repeats,
            )
    else:
        print("(no OptiX/Cupy: RT-core backends skipped)")

    print(
        "\n(GPU projection-filter numbers run the grid probe on CUDA cores;"
        "\n an OptiX port replaces only that probe stage with RT-core BVH"
        "\n traversal - vote and re-rank stages are already device code.)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("n_rows", nargs="?", type=int, default=50_000)
    parser.add_argument("n_queries", nargs="?", type=int, default=64)
    parser.add_argument(
        "--gpu-saturate",
        action="store_true",
        help="4,096 concurrent OptiX rays per projection; skip CPU paths",
    )
    parser.add_argument(
        "--gpu-exact-batch",
        type=int,
        default=None,
        help="chunk size for exact GPU GEMM; leaves OptiX queries unchunked",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="repeat each timed query batch to measure sustained throughput",
    )
    args = parser.parse_args()
    if args.gpu_saturate:
        # GPU-only saturation mode: default to a big saturating batch, but
        # honor explicit positional overrides so the GEMM-vs-RT scaling
        # crossover can be probed at large N.
        n_rows = args.n_rows if args.n_rows != 50_000 else 100_000
        n_queries = args.n_queries if args.n_queries != 64 else 4_096
        main(
            n_rows,
            n_queries,
            gpu_only=True,
            gpu_exact_batch=args.gpu_exact_batch or 256,
            repeats=args.repeat,
        )
    else:
        main(
            args.n_rows,
            args.n_queries,
            gpu_exact_batch=args.gpu_exact_batch,
            repeats=args.repeat,
        )

"""
Recall-as-Rays Prototype - candidate architectures for GPU memory recall

Recall = given a query vector, return the k nearest stored rows (memory
sites, embeddings, KV entries). This module implements the three-way
bake-off recommended by the RT-core feasibility study (docs/RT_RECALL.md):

    GemmRecall        exact brute force. One (Q,24)x(24,N) GEMM + top-k.
                      The baseline any accelerated path must beat.
    LatticeCellRecall lattice hashing: rows bucketed by their coarse
                      Leech cell id (our exact CVP used as a spatial
                      hash), probe = query cell + fallback scales.
                      Exploits codec structure, no GPU required.
    ProjectionFilterRecall
                      the RT-core pipeline, CPU reference semantics:
                      m random orthonormal 24->3 projections, per-
                      projection radius search on a 3-D grid (stand-in
                      for an OptiX BVH + degenerate-ray any-hit pass),
                      vote >= v across projections, exact re-rank of
                      survivors. Porting to RT cores = replacing the
                      grid probe with a GAS of spheres + ray launch;
                      Stage boundaries are already GPU-shaped.

All three return exact distances for whatever candidates they surface,
and report recall@k against brute force in the benchmark, so the
tradeoff is measured, not asserted (falsification-first).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from e8zip.core.leech_lattice import batch_nearest_leech_point, leech_index_decompose


def _topk_exact(rows: np.ndarray, q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Exact top-k by squared distance: ||r||^2 - 2 r.q + ||q||^2."""
    d2 = np.sum(rows**2, axis=1)[None, :] - 2.0 * (q @ rows.T)
    k = min(k, rows.shape[0])
    idx = np.argpartition(d2, k - 1, axis=1)[:, :k]
    part = np.take_along_axis(d2, idx, axis=1)
    order = np.argsort(part, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    d2 = np.take_along_axis(part, order, axis=1) + np.sum(q**2, axis=1)[:, None]
    return idx, np.maximum(d2, 0.0)


class GemmRecall:
    """Exact brute-force recall (the honesty baseline)."""

    def __init__(self, rows: np.ndarray) -> None:
        self.rows = np.asarray(rows, dtype=np.float64)

    def query(self, q: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        return _topk_exact(self.rows, np.atleast_2d(np.asarray(q, dtype=np.float64)), k)


def _cell_keys(rows: np.ndarray, cell_scale: float) -> List[Tuple]:
    """Coarse Leech cell id per row: exact CVP as a spatial hash."""
    points, _, _ = batch_nearest_leech_point(rows * cell_scale)
    g_idx, case, z = leech_index_decompose(points)
    return [
        (int(g_idx[i]), int(case[i]), z[i].tobytes()) for i in range(rows.shape[0])
    ]


class LatticeCellRecall:
    """
    Lattice-cell hashing over the codec's own quantizer.

    Rows are bucketed by their nearest coarse-Leech cell at each scale in
    `cell_scales` (finest first). A query probes its own cell per scale,
    accumulating candidates until >= k are found; survivors are re-ranked
    exactly. Coarser scales are the fallback that bounds the miss rate —
    at a coarse enough scale everything shares one cell and recall is 1.
    """

    def __init__(
        self, rows: np.ndarray, cell_scales: Tuple[float, ...] = (0.5, 0.25, 0.125)
    ) -> None:
        self.rows = np.asarray(rows, dtype=np.float64)
        self.cell_scales = cell_scales
        self.tables: List[Dict[Tuple, List[int]]] = []
        for s in cell_scales:
            table: Dict[Tuple, List[int]] = {}
            for i, key in enumerate(_cell_keys(self.rows, s)):
                table.setdefault(key, []).append(i)
            self.tables.append(table)

    def query(self, q: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        q = np.atleast_2d(np.asarray(q, dtype=np.float64))
        nq = q.shape[0]
        idx_out = np.full((nq, k), -1, dtype=np.int64)
        d_out = np.full((nq, k), np.inf)
        keys_by_scale = [_cell_keys(q, s) for s in self.cell_scales]
        for j in range(nq):
            cand: List[int] = []
            for table, keys in zip(self.tables, keys_by_scale):
                cand.extend(table.get(keys[j], ()))
                if len(set(cand)) >= k:
                    break
            cand = sorted(set(cand))
            if not cand:
                continue
            sub = self.rows[cand]
            ci, cd = _topk_exact(sub, q[j : j + 1], k)
            m = ci.shape[1]
            idx_out[j, :m] = np.asarray(cand, dtype=np.int64)[ci[0]]
            d_out[j, :m] = cd[0]
        return idx_out, d_out


class ProjectionFilterRecall:
    """
    RT-core recall pipeline, CPU reference implementation.

    Stage 0: m seeded random orthonormal 24->3 projections; per
             projection, stored rows land in a 3-D uniform grid of cell
             size `radius` (the CPU stand-in for a BVH of spheres of
             radius `radius`).
    Stage 1: a query probes its own grid cell and the 26 neighbors per
             projection (the degenerate-ray any-hit pass).
    Stage 2: rows hit in >= votes projections survive; exact 24-D
             re-rank of survivors.

    Projections are scaled by sqrt(24/3) so projected distances estimate
    true distances; `radius` should be ~ the expected k-NN distance.
    """

    def __init__(
        self,
        rows: np.ndarray,
        n_projections: int = 4,
        votes: int = 2,
        radius: float = 1.0,
        seed: int = 7,
    ) -> None:
        self.rows = np.asarray(rows, dtype=np.float64)
        self.votes = votes
        self.radius = radius
        rng = np.random.default_rng(seed)
        self.projs = []
        for _ in range(n_projections):
            a = rng.normal(size=(24, 3))
            qmat, _ = np.linalg.qr(a)
            self.projs.append(qmat * np.sqrt(24.0 / 3.0))
        self.grids: List[Dict[Tuple[int, int, int], List[int]]] = []
        self.low_rows = []
        for p in self.projs:
            low = self.rows @ p
            self.low_rows.append(low)
            grid: Dict[Tuple[int, int, int], List[int]] = {}
            cells = np.floor(low / radius).astype(np.int64)
            for i in range(low.shape[0]):
                grid.setdefault(tuple(cells[i]), []).append(i)
            self.grids.append(grid)

    def _hits(self, proj_i: int, ql: np.ndarray) -> List[int]:
        grid = self.grids[proj_i]
        base = np.floor(ql / self.radius).astype(np.int64)
        out: List[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    out.extend(
                        grid.get((base[0] + dx, base[1] + dy, base[2] + dz), ())
                    )
        # radius test = the sphere-intersection an RT any-hit shader does
        cand = np.asarray(out, dtype=np.int64)
        if cand.size == 0:
            return []
        low = self.low_rows[proj_i]
        d2 = np.sum((low[cand] - ql) ** 2, axis=1)
        return cand[d2 <= self.radius**2].tolist()

    def query(self, q: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        q = np.atleast_2d(np.asarray(q, dtype=np.float64))
        nq = q.shape[0]
        idx_out = np.full((nq, k), -1, dtype=np.int64)
        d_out = np.full((nq, k), np.inf)
        for j in range(nq):
            counts: Dict[int, int] = {}
            for i, p in enumerate(self.projs):
                for h in self._hits(i, q[j] @ p):
                    counts[h] = counts.get(h, 0) + 1
            cand = sorted(h for h, c in counts.items() if c >= self.votes)
            if not cand:
                continue
            sub = self.rows[cand]
            ci, cd = _topk_exact(sub, q[j : j + 1], k)
            m = ci.shape[1]
            idx_out[j, :m] = np.asarray(cand, dtype=np.int64)[ci[0]]
            d_out[j, :m] = cd[0]
        return idx_out, d_out


@dataclass
class RecallReport:
    name: str
    recall_at_k: float
    build_s: float
    query_ms: float


def measure_recall(
    approx_idx: np.ndarray, exact_idx: np.ndarray
) -> float:
    """Fraction of true top-k ids the candidate method recovered."""
    hits = 0
    for a, e in zip(approx_idx, exact_idx):
        hits += len(set(a.tolist()) & set(e.tolist()))
    return hits / exact_idx.size

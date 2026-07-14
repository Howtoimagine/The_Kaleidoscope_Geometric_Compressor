"""
GPU Recall Backends (PyTorch/CUDA) - recall-as-rays, stages on device

Implements the docs/RT_RECALL.md pipeline with every stage except BVH
traversal on the GPU:

    TorchGemmRecall              exact brute force: one GEMM + topk on
                                 device. Baseline #1 at real scale.
    TorchProjectionFilterRecall  the RT pipeline with the OptiX GAS
                                 replaced by a sorted-cell-key binary
                                 search (torch.searchsorted) — the same
                                 uniform-grid semantics as the CPU
                                 reference, but batched on CUDA cores.
                                 An OptiX port later replaces ONLY the
                                 _hits() stage; the projection, vote,
                                 and re-rank stages are already device
                                 code here.

Requires torch with CUDA; import lazily so the codec stays optional-
dependency clean. Falls back to device="cpu" transparently (useful for
CI), but the numbers that matter come from CUDA.
"""

from typing import List, Tuple

import numpy as np


def _torch():
    try:
        import torch

        return torch
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "GPU recall backends need PyTorch: pip install torch"
        ) from e


def _pick_device(device: str = "auto") -> str:
    torch = _torch()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class TorchGemmRecall:
    """Exact brute-force recall on device: ||r||^2 - 2 r.q + ||q||^2 + topk."""

    def __init__(self, rows: np.ndarray, device: str = "auto") -> None:
        torch = _torch()
        self.torch = torch
        self.device = _pick_device(device)
        self.rows = torch.as_tensor(
            np.asarray(rows, dtype=np.float32), device=self.device
        )
        self.row_sq = (self.rows**2).sum(dim=1)

    def query(self, q: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        qt = torch.as_tensor(
            np.atleast_2d(np.asarray(q, dtype=np.float32)), device=self.device
        )
        k = min(k, self.rows.shape[0])
        with torch.no_grad():
            d2 = self.row_sq[None, :] - 2.0 * (qt @ self.rows.T)
            vals, idx = torch.topk(d2, k, dim=1, largest=False)
            vals = vals + (qt**2).sum(dim=1, keepdim=True)
        return (
            idx.cpu().numpy().astype(np.int64),
            np.maximum(vals.cpu().numpy().astype(np.float64), 0.0),
        )


class TorchProjectionFilterRecall:
    """
    RT-style recall with all stages on device.

    Stage 0: m seeded orthonormal 24->3 projections (identical
             construction to the CPU reference, so results agree).
             Per projection, rows hash to packed 3-D grid-cell keys,
             sorted once (the "BVH build").
    Stage 1: per projection, each query probes its 27 neighboring cells
             via torch.searchsorted range lookups, then the exact
             low-D radius test (the any-hit shader's sphere test).
    Stage 2: votes across projections via sorted unique pair counting;
             survivors re-ranked exactly in 24-D on device.
    """

    _KEY_BITS = 21
    _KEY_OFF = 1 << 20  # supports cell coords in (-2^20, 2^20)

    def __init__(
        self,
        rows: np.ndarray,
        n_projections: int = 4,
        votes: int = 2,
        radius: float = 1.5,
        seed: int = 7,
        device: str = "auto",
        query_chunk: int = 1024,
    ) -> None:
        torch = _torch()
        self.torch = torch
        self.device = _pick_device(device)
        self.votes = votes
        self.radius = radius
        # Bound peak memory: the candidate-pair set scales with the query
        # count, so large batches are processed in chunks (the OptiX path
        # bounds the same growth with a fixed hit buffer).
        self.query_chunk = query_chunk
        rows = np.asarray(rows, dtype=np.float32)
        self.n = rows.shape[0]
        self.rows = torch.as_tensor(rows, device=self.device)
        self.row_sq = (self.rows**2).sum(dim=1)

        rng = np.random.default_rng(seed)
        projs = []
        for _ in range(n_projections):
            a = rng.normal(size=(24, 3))
            qmat, _ = np.linalg.qr(a)
            projs.append(qmat * np.sqrt(24.0 / 3.0))
        self.projs = torch.as_tensor(
            np.stack(projs).astype(np.float32), device=self.device
        )  # (m, 24, 3)

        # "BVH build": per projection, sorted packed cell keys
        with torch.no_grad():
            low = torch.einsum("nd,mdk->mnk", self.rows, self.projs)  # (m,N,3)
            self.low = low
            cells = torch.floor(low / radius).to(torch.int64) + self._KEY_OFF
            keys = (
                (cells[..., 0] << (2 * self._KEY_BITS))
                | (cells[..., 1] << self._KEY_BITS)
                | cells[..., 2]
            )  # (m, N)
            self.sorted_keys, self.perm = torch.sort(keys, dim=1)

        # the 27 neighbor-cell key offsets
        d = torch.tensor([-1, 0, 1], dtype=torch.int64, device=self.device)
        ox, oy, oz = torch.meshgrid(d, d, d, indexing="ij")
        self.key_offsets = (
            (ox.reshape(-1) << (2 * self._KEY_BITS))
            | (oy.reshape(-1) << self._KEY_BITS)
            | oz.reshape(-1)
        )  # (27,)

    def _hits(self, proj_i: int, ql: "object") -> Tuple["object", "object"]:
        """
        Grid probe for one projection (the stage an OptiX GAS + degenerate
        rays replaces). ql: (Q, 3) projected queries.
        Returns (qid, rowid) hit pairs after the exact radius test.
        """
        torch = self.torch
        nq = ql.shape[0]
        base = torch.floor(ql / self.radius).to(torch.int64) + self._KEY_OFF
        qkeys = (
            (base[:, 0] << (2 * self._KEY_BITS))
            | (base[:, 1] << self._KEY_BITS)
            | base[:, 2]
        )  # (Q,)
        probe = (qkeys[:, None] + self.key_offsets[None, :]).reshape(-1)  # (Q*27,)

        sk = self.sorted_keys[proj_i].contiguous()
        lo = torch.searchsorted(sk, probe, right=False)
        hi = torch.searchsorted(sk, probe, right=True)
        counts = hi - lo  # (Q*27,)
        total = int(counts.sum())
        if total == 0:
            empty = torch.empty(0, dtype=torch.int64, device=self.device)
            return empty, empty

        # flatten variable-length ranges: standard repeat_interleave trick
        starts = torch.repeat_interleave(lo, counts)
        offs = torch.arange(total, device=self.device) - torch.repeat_interleave(
            torch.cumsum(counts, 0) - counts, counts
        )
        sorted_pos = starts + offs
        rowid = self.perm[proj_i][sorted_pos]
        qid = torch.repeat_interleave(
            torch.arange(nq, device=self.device).repeat_interleave(27), counts
        )

        # exact low-D radius test = the any-hit sphere intersection
        d2 = ((self.low[proj_i][rowid] - ql[qid]) ** 2).sum(dim=1)
        keep = d2 <= self.radius**2
        return qid[keep], rowid[keep]

    def query(self, q: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        qnp = np.atleast_2d(np.asarray(q, dtype=np.float32))
        nq = qnp.shape[0]
        if nq <= self.query_chunk:
            return self._query_chunk(qnp, k)
        idx = np.empty((nq, k), dtype=np.int64)
        dist = np.empty((nq, k), dtype=np.float64)
        for s in range(0, nq, self.query_chunk):
            e = min(nq, s + self.query_chunk)
            idx[s:e], dist[s:e] = self._query_chunk(qnp[s:e], k)
        return idx, dist

    def _query_chunk(self, qnp: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        qt = torch.as_tensor(qnp, device=self.device)
        nq = qt.shape[0]

        with torch.no_grad():
            qlow = torch.einsum("qd,mdk->mqk", qt, self.projs)  # (m, Q, 3)

            pair_list: List = []
            for i in range(self.projs.shape[0]):
                qid, rowid = self._hits(i, qlow[i])
                if qid.numel():
                    pair_list.append(qid * self.n + rowid)
            idx_out = np.full((nq, k), -1, dtype=np.int64)
            d_out = np.full((nq, k), np.inf)
            if not pair_list:
                return idx_out, d_out

            # vote: count identical (qid,rowid) pairs across projections
            pairs = torch.cat(pair_list)
            pairs, counts = torch.unique(pairs, return_counts=True)
            pairs = pairs[counts >= self.votes]
            if not pairs.numel():
                return idx_out, d_out
            qid = pairs // self.n
            rowid = pairs % self.n

            # exact 24-D re-rank of survivors
            d2 = (
                self.row_sq[rowid]
                - 2.0 * (self.rows[rowid] * qt[qid]).sum(dim=1)
                + (qt[qid] ** 2).sum(dim=1)
            ).clamp_min(0)

            # per-query top-k: sort by d2 then stable-sort by qid, so each
            # query's segment comes out distance-ordered
            order = torch.argsort(d2)
            order = order[torch.argsort(qid[order], stable=True)]
            qid, rowid, d2 = qid[order], rowid[order], d2[order]
            qid_np = qid.cpu().numpy()
            rowid_np = rowid.cpu().numpy()
            d2_np = d2.cpu().numpy().astype(np.float64)

        bounds = np.searchsorted(qid_np, np.arange(nq + 1))
        for j in range(nq):
            s, e = bounds[j], min(bounds[j + 1], bounds[j] + k)
            m = e - s
            idx_out[j, :m] = rowid_np[s:e]
            d_out[j, :m] = d2_np[s:e]
        return idx_out, d_out

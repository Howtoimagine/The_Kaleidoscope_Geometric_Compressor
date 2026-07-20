"""
Adaptive (learned) Leech VQ - and the MEASURED NEGATIVE that killed it.

Motivation: the FAISS benchmark (benchmark_faiss_vq.py) showed KGC's
fixed Leech lattice loses to LEARNED quantizers (PQ/OPQ) at aggressive
rates. OPQ's edge over plain PQ is almost entirely one learned
orthogonal ROTATION applied before quantizing (Ge et al. 2013), stored
once and amortized. The hypothesis: do the same for the Leech lattice -
learn an orthogonal R that rotates the data into the frame the lattice
quantizes best (the "entanglement-deformed lattice"; cf. the glass-
network's own `entanglement_metric.py`, "the lattice is a rubber sheet,
the yarn is decoration"). This module implements exactly that: non-
parametric OPQ (alternate lattice-quantize / orthogonal-Procrustes),
plus an optional diagonal variance allocation (the dimensional cousin of
curvature / importance weighting).

MEASURED RESULT: it does not work, and the reason is fundamental.

    real 384-D embeddings, scale 0.75:  plain 3.044e-4 vs learned 3.041e-4  (0.1%)
    DELIBERATELY anisotropic data (30 dims stretched 8x):  0.0%

Learned rotation gives ~0% even when strong anisotropy is present for it
to exploit. Why: the Leech lattice is near-ISOTROPIC - its Voronoi cell
is close to spherical - so rotating the data does not change how well the
lattice tiles it. OPQ's rotation helps PQ precisely because PQ is
ANISOTROPIC (independent axis-aligned subspace codebooks whose variance
balance the rotation fixes); the lattice has no such preferred axes to
align to. The lattice's isotropy is its strength (it beats scalar
quantization for exactly this reason) and simultaneously the reason the
OPQ trick cannot transfer.

CONCLUSION (kept as a committed negative, in the house style of the
glass-network's own falsifiers): the gap from fixed-lattice VQ to learned
PQ is STRUCTURAL, not a frame-alignment problem. Closing it requires a
data-adaptive CODEBOOK (learned centroids / residual codebooks) - at
which point one has left the lattice and is doing PQ/AQLM. The lever that
DOES move a lattice quantizer is anisotropic SCALING, not rotation:
per-channel activation-aware scaling gave a measured +4% on weights (the
Resonant kernel, benchmark_resonant.py), because it changes per-axis
RESOLUTION, which the lattice is sensitive to - unlike rotation, which it
is invariant to. Frame-deformation is the wrong knob; resolution
allocation is the right one, and only where an importance signal exists.

The code below is correct and reusable (verified round-trip); it is
retained as the instrument that produced the negative, not as a
recommended path.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from e8zip.core.leech_lattice import batch_nearest_leech_point


@dataclass
class AdaptiveLeechVQ:
    """Learned-rotation Leech vector quantizer (OPQ-style, lattice codebook)."""

    dim: int
    scale: float = 1.0
    balance_variance: bool = False
    R: Optional[np.ndarray] = None  # (dim, dim) orthogonal
    d: Optional[np.ndarray] = None  # (dim,) diagonal allocation (>0)
    sigma: float = 1.0

    def _quantize_rotated(self, Xr: np.ndarray) -> np.ndarray:
        """Leech-snap rows of an already-rotated/scaled matrix; return rotated-frame recon."""
        n = Xr.shape[0]
        xb = (Xr / self.sigma * self.scale).reshape(-1, 24)
        pts, _, _ = batch_nearest_leech_point(xb)
        return pts.reshape(n, self.dim) / self.scale * self.sigma

    def fit(self, X: np.ndarray, n_iter: int = 6, seed: int = 0) -> "AdaptiveLeechVQ":
        """
        Learn R (and optional diagonal d) on a training sample X (N, dim)
        via non-parametric OPQ: alternate lattice-quantize / Procrustes.
        """
        X = np.asarray(X, dtype=np.float64)
        assert X.shape[1] == self.dim and self.dim % 24 == 0

        # PCA initialization: rotate to principal axes (good starting frame).
        cov = X.T @ X / len(X)
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        R = evecs[:, order].T.copy()  # rows = principal directions
        evals = np.clip(evals[order], 1e-12, None)

        # Diagonal allocation: whiten toward equal per-axis variance so every
        # 24-D block carries comparable energy (the importance/curvature knob).
        if self.balance_variance:
            d = 1.0 / np.sqrt(evals)
            d = d / np.exp(np.mean(np.log(d)))  # unit geometric mean (volume-preserving)
        else:
            d = np.ones(self.dim)

        for _ in range(max(1, n_iter)):
            Xr = (X @ R.T) * d  # rotate + allocate
            self.sigma = float(Xr.std()) or 1.0
            Z = self._quantize_rotated(Xr)  # rotated-frame reconstruction
            # Procrustes on the UN-allocated frame: min_R ||X - (Z/d) R||_F.
            Zc = Z / d
            M = Zc.T @ X  # (dim, dim)
            U, _, Vt = np.linalg.svd(M)
            R = (U @ Vt).T  # orthogonal update
        self.R, self.d = R, d
        # final sigma at the learned frame
        Xr = (X @ self.R.T) * self.d
        self.sigma = float(Xr.std()) or 1.0
        return self

    def quantize(self, X: np.ndarray):
        """Return (reconstructed_X, bits_per_vector) for the learned frame."""
        from e8zip.core.leech_lattice import leech_index_decompose

        X = np.asarray(X, dtype=np.float64)
        Xr = (X @ self.R.T) * self.d
        n = X.shape[0]
        xb = (Xr / self.sigma * self.scale).reshape(-1, 24)
        pts, _, _ = batch_nearest_leech_point(xb)
        Zr = pts.reshape(n, self.dim) / self.scale * self.sigma
        X_hat = (Zr / self.d) @ self.R  # unrotate

        # rate: empirical entropy of the index streams (amortize R,d: O(1)/vec)
        g, c, z = leech_index_decompose(pts)

        def H(a):
            a = np.asarray(a).ravel()
            _, cnt = np.unique(a, return_counts=True)
            p = cnt / cnt.sum()
            return float(-(p * np.log2(p)).sum())

        zz = lambda v: np.where(v >= 0, 2 * v, -2 * v - 1)
        bpb = H(g) + H(c) + sum(H(zz(z[:, j])) for j in range(24))
        return X_hat, bpb * (self.dim // 24)

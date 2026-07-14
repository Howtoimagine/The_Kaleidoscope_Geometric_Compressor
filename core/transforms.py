"""
Incoherence Transforms - Shared Preprocessing for Lattice Quantization

QuIP#'s central insight: lattice quantizers work best on incoherent
(spread-out, sub-Gaussian) data. Raw weight matrices are coherent -
individual coordinates carry structured, concentrated energy (outlier
channels, correlated directions). Two orthogonal transforms fix this,
each a different point on the speed/strength trade-off:

    hadamard  O(n log n), structured signs. QuIP#'s choice: fast enough
              to run on every weight matrix in a full-size model.
    rotation  O(n^2), a dense random orthogonal matrix (QR of a seeded
              Gaussian). Stronger decorrelation - mixes every input
              dimension into every output dimension - at quadratic
              cost. Borrowed from the Glass Network's turbo_leech
              harness (model_cookbook/turbo_leech.py), where it is the
              only transform tested against real model weights.

Both are exact orthogonal maps (norm-preserving, exactly invertible),
so either can sit in front of any lattice quantizer without changing
the rate-distortion math - only the achieved incoherence differs.
core/kgc.py and core/klc.py both dispatch through `forward`/`inverse`
here so the two codecs can be benchmarked on identical transforms.
"""

from functools import lru_cache
from typing import Literal

import numpy as np

TransformKind = Literal["hadamard", "rotation", "none"]


def _fwht(a: np.ndarray) -> np.ndarray:
    """In-place-style fast Walsh-Hadamard transform along axis 1 (power-of-2 dim)."""
    a = a.copy()
    h = 1
    n = a.shape[1]
    while h < n:
        for i in range(0, n, h * 2):
            x = a[:, i : i + h].copy()
            y = a[:, i + h : i + 2 * h].copy()
            a[:, i : i + h] = x + y
            a[:, i + h : i + 2 * h] = x - y
        h *= 2
    return a / np.sqrt(n)


def randomized_hadamard(a: np.ndarray, seed: int, inverse: bool = False) -> np.ndarray:
    """Randomized Hadamard transform (QuIP#-style incoherence pass). O(n log n)."""
    n = a.shape[1]
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=n)
    if inverse:
        return _fwht(a) * signs  # H^-1 = H (orthonormal); undo signs after
    return _fwht(a * signs)


@lru_cache(maxsize=16)
def orthogonal_matrix(dim: int, seed: int) -> np.ndarray:
    """
    Deterministic dense random orthogonal matrix (QR of a seeded Gaussian).

    Ported from the Glass Network's model_cookbook/turbo_leech.py. Mixes
    every dimension into every other - stronger incoherence than
    block-Hadamard, at O(dim^2) per application instead of O(dim log dim).
    """
    if dim <= 0:
        raise ValueError("dim must be positive")
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((dim, dim))
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return (q * signs).astype(np.float64, copy=False)


def dense_rotation(a: np.ndarray, seed: int, inverse: bool = False) -> np.ndarray:
    """Dense orthogonal rotation. a: (rows, dim). O(rows * dim^2)."""
    r = orthogonal_matrix(a.shape[1], seed)
    return a @ r.T if inverse else a @ r


def forward(a: np.ndarray, kind: TransformKind, seed: int) -> np.ndarray:
    """Apply the named incoherence transform to rows of `a`."""
    if kind == "hadamard":
        return randomized_hadamard(a, seed)
    if kind == "rotation":
        return dense_rotation(a, seed)
    if kind == "none":
        return a
    raise ValueError(f"unknown transform kind {kind!r}")


def inverse(a: np.ndarray, kind: TransformKind, seed: int) -> np.ndarray:
    """Invert the named incoherence transform."""
    if kind == "hadamard":
        return randomized_hadamard(a, seed, inverse=True)
    if kind == "rotation":
        return dense_rotation(a, seed, inverse=True)
    if kind == "none":
        return a
    raise ValueError(f"unknown transform kind {kind!r}")

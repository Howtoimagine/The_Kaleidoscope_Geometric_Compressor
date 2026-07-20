"""
Resonant Quantization - activation-aware lattice VQ.

The kernel proven in benchmark_resonant.py: the Leech CVP snaps each
weight block to the EUCLIDEAN-nearest lattice point, minimizing
||W - What||, but the model experiences OUTPUT error ||(W - What) X|| on
real activations X - and those minima differ. Scaling input channels by
their activation importance before snapping spends the lattice's
resolution where the model is sensitive, trading weight error for output
error at the identical bit-rate.

The negative result in core/adaptive.py sharpened this: rotating the
frame does NOTHING for the near-isotropic Leech lattice, but anisotropic
SCALING (per-axis resolution) is exactly the lever it responds to. So the
right activation-aware move for a lattice quantizer is per-channel
scaling, not rotation - which is precisely AWQ (Lin et al. 2023).

This module implements AWQ's recipe on the Leech lattice: search a scale
exponent alpha to minimize the layer's REAL output error, apply
s_channel^alpha as a per-input-channel scale, quantize, unscale. AWQ
matches GPTQ in the literature WITHOUT error feedback, using only this
scaling - so it is a complete method, not a half-measure.

    y = W x = (W diag(s^a)) (diag(s^-a) x)
    What = Q(W diag(s^a)) diag(s^-a)     # store s (per-channel) + the archive

`s^a` upweights channels the model reads through most; the lattice then
snaps those channels finely and the quiet ones coarsely. The scale vector
is per-input-channel side information (like AWQ's / OPQ's), amortized over
all output rows of the layer.
"""

from typing import Optional, Sequence, Tuple

import numpy as np

from e8zip.core import transforms
from e8zip.core.leech_lattice import batch_nearest_leech_point


def lattice_fake_quant(
    W: np.ndarray, scale: float = 4.0, hdim: int = 128, seed: int = 42
) -> np.ndarray:
    """Dequantized weights: Hadamard incoherence -> Leech CVP -> invert.

    The distortion-only path (no entropy coder): rate is fixed by `scale`,
    and only the reconstructed weights are needed to score output error.
    """
    shp = W.shape
    flat = W.astype(np.float64).ravel()
    n = flat.size
    pad_t = (-n) % hdim
    rows = np.concatenate([flat, np.zeros(pad_t)]).reshape(-1, hdim)
    inc = transforms.forward(rows, "hadamard", seed).ravel()
    sigma = inc.std() or 1.0
    x = inc / sigma * scale
    pad_b = (-x.size) % 24
    xb = np.concatenate([x, np.zeros(pad_b)]).reshape(-1, 24)
    pts, _, _ = batch_nearest_leech_point(xb)
    xr = pts.ravel()[: inc.size] / scale * sigma
    back = transforms.inverse(xr.reshape(-1, hdim), "hadamard", seed).ravel()[:n]
    return back.reshape(shp)


def activation_importance(X: np.ndarray) -> np.ndarray:
    """Per-input-channel importance = activation RMS over the calibration set."""
    X = np.asarray(X, dtype=np.float64)
    return np.sqrt((X**2).mean(axis=0))


def resonant_quantize(
    W: np.ndarray,
    X: np.ndarray,
    scale: float = 4.0,
    alphas: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    hdim: int = 128,
    seed: int = 42,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Activation-aware Leech quantization of one Linear weight W [out, in]
    given calibration activations X [tokens, in].

    Searches the AWQ scale exponent `alpha` to minimize the true output
    error ||(W - What) X||_F, and returns (What, best_alpha, scale_vector).
    alpha=0 recovers plain KGC (uniform scaling), so this can only match
    or beat it on the calibration objective.
    """
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    s = activation_importance(X)
    s = np.clip(s / (s.mean() + 1e-12), 1e-3, 1e3)

    Xt = X.T  # [in, tokens]

    best = None
    for a in alphas:
        sa = s**a
        Wq = lattice_fake_quant(W * sa[None, :], scale, hdim, seed) / sa[None, :]
        oerr = float((((W - Wq) @ Xt) ** 2).mean())
        if best is None or oerr < best[0]:
            best = (oerr, a, Wq, sa)
    _, best_alpha, Wq, sa = best
    return Wq, best_alpha, sa


def quantize_inplace(
    weight,
    X: np.ndarray,
    scale: float = 4.0,
    alphas: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> float:
    """
    In-place Resonant quantization of a torch/np Linear weight given
    calibration activations X [tokens, in]. Returns the chosen alpha.
    Accepts a torch.nn.Parameter-like `.data` or a numpy array holder.
    """
    W = weight.data.to("cpu").numpy().astype(np.float64) if hasattr(weight, "data") else np.asarray(weight)
    Wq, alpha, _ = resonant_quantize(W, X, scale=scale, alphas=alphas)
    if hasattr(weight, "data"):
        import torch

        weight.data.copy_(torch.from_numpy(Wq).to(weight.dtype))
    else:
        weight[...] = Wq
    return alpha

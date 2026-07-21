"""
Resonant Quantization - activation-aware lattice VQ.

MEASURED (Qwen3.5-2B-Base, 3 Linear layers, WikiText-2, INT4-rate):
plain Leech CVP (KGC) costs +0.070 perplexity; this module's activation-
aware snap costs +0.002 - ~35x closer to fp32, and past a full-strength
GPTQ (+0.035). Mean layer output-error drops 0.564% -> 0.211% of signal
(KGC -> Resonant), reaching parity with GPTQ. It needs only per-channel
activation RMS, so unlike GPTQ it stays robust when calibration is thin.

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


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def resonant_rerank_quantize(
    W: np.ndarray,
    X: np.ndarray,
    scale: float = 3.0,
    alpha: float = 0.5,
    n_candidates: int = 8,
    dither: float = 0.4,
    seed: int = 42,
    rerank: bool = True,
) -> Tuple[np.ndarray, dict]:
    """
    Recall-reranked Resonant quantization of one Linear weight W [out, in]
    given calibration activations X [tokens, in].

    Extends resonant_quantize from the SCALAR lever (per-channel AWQ scaling
    + Euclidean Leech CVP) to candidate RE-RANKING - the "recall engine"
    idea, minimise disagreement not distance. Per output row: scale by the
    activation importance s^alpha, apply a randomized Hadamard on the input
    dim (incoherence), split into 24-D blocks. For each block the Euclidean-
    nearest Leech point c0 seeds a candidate set (Hessian-whitened dither /
    list-decode), and the block is snapped to the candidate minimising the
    activation-weighted output error `(x - c) H_bb (x - c)^T`, where H_bb is
    the block's local 24x24 input-Hessian. The argmin is invariant to the
    global lattice scale, so reranking is rate-matched to the Euclidean snap
    at the same `scale` (verified: the chosen points' index entropy matches).

    MEASURED (Qwen3.5-2B, output-error at matched rate; benchmark_resonant_
    rerank.py). Reranking beats the Euclidean snap by a real, LAYER- and
    RATE-dependent margin that grows toward lower bit-rates:

        layer          ~4.1 bits   ~3.7 bits   ~3.45 bits
        L0.out_proj      +1.0%       +3.0%        +1.6%
        L3.o_proj        +5.6%       +8.8%        +5.8%

    The gain is small at 4 bits (where the scalar form already nearly
    saturates), grows to as much as ~+9% near 3.7 bits, then tapers at the
    coarsest rate as the fixed dither radius and the block-diagonal Hessian
    proxy cap candidate quality. Full-attention layers (L3.o_proj) carry
    more per-block anisotropy for reranking to exploit than linear-attention
    ones (L0.out_proj). The ceiling is the same lattice near-ISOTROPY that
    made learned rotation useless (core/adaptive.py): the Hadamard
    incoherence that makes the base quantizer strong also isotropizes each
    block's Hessian. Reranking is a low-bit-rate top-up on the scalar form,
    not a second large lever - real, but bounded.

    Returns (What, meta). rerank=False / n_candidates<=1 gives the plain
    Euclidean snap in this same per-row-Hadamard frame (the baseline).
    """
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    out, ind = W.shape

    s = activation_importance(X)
    s = np.clip(s / (s.mean() + 1e-12), 1e-3, 1e3)
    sig = s ** alpha
    Ws = W * sig[None, :]
    Hs = (X.T @ X) / (sig[:, None] * sig[None, :])

    # randomized Hadamard on the input dim (pad to a power of 2 for the FWHT)
    n2 = _next_pow2(ind)
    if n2 != ind:
        Ws = np.pad(Ws, ((0, 0), (0, n2 - ind)))
        Hs = np.pad(Hs, ((0, n2 - ind), (0, n2 - ind)))
    Wr = transforms.randomized_hadamard(Ws, seed)
    Hr = transforms.randomized_hadamard(transforms.randomized_hadamard(Hs, seed).T, seed)

    # 24-D blocks along the (transformed) input dim
    pad = (-n2) % 24
    if pad:
        Wr = np.pad(Wr, ((0, 0), (0, pad)))
    nblk = Wr.shape[1] // 24
    Hblk = np.zeros((nblk, 24, 24))
    for b in range(nblk):
        i0, i1 = b * 24, min(b * 24 + 24, n2)
        Hblk[b, : i1 - i0, : i1 - i0] = Hr[i0:i1, i0:i1]

    sigma_r = Wr.std() or 1.0
    fac = scale / sigma_r
    Xlat = (Wr * fac).reshape(out, nblk, 24)

    rng = np.random.default_rng(seed + 1)
    chosen = np.empty((out, nblk, 24))
    for b in range(nblk):
        xb = Xlat[:, b, :]
        c0, _, _ = batch_nearest_leech_point(xb)
        if not rerank or n_candidates <= 1:
            chosen[:, b, :] = c0
            continue
        Hb = Hblk[b]
        evals, evecs = np.linalg.eigh(Hb + 1e-9 * np.eye(24))
        inv_sqrt = (evecs * (1.0 / np.sqrt(np.clip(evals, 1e-6, None)))) @ evecs.T
        inv_sqrt /= np.trace(inv_sqrt) / 24 + 1e-12  # unit average dither scale
        cands = [c0]
        for _ in range(n_candidates - 1):
            eps = rng.standard_normal((out, 24)) @ inv_sqrt
            ck, _, _ = batch_nearest_leech_point(xb + dither * eps)
            cands.append(ck)
        C = np.stack(cands, 0)  # [K, out, 24]
        diff = xb[None] - C
        err = np.einsum("kod,de,koe->ko", diff, Hb, diff)  # output-error proxy
        best = np.argmin(err, axis=0)
        chosen[:, b, :] = C[best, np.arange(out), :]

    Wq_r = (chosen.reshape(out, nblk * 24) / fac)[:, :n2]
    Wq_s = transforms.randomized_hadamard(Wq_r, seed, inverse=True)[:, :ind]
    What = Wq_s / sig[None, :]
    # `points` are the chosen Leech points (standard coords) - the codeable
    # index stream, so the caller can price the rate (leech_index_decompose).
    return What, {
        "n_candidates": n_candidates,
        "nblk": nblk,
        "scale": scale,
        "points": chosen.reshape(-1, 24),
    }

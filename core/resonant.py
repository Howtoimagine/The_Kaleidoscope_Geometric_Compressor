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
    coupling: bool = True,
    sweeps: int = 3,
) -> Tuple[np.ndarray, dict]:
    """
    Recall-reranked Resonant quantization of one Linear weight W [out, in]
    given calibration activations X [tokens, in] - "GPTQ on the Leech lattice".

    Per output row: scale by the activation importance s^alpha, apply a
    randomized Hadamard on the input dim (incoherence), split into 24-D
    blocks. Each block's Euclidean-nearest Leech point c0 seeds a candidate
    set (Hessian-whitened dither / list-decode - the "recall engine" move).
    The block is then snapped to the candidate that minimises the model's
    OUTPUT error, not the weight distance.

    Two rerank objectives:

    * `coupling=False` - block-diagonal proxy `(x-c) H_bb (x-c)^T` using only
      the block's local 24x24 input-Hessian. Cheap, order-independent, but
      WEAK: it ignores inter-block error propagation and even goes negative at
      the coarsest rates (measured -2.7% at ~3.4 bits).

    * `coupling=True` (default) - GAUSS-SEIDEL coordinate descent on the EXACT
      output error `r^T H_r r` (r = x - chosen, H_r the full transformed
      Hessian; exact because the Hadamard is orthogonal). Blocks are updated
      in sequence; the full gradient `G = r @ H_r` is maintained incrementally
      so each block sees the fresh coupling `g_b = (G - r_b H_bb)_b` from the
      blocks already updated, and snaps to argmin `(x-c) H_bb (x-c) + 2(x-c).g_b`.
      This is GPTQ's error feedback done across LATTICE blocks with candidate
      enumeration instead of scalar rounding. Because c0 is always a candidate,
      the true output error is monotone non-increasing (a Jacobi / parallel
      update instead DIVERGES - it must be sequential).

    Both are rate-matched to the Euclidean snap (they pick among the same
    candidate pool; the chosen points' Leech index entropy is unchanged -
    verified in benchmark_resonant_rerank.py).

    MEASURED (Qwen3.5-2B L3.o_proj, output-error reduction vs the Euclidean
    snap, rate-matched):

        scale/bits     block-diag    coupling (3 sweeps)
        ~4.11 bits       +7.6%            +62.9%
        ~3.70 bits       +9.6%            +63.7%
        ~3.44 bits       -2.7%            +59.6%

    The coupling objective is the real lever: ~+60% output-MSE reduction at
    every rate, an order of magnitude past the block-diagonal proxy, and it
    holds where the proxy collapses. `sweeps=1` already gets ~+58%; extra
    sweeps add a few percent.

    Returns (What, meta) with meta["points"] the chosen Leech points (the
    codeable index stream, for pricing the rate). rerank=False /
    n_candidates<=1 gives the plain Euclidean snap (the baseline).
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
        Hr = np.pad(Hr, ((0, pad), (0, pad)))
    nblk = Wr.shape[1] // 24
    Hblk = np.stack([Hr[b * 24:(b + 1) * 24, b * 24:(b + 1) * 24] for b in range(nblk)])

    sigma_r = Wr.std() or 1.0
    fac = scale / sigma_r
    Xlat = (Wr * fac).reshape(out, nblk, 24)
    idx = np.arange(out)

    # 1) candidate generation: Euclidean seed + Hessian-whitened dither
    rng = np.random.default_rng(seed + 1)
    Cand = []
    seedpts = np.empty((out, nblk, 24))
    do_rerank = rerank and n_candidates > 1
    for b in range(nblk):
        xb = Xlat[:, b, :]
        c0, _, _ = batch_nearest_leech_point(xb)
        seedpts[:, b, :] = c0
        if not do_rerank:
            Cand.append(c0[None])
            continue
        ev, V = np.linalg.eigh(Hblk[b] + 1e-9 * np.eye(24))
        inv_sqrt = (V * (1.0 / np.sqrt(np.clip(ev, 1e-6, None)))) @ V.T
        inv_sqrt /= np.trace(inv_sqrt) / 24 + 1e-12  # unit average dither scale
        cands = [c0]
        for _ in range(n_candidates - 1):
            eps = rng.standard_normal((out, 24)) @ inv_sqrt
            ck, _, _ = batch_nearest_leech_point(xb + dither * eps)
            cands.append(ck)
        Cand.append(np.stack(cands, 0))  # [K, out, 24]

    # 2) rerank among candidates
    chosen = seedpts.copy()
    if do_rerank and not coupling:
        for b in range(nblk):
            diff = Xlat[:, b, :][None] - Cand[b]
            obj = np.einsum("kod,de,koe->ko", diff, Hblk[b], diff)
            chosen[:, b, :] = Cand[b][np.argmin(obj, axis=0), idx, :]
    elif do_rerank:
        # Gauss-Seidel coordinate descent on the exact output error r^T H_r r
        r = (Xlat - chosen).reshape(out, -1)
        G = r @ Hr
        for _sw in range(max(1, sweeps)):
            for b in range(nblk):
                c0i, c1i = b * 24, b * 24 + 24
                xb, Cb, Hb = Xlat[:, b, :], Cand[b], Hblk[b]
                r_b = r[:, c0i:c1i]
                g_b = G[:, c0i:c1i] - r_b @ Hb          # coupling from other blocks
                diff = xb[None] - Cb
                obj = np.einsum("kod,de,koe->ko", diff, Hb, diff) \
                    + 2.0 * np.einsum("kod,od->ko", diff, g_b)
                new_c = Cb[np.argmin(obj, axis=0), idx, :]
                new_r_b = xb - new_c
                chosen[:, b, :] = new_c
                G += (new_r_b - r_b) @ Hr[c0i:c1i, :]   # incremental gradient
                r[:, c0i:c1i] = new_r_b

    Wq_r = (chosen.reshape(out, nblk * 24) / fac)[:, :n2]
    Wq_s = transforms.randomized_hadamard(Wq_r, seed, inverse=True)[:, :ind]
    What = Wq_s / sig[None, :]
    return What, {
        "n_candidates": n_candidates,
        "nblk": nblk,
        "scale": scale,
        "coupling": bool(do_rerank and coupling),
        "sweeps": sweeps,
        "points": chosen.reshape(-1, 24),
    }

"""
Recall-reranked Resonant quantization - layer-level output-error benchmark.

Compares three snaps in the identical per-row-Hadamard frame, all at the same
`scale` (rate), on real Qwen3.5-2B layers:

  euclid     - Euclidean-nearest Leech point (the scalar-Resonant baseline)
  block-diag - rerank candidates by the block-local proxy (x-c) H_bb (x-c)
  coupling   - rerank by GAUSS-SEIDEL coordinate descent on the EXACT output
               error r^T H_r r (GPTQ error feedback across lattice blocks)

Reports the TRUE output error ||(W-What)X||^2/||WX||^2 and the Leech index
entropy (bits/weight) - the comparison is only honest if the rate matches, so
both are printed.

Measured (96-row subset, L3.o_proj; reduction vs the Euclidean snap):

    scale/bits     block-diag    coupling (3 sweeps)
    ~4.11 bits       +7.6%            +62.9%
    ~3.70 bits       +9.6%            +63.7%
    ~3.44 bits       -2.7%            +59.6%

The block-diagonal proxy is a weak, non-robust top-up (it even goes negative
at the coarsest rate). The coupling objective is the real lever: ~+60%
output-MSE reduction at every rate, RATE-MATCHED (same bits), an order of
magnitude past the proxy - and it holds where the proxy collapses, because it
optimises the exact output error rather than a per-block approximation. It is
"GPTQ on the Leech lattice": sequential error feedback, but snapping to the
best of K lattice candidates instead of scalar rounding.

WARNING - this measures CALIBRATION output-error, and the coupling win does
NOT transfer. End-to-end 3-bit perplexity (held-out WikiText-2) OVERFITS:
scalar +0.025 < coupling +0.056 < KGC +0.125 < GPTQ +0.311. Coupling has the
best calibration fit yet a worse held-out loss than the plain scalar snap -
it fits the rank-deficient 1024-token Hessian like GPTQ does. The honest
3-bit winner is scalar-Resonant (rerank=False); damped coupling (regularised
null space) is the path to making the calibration gain actually transfer.
A big calibration-proxy win can be mostly overfitting.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent
_root = _here.parent
if "e8zip" not in sys.modules:
    import importlib.util

    _spec = importlib.util.spec_from_file_location(
        "e8zip", _root / "__init__.py", submodule_search_locations=[str(_root)]
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["e8zip"] = _mod
    _spec.loader.exec_module(_mod)

sys.path.insert(0, str(_here))
from e8zip.core.leech_lattice import leech_index_decompose  # noqa: E402
from e8zip.core.resonant import resonant_rerank_quantize  # noqa: E402


def _entropy(a):
    a = np.asarray(a).ravel()
    _, cnt = np.unique(a, return_counts=True)
    p = cnt / cnt.sum()
    return float(-(p * np.log2(p)).sum())


def bits_per_weight(points, n_weights):
    g, c, z = leech_index_decompose(points)
    zz = lambda v: np.where(v >= 0, 2 * v, -2 * v - 1)
    per_block = _entropy(g) + _entropy(c) + sum(_entropy(zz(z[:, j])) for j in range(24))
    return per_block * points.shape[0] / n_weights


def output_relmse(W, X, What):
    Xt = X.T
    return float((((W - What) @ Xt) ** 2).mean()) / float(((W @ Xt) ** 2).mean())


def sweep(W, X, scales, n_candidates=8, dither=0.4, sweeps=3, seed=42):
    nw = W.size
    for scale in scales:
        t0 = time.time()
        We, me = resonant_rerank_quantize(W, X, scale=scale, rerank=False, seed=seed)
        Wb, mb = resonant_rerank_quantize(
            W, X, scale=scale, n_candidates=n_candidates, dither=dither,
            coupling=False, seed=seed,
        )
        Wc, mc = resonant_rerank_quantize(
            W, X, scale=scale, n_candidates=n_candidates, dither=dither,
            coupling=True, sweeps=sweeps, seed=seed,
        )
        eu = output_relmse(W, X, We)
        be = bits_per_weight(me["points"], nw)
        gain = lambda v: 100 * (1 - v / eu)
        print(f"  scale={scale:.2f}  euclid={eu:.4e} ({be:.2f}b)", flush=True)
        for name, Wq, mq in [("block-diag", Wb, mb), ("coupling", Wc, mc)]:
            v = output_relmse(W, X, Wq)
            print(f"      {name:11s} {v:.4e} ({bits_per_weight(mq['points'], nw):.2f}b)  "
                  f"gain={gain(v):+.1f}%", flush=True)
        print(f"      [{time.time()-t0:.0f}s]", flush=True)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from benchmark_resonant_perplexity import capture_activations, layer_handles

    rows = int(os.environ.get("RERANK_ROWS", "160"))
    scales = [float(s) for s in os.environ.get("RERANK_SCALES", "4.0,3.0,2.5").split(",")]

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B-Base")
    import torch

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-2B-Base", dtype=torch.float32).eval()
    from benchmark_llm_perplexity import load_wikitext2

    train, _ = load_wikitext2()
    mods = layer_handles(model)
    X = capture_activations(model, tok, mods, train["text"])

    for name, m in mods.items():
        W = m.weight.data.numpy().astype(np.float64)[:rows]
        print(f"\n=== {name}  W[:{rows}]={W.shape} ===", flush=True)
        sweep(W, X[name], scales)


if __name__ == "__main__":
    main()

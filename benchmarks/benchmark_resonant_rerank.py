"""
Recall-reranked Resonant quantization - layer-level output-error benchmark.

Stresses candidate RE-RANKING (core.resonant.resonant_rerank_quantize) against
the plain Euclidean snap in the identical per-row-Hadamard frame, as the rate
drops from ~4 bits toward ~3 bits, on real Qwen3.5-2B Linear layers. Reports
the TRUE output error ||(W-What)X||^2/||WX||^2 and the Leech index entropy
(bits/weight) for each - the comparison is only meaningful if the rate matches,
so both are printed.

Measured (160-row subset; reranked vs Euclidean, matched rate):

    layer          ~4.1 bits   ~3.7 bits   ~3.45 bits
    L0.out_proj      +1.0%       +3.0%        +1.6%
    L3.o_proj        +5.6%       +8.8%        +5.8%

The reranking gain is real, grows toward lower bit-rates (the scalar form
saturates at 4 bits), is layer-dependent (full-attention layers carry more
per-block anisotropy), and is bounded by the lattice near-isotropy the
Hadamard incoherence induces. It is a low-bit-rate top-up, not a second big
lever. End-to-end perplexity at 3 bits is the remaining arbiter (expensive:
candidate enumeration is ~n_candidates x the scalar CVP cost).
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


def sweep(W, X, scales, n_candidates=8, dither=0.4, seed=42):
    nw = W.size
    for scale in scales:
        t0 = time.time()
        We, me = resonant_rerank_quantize(W, X, scale=scale, n_candidates=1, rerank=False, seed=seed)
        eu, be = output_relmse(W, X, We), bits_per_weight(me["points"], nw)
        Wr, mr = resonant_rerank_quantize(
            W, X, scale=scale, n_candidates=n_candidates, dither=dither, rerank=True, seed=seed
        )
        rr, br = output_relmse(W, X, Wr), bits_per_weight(mr["points"], nw)
        gain = 100 * (1 - rr / eu)
        print(
            f"  scale={scale:.2f}  euclid={eu:.4e} ({be:.2f}b)  "
            f"rerank={rr:.4e} ({br:.2f}b)  gain={gain:+.1f}%  [{time.time()-t0:.0f}s]",
            flush=True,
        )


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

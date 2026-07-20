"""
Resonant Quantization - end-to-end perplexity reproduction.

Measures WikiText-2 perplexity on Qwen3.5-2B-Base with 3 representative
Linear layers quantized to ~INT4 rate by each of: plain KGC (Leech CVP),
GPTQ (own impl), and Resonant (activation-aware Leech, core/resonant.py).
Everything else stays fp32. Calibration = WikiText-2 TRAIN, eval = TEST
(no leakage; the standard PTQ protocol).

Measured result (this script, CPU sandbox):

    fp32       ppl 8.887   -
    KGC        ppl 8.957   +0.070
    GPTQ       ppl 8.944   +0.057   (rank-deficient Hessian at 1024 tokens;
                                      full-Hessian GPTQ is +0.035, see
                                      benchmark_llm_perplexity.py)
    Resonant   ppl 8.889   +0.002   <- activation-aware, alpha=0.5

Resonant reduces the penalty ~35x vs plain KGC and beats GPTQ in both the
matched-calibration (+0.057) and full-Hessian (+0.035) regimes, because it
depends only on per-channel activation RMS - robust when calibration is
thin - not on a full-rank Hessian.

NOTE ON RUNTIME: exact Leech CVP is slow on CPU (~9 min per layer per
method for a 2048x2048 weight). The reference run quantized the 6 lattice
weights (2 methods x 3 layers) out-of-process and in parallel to dodge the
torch/OpenBLAS thread contention that throttles an in-process CVP; this
single-process script is the readable spec of that pipeline, not the fast
path. Set KGC_RESONANT_FAST=1 to shrink to a smoke test.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
from benchmark_llm_perplexity import (  # noqa: E402
    compute_perplexity,
    gptq_quantize_,
    load_wikitext2,
)

if "e8zip" not in sys.modules:
    import importlib.util

    _root = _here.parent
    _spec = importlib.util.spec_from_file_location(
        "e8zip", _root / "__init__.py", submodule_search_locations=[str(_root)]
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["e8zip"] = _mod
    _spec.loader.exec_module(_mod)

from e8zip.core.resonant import resonant_quantize  # noqa: E402

MODEL = "Qwen/Qwen3.5-2B-Base"
CALIB_TOKENS = 1024
SCALE = 4.0


def layer_handles(model):
    """The 3 representative Linear layers (both attention flavours)."""
    return {
        "L0.out_proj": model.model.layers[0].linear_attn.out_proj,
        "L0.in_proj_z": model.model.layers[0].linear_attn.in_proj_z,
        "L3.o_proj": model.model.layers[3].self_attn.o_proj,
    }


def capture_activations(model, tokenizer, mods, train_texts):
    """Per-layer calibration input activations X [tokens, in] from TRAIN."""
    store = {k: [] for k in mods}
    hooks = [
        m.register_forward_hook(
            lambda mod, inp, out, k=k: store[k].append(
                inp[0].reshape(-1, inp[0].shape[-1]).detach()
            )
        )
        for k, m in mods.items()
    ]
    calib = "\n\n".join([t for t in train_texts if len(t) > 200][:64])
    ids = tokenizer(calib, return_tensors="pt").input_ids[:, :2048]
    with torch.no_grad():
        model(ids)
    for h in hooks:
        h.remove()
    return {k: torch.cat(v, 0).numpy().astype(np.float32)[:CALIB_TOKENS] for k, v in store.items()}


def quantize_weight(W, X, method):
    """Return a quantized copy of W [out, in] under the named method."""
    W64 = W.astype(np.float64)
    if method == "kgc":
        Wq, _, _ = resonant_quantize(W64, X, scale=SCALE, alphas=(0.0,))
        return Wq.astype(np.float32)
    if method == "resonant":
        Wq, _, _ = resonant_quantize(W64, X, scale=SCALE, alphas=(0.5,))
        return Wq.astype(np.float32)
    if method == "gptq":
        Wt = torch.from_numpy(W64.copy())
        H = torch.from_numpy((X.astype(np.float64).T @ X.astype(np.float64)).astype(np.float32))
        gptq_quantize_(Wt, H, bits=4, group_size=128)
        return Wt.numpy().astype(np.float32)
    raise ValueError(method)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    fast = os.environ.get("KGC_RESONANT_FAST") == "1"
    n_windows = 4 if fast else 40

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    train, test = load_wikitext2()
    mods = layer_handles(model)
    orig = {k: m.weight.data.clone() for k, m in mods.items()}
    print(f"[{time.time()-t0:.0f}s] model + data loaded", flush=True)

    X = capture_activations(model, tok, mods, train["text"])
    print(f"[{time.time()-t0:.0f}s] captured activations", flush=True)

    # Pre-quantize every (layer, method) weight once.
    quant = {}
    for k, m in mods.items():
        W = orig[k].numpy()
        for method in ("kgc", "resonant", "gptq"):
            quant[(k, method)] = quantize_weight(W, X[k], method)
            print(f"[{time.time()-t0:.0f}s] quantized {k}/{method}", flush=True)

    eval_text = "\n\n".join(test["text"][:400])

    def ppl():
        return compute_perplexity(
            model, tok, eval_text, seq_len=1024, stride=512, max_windows=n_windows
        )

    def restore():
        for k, m in mods.items():
            m.weight.data.copy_(orig[k])

    results = {}
    results["fp32"] = ppl()
    print(f"fp32     {results['fp32']:.3f}", flush=True)
    for method in ("kgc", "resonant", "gptq"):
        for k, m in mods.items():
            m.weight.data.copy_(torch.from_numpy(quant[(k, method)]).to(m.weight.dtype))
        results[method] = ppl()
        print(
            f"{method:8s} {results[method]:.3f}  (delta {results[method]-results['fp32']:+.3f})",
            flush=True,
        )
        restore()

    print("\n=== RESONANT PERPLEXITY (3 layers, Qwen3.5-2B-Base) ===")
    for k, v in results.items():
        print(f"  {k:9s} ppl={v:.3f}  delta={v-results['fp32']:+.3f}")


if __name__ == "__main__":
    main()

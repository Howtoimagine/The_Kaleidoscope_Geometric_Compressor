"""
Resonant Quantization - proof of the kernel

The perplexity benchmark (benchmark_llm_perplexity.py) showed KGC beats
naive RTN but trails GPTQ (+0.070 vs +0.035 ppl at 4 bits). This script
locates *why* and proves the mechanism that closes the gap - using
parts already in the system, from first principles.

FIRST, a hypothesis honestly killed. The repo's founding slogan is
"compress geodesic PATHS through the lattice, not just nodes." So: is a
real weight matrix, snapped block-by-block to the Leech lattice, a
smooth trajectory whose consecutive-point DELTAS live on low shells
(cheap) while the absolute points live on high shells (dear)? Measured
on real Qwen3.5-2B weights: NO. Deltas land on ~2x HIGHER shells than
the absolute points, in BOTH matrix axes. At the scale that gives ~4
bits/weight the blocks are essentially independent high-shell points;
g_idx is already near-uniform (11.4 / 12 bits); there are zero exact
duplicate blocks. The RATE is near-optimal - ~4 bits/weight is close to
the real information content. There is no free lunch in path/codebook/
predictive coding of the indices.

So the lever is DISTORTION, not rate. And here is the kernel this script
proves: the Leech CVP snaps each block to the EUCLIDEAN-nearest lattice
point - it minimizes ||W - What||. But the model does not experience
weight error; it experiences OUTPUT error ||(W - What) X|| on real
activations X. Those are different objectives, and the Euclidean-nearest
point is NOT the output-nearest point.

The cheapest possible fix (proven below): scale each input channel by
its activation importance BEFORE snapping, so the lattice spends its
resolution where the model is sensitive, and unscale after. This
deliberately RAISES weight error and LOWERS output error at the identical
bit-rate. It is AWQ's insight, realized inside the lattice quantizer.

The full upgrade this kernel motivates - "Resonant Quantization" - is in
ARCHITECTURE.md: use the recall engine (core/recall.py, built for
associative memory) to enumerate the lattice points in the neighborhood
of each weight block, and snap to the one that minimizes activation-
weighted output error rather than distance. Quantization as behavior-
matched associative recall: choose the lattice point the model cannot
tell from the original, not merely the closest one.

Run:  python benchmarks/benchmark_resonant.py
truth_status: real_activation_distortion_measurement
"""

import sys
import time
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
import importlib.util

if "e8zip" not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        "e8zip", _root / "__init__.py", submodule_search_locations=[str(_root)]
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["e8zip"] = _mod
    _spec.loader.exec_module(_mod)

from e8zip.core.leech_lattice import batch_nearest_leech_point  # noqa: E402
from e8zip.core import transforms  # noqa: E402

MODEL_ID = "Qwen/Qwen3.5-2B-Base"


def capture_layer_activations(layer_idx=3, n_tokens=512, in_cols=1536, out_rows=160):
    """One forward pass; return (W, X) for one down_proj, sliced for speed."""
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    model.eval()
    target = model.model.layers[layer_idx].mlp.down_proj

    store = []
    h = target.register_forward_hook(
        lambda m, inp, out: store.append(
            inp[0].reshape(-1, inp[0].shape[-1]).detach()
        )
    )
    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(test["text"][:60]), return_tensors="pt").input_ids[:, :n_tokens]
    with torch.no_grad():
        model(ids)
    h.remove()

    X = torch.cat(store, 0).numpy().astype(np.float64)[:, :in_cols]
    W = target.weight.data.numpy().astype(np.float64)[:out_rows, :in_cols]
    return W, X


def fake_quant(W, scale=4.0, hdim=128, seed=42):
    """Dequantized weights via Hadamard incoherence + exact Leech CVP.

    Skips the (slow, pure-Python) range coder: rate is identical at fixed
    scale, and only the dequantized weights are needed to compare
    distortion.
    """
    shp = W.shape
    flat = W.ravel()
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


def main():
    print("capturing real activations (one forward pass) ...", flush=True)
    t0 = time.time()
    W, X = capture_layer_activations()
    print(f"  W={W.shape} X={X.shape}  ({time.time()-t0:.0f}s)\n", flush=True)

    def out_mse(Wh):
        return float((((W - Wh) @ X.T) ** 2).mean())

    def w_mse(Wh):
        return float(((W - Wh) ** 2).mean())

    # per-input-channel importance = activation RMS
    s = np.sqrt((X**2).mean(0))
    s = np.clip(s / s.mean(), 1e-3, 1e3)

    t0 = time.time()
    W_plain = fake_quant(W)
    W_aware = fake_quant(W * s) / s
    print(f"quantized both ({time.time()-t0:.0f}s), identical bit-rate\n")

    print(f"{'method':16s} {'weight-MSE':>13s} {'OUTPUT-MSE':>13s}")
    print(f"{'plain CVP':16s} {w_mse(W_plain):13.4e} {out_mse(W_plain):13.4e}")
    print(f"{'activation-aware':16s} {w_mse(W_aware):13.4e} {out_mse(W_aware):13.4e}")
    delta = (1 - out_mse(W_aware) / out_mse(W_plain)) * 100
    print(f"\nOUTPUT-error change (aware vs plain): {delta:+.1f}%")
    print(
        "Plain CVP wins weight-MSE by construction. Aware winning OUTPUT-MSE\n"
        "at the same rate proves the kernel: minimize disagreement, not distance.\n"
        "This scalar version is AWQ-in-the-lattice; the full Resonant upgrade\n"
        "(recall-reranked lattice candidates, see ARCHITECTURE.md) goes further."
    )


if __name__ == "__main__":
    main()

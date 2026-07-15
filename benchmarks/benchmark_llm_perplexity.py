"""
LLM Perplexity Benchmark: KGC vs RTN vs GPTQ on a real generative model

Every previous tensor-regime benchmark in this repo measured weight-
space SNR/MSE on individual tensors. That is a proxy. The metric that
actually decides whether a quantizer is good for LLMs is perplexity
after the quantized weights are put back into the model and it
generates text. This script closes that gap.

Model: Qwen/Qwen3.5-2B-Base (text-only causal LM; the repo also ships
a vision-language variant, unused here). A hybrid architecture -
gated linear attention (GatedDeltaNet-style) on 3 of every 4 layers,
full GQA attention on the 4th - but every quantizable weight is still
a plain nn.Linear, so "quantize every nn.Linear under model.model,
skip embeddings/lm_head/norms/conv1d" is a clean, standard policy.

Eval: standard sliding-window causal LM perplexity on WikiText-2
(raw), matching how GPTQ/AWQ papers report numbers - stride=512,
seq_len=2048 (context-limited to what's practical on CPU).

Methods compared, all "fake-quantize" (dequantize immediately, no
packed-int4 kernels - the point is measuring the effect on model
output, not inference speed):

    fp32       baseline, no quantization
    rtn_int4   round-to-nearest, per-output-row scale (the naive
               baseline every PTQ paper beats)
    gptq_int4  Hessian-based sequential quantization with error
               compensation (Frantar et al. 2022) - implemented here
               in plain torch since auto-gptq's CUDA-era API doesn't
               import against this transformers version
    kgc        this repo's TENSOR regime (Hadamard incoherence ->
               exact Leech CVP -> entropy-coded indices)

truth_status: real_model_perplexity_measurement
"""

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_root = Path(__file__).resolve().parent.parent
import importlib.util

if "e8zip" not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        "e8zip", _root / "__init__.py", submodule_search_locations=[str(_root)]
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["e8zip"] = _mod
    _spec.loader.exec_module(_mod)

from e8zip.core.kgc import KGCCompressor  # noqa: E402

MODEL_ID = "Qwen/Qwen3.5-2B-Base"


# ======================================================================
# Perplexity
# ======================================================================


def load_wikitext2():
    from datasets import load_dataset

    # Salesforce/wikitext is a parquet mirror of the same data; the
    # original "wikitext" repo uses a legacy loading script that this
    # datasets/huggingface_hub version combination fails to resolve
    # (an hf:// URI parsing bug on the .huggingface.yaml lookup).
    train = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    return train, test


def compute_perplexity(model, tokenizer, text: str, seq_len: int = 2048, stride: int = 512,
                        max_windows: int = None, device: str = "cpu") -> float:
    """Standard sliding-window causal LM perplexity (matches GPTQ/AWQ papers)."""
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    n = input_ids.size(1)

    nlls = []
    n_tokens = 0
    prev_end = 0
    windows = list(range(0, n, stride))
    if max_windows is not None:
        windows = windows[:max_windows]

    model.eval()
    with torch.no_grad():
        for start in windows:
            end = min(start + seq_len, n)
            trg_len = end - prev_end
            ids = input_ids[:, start:end]
            target = ids.clone()
            target[:, :-trg_len] = -100  # only score the new (non-overlapping) part

            out = model(ids, labels=target)
            n_valid = (target != -100).sum().item()
            nlls.append(out.loss.item() * n_valid)
            n_tokens += n_valid
            prev_end = end
            if end == n:
                break

    return math.exp(sum(nlls) / n_tokens)


# ======================================================================
# Layer discovery
# ======================================================================


def quantizable_linears(model) -> dict:
    """All nn.Linear under model.model (decoder body) - excludes lm_head, embeddings, norms."""
    out = {}
    for name, mod in model.model.named_modules():
        if isinstance(mod, nn.Linear):
            out[f"model.{name}"] = mod
    return out


# ======================================================================
# Method 1: RTN INT4 (per-output-row scale, the standard naive baseline)
# ======================================================================


def rtn_quantize_(weight: torch.Tensor, bits: int = 4) -> None:
    """In-place round-to-nearest quantization, symmetric, per-row scale."""
    qmax = 2 ** (bits - 1) - 1
    w = weight.data
    scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
    q = torch.clamp(torch.round(w / scale), -qmax, qmax)
    w.copy_(q * scale)


# ======================================================================
# Method 2: GPTQ INT4 (Hessian-based, Frantar et al. 2022)
# ======================================================================


class HessianCollector:
    """Accumulates X^T X per targeted Linear layer via forward hooks."""

    def __init__(self, layers: dict):
        self.layers = layers
        self.H = {name: None for name in layers}
        self.n_samples = {name: 0 for name in layers}
        self.handles = []

    def _hook(self, name):
        def fn(module, inputs, output):
            x = inputs[0]
            x = x.reshape(-1, x.shape[-1]).to(torch.float32)
            n = x.shape[0]
            h = x.t() @ x
            if self.H[name] is None:
                self.H[name] = h
            else:
                self.H[name] += h
            self.n_samples[name] += n

        return fn

    def __enter__(self):
        for name, mod in self.layers.items():
            self.handles.append(mod.register_forward_hook(self._hook(name)))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()


def gptq_quantize_(weight: torch.Tensor, H: torch.Tensor, bits: int = 4,
                    damp: float = 0.01, group_size: int = 128) -> None:
    """
    In-place GPTQ quantization of one Linear layer's weight (d_out, d_in),
    given its accumulated input Hessian H (d_in, d_in).

    Grouped symmetric quantization (group_size columns share a scale,
    recomputed per group at the moment that group is reached - matches
    standard GPTQ practice), sequential column-wise error compensation
    via the Cholesky factor of the damped Hessian inverse.
    """
    W = weight.data.to(torch.float32).clone()
    d_out, d_in = W.shape
    qmax = 2 ** (bits - 1) - 1

    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0  # avoid singular columns (e.g. never-activated inputs)
    damp_val = damp * torch.mean(torch.diag(H))
    H += torch.eye(d_in) * damp_val

    # Cholesky of H^-1, upper triangular (standard GPTQ formulation)
    Hinv = torch.linalg.cholesky(torch.linalg.inv(H), upper=True)

    for c0 in range(0, d_in, group_size):
        c1 = min(c0 + group_size, d_in)
        block = W[:, c0:c1].clone()
        scale = block.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax

        Hinv_block = Hinv[c0:c1, c0:c1]
        err_block = torch.zeros_like(block)
        for i in range(c1 - c0):
            w_col = block[:, i]
            q_col = torch.clamp(torch.round(w_col / scale.squeeze(1)), -qmax, qmax) * scale.squeeze(1)
            d = Hinv_block[i, i]
            err = (w_col - q_col) / d
            block[:, i] = q_col
            if i < c1 - c0 - 1:
                block[:, i + 1 :] -= torch.outer(err, Hinv_block[i, i + 1 :])
            err_block[:, i] = err

        W[:, c0:c1] = block
        if c1 < d_in:
            # propagate the whole block's compensation to remaining columns
            W[:, c1:] -= err_block @ Hinv[c0:c1, c1:]

    weight.data.copy_(W.to(weight.dtype))


# ======================================================================
# Method 3: KGC TENSOR regime
# ======================================================================


def kgc_quantize_(weight: torch.Tensor, kgc: KGCCompressor, scale: float = 4.0) -> int:
    """In-place KGC quantization. Returns compressed archive size in bytes."""
    w = weight.data.to(torch.float32).numpy()
    blob = kgc.compress_tensor(w, scale=scale)
    w2, _ = kgc.decompress_tensor(blob)
    weight.data.copy_(torch.from_numpy(w2.reshape(w.shape)).to(weight.dtype))
    return len(blob)


# ======================================================================
# Orchestration
# ======================================================================


def get_calibration_batches(tokenizer, train_texts, n_samples=16, seq_len=512):
    """
    Non-overlapping seq_len-token windows from concatenated train text.

    WikiText-2's `text` field is per-line (headers, short paragraphs) -
    tokenizing lines individually and requiring each to reach seq_len
    on its own silently returns zero batches (nearly every line is
    shorter than 512 tokens). Concatenate first, like the eval text.
    """
    joined = "\n\n".join(train_texts)
    ids = tokenizer(joined, return_tensors="pt").input_ids
    batches = []
    for start in range(0, ids.size(1) - seq_len + 1, seq_len):
        batches.append(ids[:, start : start + seq_len])
        if len(batches) >= n_samples:
            break
    return batches


def run_experiment(
    layer_names,
    eval_max_windows=40,
    eval_seq_len=1024,
    eval_stride=512,
    rtn_bits=4,
    gptq_bits=4,
    gptq_group_size=128,
    kgc_scale=4.0,
    log=print,
):
    """
    Fixed-scope comparison: fp32 baseline vs RTN vs GPTQ vs KGC, applied
    to exactly `layer_names` (everything else in the model stays fp32).
    Returns a dict of results; also prints progress via `log`.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log(f"loading {MODEL_ID} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    model.eval()

    all_layers = quantizable_linears(model)
    targets = {name: all_layers[name] for name in layer_names}
    total_params = sum(m.weight.numel() for m in targets.values())
    total_model_params = sum(m.weight.numel() for m in all_layers.values())
    log(
        f"target scope: {len(targets)} layers, {total_params/1e6:.2f}M params "
        f"({total_params/total_model_params*100:.3f}% of {total_model_params/1e6:.0f}M quantizable)"
    )

    log("loading WikiText-2 ...")
    _, test_ds = load_wikitext2()
    eval_text = "\n\n".join(test_ds["text"][:400])  # bounded slice, not the full test set

    originals = {name: m.weight.data.clone() for name, m in targets.items()}

    results = {}

    def restore():
        for name, m in targets.items():
            m.weight.data.copy_(originals[name])

    t0 = time.time()
    ppl = compute_perplexity(model, tok, eval_text, seq_len=eval_seq_len,
                              stride=eval_stride, max_windows=eval_max_windows)
    log(f"[fp32 baseline]  ppl={ppl:.3f}  ({time.time()-t0:.0f}s)")
    results["fp32"] = {"ppl": ppl, "bits_per_weight": 32.0}

    # RTN
    t0 = time.time()
    for name, m in targets.items():
        rtn_quantize_(m.weight, bits=rtn_bits)
    ppl = compute_perplexity(model, tok, eval_text, seq_len=eval_seq_len,
                              stride=eval_stride, max_windows=eval_max_windows)
    log(f"[rtn_int{rtn_bits}]     ppl={ppl:.3f}  ({time.time()-t0:.0f}s)")
    results["rtn"] = {"ppl": ppl, "bits_per_weight": float(rtn_bits)}
    restore()

    # GPTQ
    log("GPTQ: collecting calibration Hessians ...")
    train_ds, _ = load_wikitext2()
    calib_texts = [t for t in train_ds["text"] if len(t) > 200][:64]
    calib_batches = get_calibration_batches(tok, calib_texts, n_samples=16, seq_len=512)
    t0 = time.time()
    with HessianCollector(targets) as hc:
        with torch.no_grad():
            for b in calib_batches:
                model(b)
    log(f"  calibration forward passes: {time.time()-t0:.0f}s, "
        f"n_samples={[hc.n_samples[n] for n in targets]}")

    t0 = time.time()
    for name, m in targets.items():
        gptq_quantize_(m.weight, hc.H[name], bits=gptq_bits, group_size=gptq_group_size)
    ppl = compute_perplexity(model, tok, eval_text, seq_len=eval_seq_len,
                              stride=eval_stride, max_windows=eval_max_windows)
    log(f"[gptq_int{gptq_bits}]    ppl={ppl:.3f}  ({time.time()-t0:.0f}s incl. quantization)")
    results["gptq"] = {"ppl": ppl, "bits_per_weight": float(gptq_bits)}
    restore()

    # KGC
    kgc = KGCCompressor()
    t0 = time.time()
    total_bytes = 0
    for name, m in targets.items():
        total_bytes += kgc_quantize_(m.weight, kgc, scale=kgc_scale)
    kgc_bpw = total_bytes * 8 / total_params
    ppl = compute_perplexity(model, tok, eval_text, seq_len=eval_seq_len,
                              stride=eval_stride, max_windows=eval_max_windows)
    log(f"[kgc scale={kgc_scale}] ppl={ppl:.3f}  bits/weight={kgc_bpw:.2f}  "
        f"({time.time()-t0:.0f}s incl. quantization)")
    results["kgc"] = {"ppl": ppl, "bits_per_weight": kgc_bpw}
    restore()

    results["_meta"] = {
        "model": MODEL_ID,
        "target_layers": list(layer_names),
        "target_params": total_params,
        "total_quantizable_params": total_model_params,
        "scope_fraction": total_params / total_model_params,
        "eval_max_windows": eval_max_windows,
        "eval_seq_len": eval_seq_len,
        "truth_status": "partial_scope_real_model_perplexity_measurement",
    }
    return results


if __name__ == "__main__":
    LAYERS = [
        "model.layers.0.linear_attn.out_proj",
        "model.layers.0.linear_attn.in_proj_z",
        "model.layers.3.self_attn.o_proj",
    ]
    results = run_experiment(LAYERS)
    print("\n=== SUMMARY ===")
    for method in ("fp32", "rtn", "gptq", "kgc"):
        r = results[method]
        print(f"{method:6s}: ppl={r['ppl']:.3f}  bits/weight={r['bits_per_weight']:.2f}")
    print(f"\nscope: {results['_meta']['target_params']/1e6:.2f}M / "
          f"{results['_meta']['total_quantizable_params']/1e6:.0f}M params "
          f"({results['_meta']['scope_fraction']*100:.3f}%)")

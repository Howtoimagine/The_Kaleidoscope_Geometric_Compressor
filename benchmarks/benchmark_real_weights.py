"""
Real-Weight Rate-Distortion Benchmark

Everything in benchmark_kgc.py runs on synthetic Gaussian tensors,
which is the easy case for a lattice quantizer (i.i.d., no outliers,
no per-channel scale structure). This benchmark runs on ACTUAL model
weights, matching the Glass Network's own methodology
(benchmarks/turbo_leech_rate_distortion.py, glass_windows branch) so
the comparison is apples-to-apples with its published numbers:

    Leech VQ (their harness, no gain/shape split): 19.85 dB @ 4.81 bpw
    INT5 (entropy-coded):                          21.50 dB @ 3.96 bpw
    INT6 (entropy-coded):                           27.79 dB @ 4.99 bpw

i.e. on real weights, undifferentiated Leech VQ LOSES to entropy-coded
scalar quantization. Measured result (see ARCHITECTURE.md for the full
writeup): KGC's TENSOR regime (exact-CVP + real entropy coding) reverses
that - 23.85 dB @ 4.34 bpw average, beating both the Glass Network
reference AND INT5 at a comparable rate.

The gain/shape progressive codec (core/klc.py) was hypothesized to win
outright by additionally separating magnitude from direction. Measured
result: it does NOT - at every tested rate (default and hand-tuned
hyperparameters, low-rate and high-rate), the flat TENSOR regime
Pareto-dominates it (equal-or-better dB at equal-or-fewer bits). KLC's
real, verified value is progressive/truncatable decode, which the flat
regime cannot do at all - not a compression-ratio win. Both findings are
reported here rather than only the flattering one.

truth_status: real_weights_rate_distortion_measurement
"""

import math
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

from e8zip.core.kgc import KGCCompressor  # noqa: E402
from e8zip.core import klc  # noqa: E402

import os

# The Glass Network harness caps at 49_152 elements/tensor; the exact
# Leech CVP here runs in pure numpy (~35 blocks/s, documented limitation
# in ARCHITECTURE.md) and two codecs each do one CVP pass per tensor, so
# that cap is a ~2min/tensor benchmark. Default to a smaller real sample
# for a benchmark that finishes in reasonable time; override via env var
# for a full-size run.
CAP = int(os.environ.get("KGC_BENCH_CAP", "9600"))


def _snr_db(orig: np.ndarray, recon: np.ndarray) -> float:
    err = recon.astype(np.float64) - orig.astype(np.float64)
    mse = float(np.mean(err * err))
    sig = float(np.mean(orig.astype(np.float64) ** 2))
    return 10.0 * math.log10(sig / mse) if (sig > 1e-12 and mse > 1e-12) else 0.0


def _entropy_bits(symbols: np.ndarray) -> float:
    if symbols.size == 0:
        return 0.0
    _, counts = np.unique(symbols, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p + 1e-12)))


def scalar_int_entropy_coded(flat: np.ndarray, bits: int, block_dim: int = 384) -> dict:
    """Blockwise symmetric scalar quant with entropy-coded rate (Glass Network method)."""
    n = flat.size
    pad = int(math.ceil(n / block_dim) * block_dim)
    padded = np.zeros(pad, dtype=np.float64)
    padded[:n] = flat
    qmax = (2 ** (bits - 1)) - 1
    recon = np.zeros_like(padded)
    codes = []
    for s in range(0, pad, block_dim):
        blk = padded[s : s + block_dim]
        mx = float(np.max(np.abs(blk))) if blk.size else 0.0
        if mx <= 1e-12:
            continue
        scale = max(mx / qmax, 1e-12)
        q = np.clip(np.round(blk / scale), -qmax, qmax)
        codes.append(q.astype(np.int64))
        recon[s : s + block_dim] = q * scale
    ent = _entropy_bits(np.concatenate(codes)) if codes else 0.0
    return {"recon": recon[:n], "entropy_bits": ent}


def find_or_fetch_model() -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download("sentence-transformers/all-MiniLM-L6-v2", "model.safetensors")


def main() -> None:
    model_path = sys.argv[1] if len(sys.argv) > 1 else find_or_fetch_model()
    max_tensors = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    from safetensors import safe_open

    kgc = KGCCompressor()
    rows = []

    print(f"model: {model_path}")
    print(f"{'tensor':46s} | {'KGC-flat':>16s} | {'KLC L0':>14s} | {'KLC L2':>14s} | {'INT5':>14s} | {'INT6':>14s}")

    with safe_open(model_path, framework="np") as h:
        keys = [
            k
            for k in h.keys()
            if k.endswith(".weight") and len(h.get_slice(k).get_shape()) == 2
        ]
        keys = keys[:max_tensors]
        for k in keys:
            arr = np.asarray(h.get_slice(k)[:, :], dtype=np.float64).reshape(-1)
            if arr.size < 24:
                continue
            sampled = arr.size > CAP
            if sampled:
                arr = arr[:CAP]

            # KGC flat (Hadamard -> exact Leech CVP -> entropy code)
            t0 = time.time()
            blob = kgc.compress_tensor(arr.astype(np.float32), scale=4.0)
            recon, _ = kgc.decompress_tensor(blob)
            flat_bpw = len(blob) * 8 / arr.size
            flat_snr = _snr_db(arr, recon.astype(np.float64).ravel()[: arr.size])
            flat_t = time.time() - t0

            # KLC gain/shape progressive, base layer only (L0) and +2 layers (L2)
            t0 = time.time()
            kblob = klc.encode(arr.astype(np.float32), n_layers=2, shape_scale=16.0)
            klc_t = time.time() - t0
            r0 = klc.decode(kblob, up_to_layer=0)
            r2 = klc.decode(kblob, up_to_layer=2)
            sizes = klc.layer_sizes(kblob)
            l0_bpw = sizes["base"] * 8 / arr.size
            l2_bpw = sizes["layer_2"] * 8 / arr.size
            l0_snr = _snr_db(arr, r0.astype(np.float64).ravel()[: arr.size])
            l2_snr = _snr_db(arr, r2.astype(np.float64).ravel()[: arr.size])

            i5 = scalar_int_entropy_coded(arr, 5)
            i6 = scalar_int_entropy_coded(arr, 6)
            i5_snr = _snr_db(arr, i5["recon"])
            i6_snr = _snr_db(arr, i6["recon"])

            row = {
                "tensor": k,
                "elements": int(arr.size),
                "sampled": bool(sampled),
                "kgc_flat_snr_db": flat_snr,
                "kgc_flat_bpw": flat_bpw,
                "klc_l0_snr_db": l0_snr,
                "klc_l0_bpw": l0_bpw,
                "klc_l2_snr_db": l2_snr,
                "klc_l2_bpw": l2_bpw,
                "int5_snr_db": i5_snr,
                "int5_bpw": i5["entropy_bits"],
                "int6_snr_db": i6_snr,
                "int6_bpw": i6["entropy_bits"],
            }
            rows.append(row)
            print(
                f"{k[:46]:46s} | {flat_snr:5.1f}dB@{flat_bpw:4.2f}b | "
                f"{l0_snr:5.1f}dB@{l0_bpw:4.2f}b | {l2_snr:5.1f}dB@{l2_bpw:4.2f}b | "
                f"{i5_snr:5.1f}dB@{i5['entropy_bits']:4.2f}b | {i6_snr:5.1f}dB@{i6['entropy_bits']:4.2f}b"
                f"{'  [sampled]' if sampled else ''}"
            )

    def avg(key):
        return float(np.mean([r[key] for r in rows]))

    print("\n=== AGGREGATE (real weights, all-MiniLM-L6-v2) ===")
    print(f"KGC flat  : {avg('kgc_flat_snr_db'):.2f} dB @ {avg('kgc_flat_bpw'):.2f} bits/weight")
    print(f"KLC base  : {avg('klc_l0_snr_db'):.2f} dB @ {avg('klc_l0_bpw'):.2f} bits/weight")
    print(f"KLC +2L   : {avg('klc_l2_snr_db'):.2f} dB @ {avg('klc_l2_bpw'):.2f} bits/weight")
    print(f"INT5      : {avg('int5_snr_db'):.2f} dB @ {avg('int5_bpw'):.2f} entropy-bits")
    print(f"INT6      : {avg('int6_snr_db'):.2f} dB @ {avg('int6_bpw'):.2f} entropy-bits")
    print("\nreference (Glass Network turbo_leech_rate_distortion.json, no gain/shape split):")
    print("Leech(theirs): 19.85 dB @ 4.81 bpw | INT5: 21.50 dB @ 3.96 bpw | INT6: 27.79 dB @ 4.99 bpw")


if __name__ == "__main__":
    main()

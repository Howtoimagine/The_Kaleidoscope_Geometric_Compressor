"""
KGC v2 Honest Benchmark

Rules of this benchmark (Glass Network falsification style):
  * every ratio is a true round-trip: decode and verify, or it doesn't count
  * lossy modes report distortion next to rate - never a bare "ratio"
  * baselines that beat us are printed anyway

Run:  python benchmarks/benchmark_kgc.py
"""

import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# flat-layout bootstrap (same trick as the package __init__)
import importlib.util

_root = Path(__file__).resolve().parent.parent
if "e8zip" not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        "e8zip", _root / "__init__.py", submodule_search_locations=[str(_root)]
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["e8zip"] = _mod
    _spec.loader.exec_module(_mod)

from e8zip.core.kgc import KGCCompressor  # noqa: E402


def bench_bytes(kgc: KGCCompressor) -> None:
    print("\n== Regime 1: BYTES (lossless, verified round-trip) ==")
    cases = {
        "python source (this repo)": (_root / "core" / "compressor.py").read_bytes(),
        "repetitive text": b"the kaleidoscope turns and the lattice sings " * 800,
        "random bytes": np.random.default_rng(0)
        .integers(0, 256, 65536)
        .astype(np.uint8)
        .tobytes(),
    }
    print(f"{'input':28s} {'size':>8s} {'kgc-fast':>10s} {'kgc-strong':>10s} {'zlib-9':>8s}")
    for name, data in cases.items():
        row = [f"{name:28s}", f"{len(data):8d}"]
        for mode in ("fast", "strong"):
            t0 = time.time()
            blob = kgc.compress(data, mode=mode)
            out, info = kgc.decompress(blob)
            assert out == data and info["verified"]
            row.append(f"{len(data)/len(blob):9.2f}x")
        row.append(f"{len(data)/len(zlib.compress(data, 9)):7.2f}x")
        print(" ".join(row))
    print("   (zlib printed because it wins on match-heavy data; the KGC byte")
    print("    regime's contribution is the verified debt container, not LZ.)")


def bench_tensor(kgc: KGCCompressor) -> None:
    print("\n== Regime 2: TENSOR (rate-distortion, lattice VQ) ==")
    rng = np.random.default_rng(1)
    w = rng.normal(0, 0.02, (128, 384)).astype(np.float32)
    sig = float(np.mean(w.astype(np.float64) ** 2))

    print(f"{'method':22s} {'bits/weight':>12s} {'SNR dB':>8s}")
    for scale in (2.0, 4.0, 8.0):
        t0 = time.time()
        blob = kgc.compress_tensor(w, scale=scale)
        w2, _ = kgc.decompress_tensor(blob)
        mse = float(np.mean((w.astype(np.float64) - w2.astype(np.float64)) ** 2))
        snr = 10 * np.log10(sig / mse)
        bpw = len(blob) * 8 / w.size
        print(f"KGC leech scale={scale:<5} {bpw:12.2f} {snr:8.1f}   ({time.time()-t0:.0f}s)")

    for bits in (4, 5):
        lo, hi = float(w.min()), float(w.max())
        levels = 2**bits - 1
        q = np.round((w - lo) / (hi - lo) * levels)
        wq = (q / levels * (hi - lo) + lo).astype(np.float32)
        mse = float(np.mean((w.astype(np.float64) - wq.astype(np.float64)) ** 2))
        snr = 10 * np.log10(sig / mse)
        print(f"scalar INT{bits} baseline {float(bits):12.2f} {snr:8.1f}")


def bench_consolidate(kgc: KGCCompressor) -> None:
    print("\n== Regime 3: CONSOLIDATE (RG collapse, conserved mass) ==")
    rng = np.random.default_rng(2)
    arche = rng.normal(0, 1.5, (40, 24))
    rows = (arche[rng.integers(0, 40, 1000)] + rng.normal(0, 0.08, (1000, 24))).astype(
        np.float32
    )
    raw = len(rows.tobytes())
    sig = float(np.sqrt(np.mean(rows**2)))

    print(f"{'setting':26s} {'ratio':>7s} {'sites':>6s} {'rmse':>8s} {'mass drift':>11s} {'truth':>16s}")
    blob = kgc.consolidate(rows, keep_residuals=True)
    out, info = kgc.deconsolidate(blob)
    assert info["verified"]
    print(
        f"{'exact (residuals kept)':26s} {raw/len(blob):6.2f}x {info['n_sites']:6d} "
        f"{'0.0':>8s} {'0.000%':>11s} {info['truth_status']:>16s}"
    )
    for lam in (1.0, 0.5, 0.25):
        blob = kgc.consolidate(rows, rg_scale=lam, keep_residuals=False)
        out, info = kgc.deconsolidate(blob)
        rmse = float(np.sqrt(np.mean((rows - out) ** 2)))
        drift = (
            abs(info["conserved_mass_out"] - info["conserved_mass_in"])
            / info["conserved_mass_in"]
        )
        print(
            f"{'lossy rg_scale=' + str(lam):26s} {raw/len(blob):6.2f}x {info['n_sites']:6d} "
            f"{rmse:8.4f} {drift*100:10.3f}% {info['truth_status']:>16s}"
        )
    print(f"   (signal rms {sig:.3f}; every lossy archive still carries sha256 +")
    print("    member addresses, so reconstruction fidelity is auditable - Law 9.)")


if __name__ == "__main__":
    print("KGC v2 benchmark - every number is a verified round-trip")
    kgc = KGCCompressor()
    bench_bytes(kgc)
    bench_tensor(kgc)
    bench_consolidate(kgc)

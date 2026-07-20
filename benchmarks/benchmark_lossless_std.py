"""
Lossless compression: KGC BYTES regime vs the industry-standard codecs.

Standard corpus (enwik8, the Large Text Compression Benchmark's file),
standard metric (bits/byte and ratio), standard competitors:

    gzip / zlib   (DEFLATE, level 9)
    bzip2         (BWT, level 9)
    xz / LZMA     (level 9 - the 7-Zip engine)
    zstd          (level 19)
    brotli        (level 11)
    KGC bytes     (this repo: order-1 / order-2 adaptive range coder)

This benchmarks the repo's WEAKEST regime against specialists - the
BYTES path is a verified debt container, not an LZ replacement, and it
is expected to lose on generic text. Run it anyway: honest positioning
is the point, and the number quantifies exactly how far the general-
purpose byte path is from the state of the art.

Usage:  python benchmark_lossless_std.py [path_to_corpus] [n_megabytes]
truth_status: standard_corpus_lossless_measurement
"""

import bz2
import gzip
import lzma
import sys
import time
import zlib
from pathlib import Path

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


def codecs():
    c = {
        "gzip -9": (lambda d: gzip.compress(d, 9), None),
        "zlib -9": (lambda d: zlib.compress(d, 9), None),
        "bzip2 -9": (lambda d: bz2.compress(d, 9), None),
        "xz/LZMA -9": (lambda d: lzma.compress(d, preset=9 | lzma.PRESET_EXTREME), None),
    }
    try:
        import zstandard as zstd

        c["zstd -19"] = (lambda d: zstd.ZstdCompressor(level=19).compress(d), None)
    except ImportError:
        pass
    try:
        import brotli

        c["brotli -11"] = (lambda d: brotli.compress(d, quality=11), None)
    except ImportError:
        pass
    return c


def main():
    corpus = sys.argv[1] if len(sys.argv) > 1 else "enwik8"
    n_mb = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    n_bytes = int(n_mb * 1024 * 1024)

    data = Path(corpus).read_bytes()[:n_bytes]
    n = len(data)
    print(f"corpus: {corpus}  slice: {n/1e6:.2f} MB ({n} bytes)\n")
    print(f"{'codec':16s} {'ratio':>8s} {'bits/byte':>10s} {'enc MB/s':>9s}")

    rows = {}
    for name, (comp, _) in codecs().items():
        t0 = time.time()
        out = comp(data)
        dt = time.time() - t0
        ratio = n / len(out)
        bpb = len(out) * 8 / n
        rows[name] = (ratio, bpb)
        print(f"{name:16s} {ratio:8.2f} {bpb:10.4f} {n/1e6/dt:9.1f}")

    # KGC byte regime - pure-Python range coder, slower; use modest slice
    kgc = KGCCompressor()
    for mode in ("fast", "strong"):
        t0 = time.time()
        blob = kgc.compress(data, mode=mode)
        out, info = kgc.decompress(blob)
        dt = time.time() - t0
        assert out == data and info["verified"], "KGC round-trip failed!"
        ratio = n / len(blob)
        bpb = len(blob) * 8 / n
        rows[f"KGC {mode}"] = (ratio, bpb)
        print(f"{'KGC ' + mode:16s} {ratio:8.2f} {bpb:10.4f} {n/1e6/dt:9.2f}")

    best = min(rows.items(), key=lambda kv: kv[1][1])
    kgc_best = min((v[1] for k, v in rows.items() if k.startswith("KGC")), default=None)
    print(f"\nbest bits/byte: {best[0]} @ {best[1][1]:.4f}")
    if kgc_best is not None:
        print(f"KGC best: {kgc_best:.4f} bits/byte  ({kgc_best/best[1][1]:.2f}x the leader's size)")
    print("(KGC byte regime is a verified debt container, not an LZ codec - "
          "its real strength is lattice VQ, see benchmark_faiss_vq.py)")


if __name__ == "__main__":
    main()

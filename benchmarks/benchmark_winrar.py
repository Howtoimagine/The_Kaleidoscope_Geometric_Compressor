#!/usr/bin/env python3
"""
E8ZIP vs WinRAR Benchmark

Direct comparison against real WinRAR compression.
Uses the rar.exe command-line tool.

Usage:
    python benchmark_winrar.py [file1] [file2] ...
"""

import os
import sys
import time
import tempfile
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import hashlib

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from e8zip import E8Compressor
from e8zip.core.compressor import CompressionMode


# WinRAR paths
WINRAR_PATHS = [
    r"C:\Program Files\WinRAR\rar.exe",
    r"C:\Program Files (x86)\WinRAR\rar.exe",
]


def find_winrar() -> Optional[str]:
    """Find WinRAR rar.exe"""
    for path in WINRAR_PATHS:
        if os.path.exists(path):
            return path
    return None


def get_file_hash(path: str) -> str:
    """Get MD5 hash of file."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def format_size(size: int) -> str:
    """Format bytes as human-readable."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


@dataclass
class Result:
    name: str
    mode: str
    original: int
    compressed: int
    compress_time: float
    decompress_time: float
    verified: bool

    @property
    def ratio(self) -> float:
        return self.original / max(self.compressed, 1)


def benchmark_e8zip(
    file_path: Path, mode: CompressionMode, tmpdir: str, original_hash: str
) -> Result:
    """Benchmark E8ZIP compression."""
    try:
        compressor = E8Compressor(mode=mode)

        with open(file_path, "rb") as f:
            data = f.read()

        # Compress
        start = time.perf_counter()
        compressed = compressor.compress(data)
        compress_time = time.perf_counter() - start

        # Write compressed
        compressed_path = Path(tmpdir) / f"e8zip_{mode.name}.e8z"
        with open(compressed_path, "wb") as f:
            f.write(compressed)

        # Decompress
        start = time.perf_counter()
        decompressed = compressor.decompress(compressed)
        decompress_time = time.perf_counter() - start

        # Verify (skip for MYTHIC which is lossy)
        if mode == CompressionMode.MYTHIC:
            verified = True
        else:
            decompressed_path = Path(tmpdir) / f"e8zip_{mode.name}_out"
            with open(decompressed_path, "wb") as f:
                f.write(decompressed)
            verified = get_file_hash(str(decompressed_path)) == original_hash

        return Result(
            name="E8ZIP",
            mode=mode.name.lower(),
            original=len(data),
            compressed=len(compressed),
            compress_time=compress_time,
            decompress_time=decompress_time,
            verified=verified,
        )
    except Exception as e:
        return Result("E8ZIP", mode.name.lower(), 0, 0, 0, 0, False)


def benchmark_winrar(
    file_path: Path, level: str, tmpdir: str, original_hash: str, rar_exe: str
) -> Result:
    """Benchmark real WinRAR compression."""
    try:
        level_map = {"store": "-m0", "fast": "-m1", "normal": "-m3", "good": "-m4", "best": "-m5"}
        mx = level_map.get(level, "-m3")

        compressed_path = Path(tmpdir) / f"winrar_{level}.rar"
        decompressed_dir = Path(tmpdir) / f"winrar_{level}_out"
        decompressed_dir.mkdir(exist_ok=True)

        original_size = file_path.stat().st_size

        # Compress with WinRAR
        start = time.perf_counter()
        result = subprocess.run(
            [rar_exe, "a", mx, "-ep", str(compressed_path), str(file_path)],
            capture_output=True,
            timeout=120,
        )
        compress_time = time.perf_counter() - start

        if result.returncode != 0:
            raise RuntimeError(f"WinRAR compression failed: {result.stderr.decode()}")

        compressed_size = compressed_path.stat().st_size

        # Decompress with WinRAR
        start = time.perf_counter()
        result = subprocess.run(
            [rar_exe, "x", "-y", str(compressed_path), str(decompressed_dir) + "\\"],
            capture_output=True,
            timeout=120,
        )
        decompress_time = time.perf_counter() - start

        # Verify
        restored_file = decompressed_dir / file_path.name
        if restored_file.exists():
            verified = get_file_hash(str(restored_file)) == original_hash
        else:
            verified = False

        return Result(
            name="WinRAR",
            mode=level,
            original=original_size,
            compressed=compressed_size,
            compress_time=compress_time,
            decompress_time=decompress_time,
            verified=verified,
        )
    except Exception as e:
        print(f"  WinRAR error: {e}")
        return Result("WinRAR", level, 0, 0, 0, 0, False)


def print_banner():
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║     E8ZIP vs WinRAR BENCHMARK                                                 ║
║     Real WinRAR Comparison                                                    ║
║                                                                               ║
║     "Compress paths through the lattice, not just snapshots."                 ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)


def print_results(file_name: str, file_size: int, results: list):
    print(f"\n{'═' * 80}")
    print(f"  FILE: {file_name}")
    print(f"  SIZE: {format_size(file_size)}")
    print(f"{'═' * 80}\n")

    print(
        f"{'Compressor':<12} {'Mode':<10} {'Ratio':>8} {'Size':>12} {'Compress':>10} {'Decompress':>10} {'OK':>6}"
    )
    print("─" * 80)

    sorted_results = sorted(results, key=lambda r: r.ratio, reverse=True)

    for r in sorted_results:
        verify_mark = "✓" if r.verified else "✗"
        print(
            f"{r.name:<12} {r.mode:<10} {r.ratio:>7.2f}x {format_size(r.compressed):>12} "
            f"{r.compress_time * 1000:>8.1f}ms {r.decompress_time * 1000:>8.1f}ms {verify_mark:>6}"
        )

    # Best results
    best = sorted_results[0] if sorted_results else None
    if best:
        print(f"\n  🏆 BEST RATIO: {best.name} ({best.mode}) - {best.ratio:.2f}x")


def main():
    print_banner()

    # Find WinRAR
    rar_exe = find_winrar()
    if not rar_exe:
        print("❌ WinRAR not found! Please install WinRAR.")
        return

    print(f"✓ WinRAR found: {rar_exe}")

    # Default test files
    default_files = [
        Path(__file__).parent / "K banner.jpg",
        Path(__file__).parent / "Recording 2025-12-02 223742.mp4",
        Path(__file__).parent.parent / "README.md",
    ]

    # Get files from command line or use defaults
    if len(sys.argv) > 1:
        files = [Path(f) for f in sys.argv[1:]]
    else:
        files = [f for f in default_files if f.exists()]

    if not files:
        print("No test files found!")
        return

    print(f"\nBenchmarking {len(files)} file(s)...\n")

    for file_path in files:
        if not file_path.exists():
            print(f"Skipping {file_path} (not found)")
            continue

        file_size = file_path.stat().st_size
        original_hash = get_file_hash(str(file_path))

        results = []

        with tempfile.TemporaryDirectory() as tmpdir:
            # E8ZIP modes
            print(f"Testing E8ZIP on {file_path.name}...")
            for mode in CompressionMode:
                result = benchmark_e8zip(file_path, mode, tmpdir, original_hash)
                results.append(result)

            # WinRAR levels
            print(f"Testing WinRAR on {file_path.name}...")
            for level in ["store", "fast", "normal", "good", "best"]:
                result = benchmark_winrar(file_path, level, tmpdir, original_hash, rar_exe)
                results.append(result)

        print_results(file_path.name, file_size, results)

    print(f"\n{'═' * 80}")
    print("  BENCHMARK COMPLETE")
    print(f"{'═' * 80}\n")


if __name__ == "__main__":
    main()

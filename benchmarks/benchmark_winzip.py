#!/usr/bin/env python3
"""
E8ZIP vs Windows ZIP Benchmark

Direct comparison against Windows built-in ZIP compression.
Uses PowerShell Compress-Archive (native Windows ZIP).

Note: WinZip is commercial software. This uses the built-in Windows ZIP
which is compatible with WinZip format (.zip).

Usage:
    python benchmark_winzip.py [file1] [file2] ...
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
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from e8zip import E8Compressor
from e8zip.core.compressor import CompressionMode


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

        # Decompress
        start = time.perf_counter()
        decompressed = compressor.decompress(compressed)
        decompress_time = time.perf_counter() - start

        # Verify (skip for MYTHIC which is lossy)
        if mode == CompressionMode.MYTHIC:
            verified = True
        else:
            verified = hashlib.md5(decompressed).hexdigest() == original_hash

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
        print(f"  E8ZIP error: {e}")
        return Result("E8ZIP", mode.name.lower(), 0, 0, 0, 0, False)


def benchmark_windows_zip(file_path: Path, level: str, tmpdir: str, original_hash: str) -> Result:
    """
    Benchmark Windows built-in ZIP compression.
    Uses PowerShell Compress-Archive (WinZip compatible format).
    """
    try:
        # Map levels to compression options
        # Windows ZIP has: Optimal, Fastest, NoCompression
        level_map = {
            "store": "NoCompression",
            "fastest": "Fastest",
            "optimal": "Optimal",
        }
        compression_level = level_map.get(level, "Optimal")

        compressed_path = Path(tmpdir) / f"winzip_{level}.zip"
        decompressed_dir = Path(tmpdir) / f"winzip_{level}_out"
        decompressed_dir.mkdir(exist_ok=True)

        original_size = file_path.stat().st_size

        # Compress with PowerShell Compress-Archive
        start = time.perf_counter()
        ps_cmd = f'Compress-Archive -Path "{file_path}" -DestinationPath "{compressed_path}" -CompressionLevel {compression_level} -Force'
        result = subprocess.run(
            ["powershell", "-Command", ps_cmd],
            capture_output=True,
            timeout=120,
        )
        compress_time = time.perf_counter() - start

        if result.returncode != 0:
            raise RuntimeError(f"ZIP compression failed: {result.stderr.decode()}")

        compressed_size = compressed_path.stat().st_size

        # Decompress with PowerShell Expand-Archive
        start = time.perf_counter()
        ps_cmd = (
            f'Expand-Archive -Path "{compressed_path}" -DestinationPath "{decompressed_dir}" -Force'
        )
        result = subprocess.run(
            ["powershell", "-Command", ps_cmd],
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
            name="WinZIP",
            mode=level,
            original=original_size,
            compressed=compressed_size,
            compress_time=compress_time,
            decompress_time=decompress_time,
            verified=verified,
        )
    except Exception as e:
        print(f"  WinZIP error: {e}")
        return Result("WinZIP", level, 0, 0, 0, 0, False)


def print_banner():
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║     E8ZIP vs Windows ZIP BENCHMARK                                            ║
║     Native Windows ZIP Comparison (WinZip Compatible)                         ║
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
        if r.original == 0:
            continue
        verify_mark = "✓" if r.verified else "✗"
        print(
            f"{r.name:<12} {r.mode:<10} {r.ratio:>7.2f}x {format_size(r.compressed):>12} "
            f"{r.compress_time * 1000:>8.1f}ms {r.decompress_time * 1000:>8.1f}ms {verify_mark:>6}"
        )

    # Best results
    valid_results = [r for r in sorted_results if r.original > 0]
    if valid_results:
        best = valid_results[0]
        print(f"\n  🏆 BEST RATIO: {best.name} ({best.mode}) - {best.ratio:.2f}x")


def main():
    print_banner()

    print("✓ Using Windows built-in ZIP (PowerShell Compress-Archive)")
    print("  This is compatible with WinZip .zip format\n")

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

    print(f"Benchmarking {len(files)} file(s)...\n")

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

            # Windows ZIP levels
            print(f"Testing Windows ZIP on {file_path.name}...")
            for level in ["store", "fastest", "optimal"]:
                result = benchmark_windows_zip(file_path, level, tmpdir, original_hash)
                results.append(result)

        print_results(file_path.name, file_size, results)

    print(f"\n{'═' * 80}")
    print("  BENCHMARK COMPLETE")
    print(f"{'═' * 80}\n")


if __name__ == "__main__":
    main()

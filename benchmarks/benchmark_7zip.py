#!/usr/bin/env python3
"""
E8ZIP vs 7-Zip Benchmark

Direct comparison against real 7-Zip compression.
Uses the 7z.exe command-line tool.

Usage:
    python benchmark_7zip.py [file1] [file2] ...
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


# 7-Zip paths
SEVENZIP_PATHS = [
    r"C:\Program Files\7-Zip\7z.exe",
    r"C:\Program Files (x86)\7-Zip\7z.exe",
]


def find_7zip() -> Optional[str]:
    """Find 7-Zip 7z.exe"""
    for path in SEVENZIP_PATHS:
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


def benchmark_7zip(
    file_path: Path, level: str, format_type: str, tmpdir: str, original_hash: str, exe: str
) -> Result:
    """
    Benchmark 7-Zip compression.

    Args:
        format_type: "7z" for 7z format, "zip" for zip format
    """
    try:
        # Compression levels: 0=Store, 1=Fastest, 5=Normal, 9=Ultra
        level_map = {
            "store": "0",
            "fastest": "1",
            "fast": "3",
            "normal": "5",
            "maximum": "7",
            "ultra": "9",
        }
        mx = level_map.get(level, "5")

        ext = "7z" if format_type == "7z" else "zip"
        compressed_path = Path(tmpdir) / f"7zip_{level}_{format_type}.{ext}"
        decompressed_dir = Path(tmpdir) / f"7zip_{level}_{format_type}_out"
        decompressed_dir.mkdir(exist_ok=True)

        original_size = file_path.stat().st_size

        # Compress with 7-Zip
        start = time.perf_counter()
        cmd = [exe, "a", f"-t{format_type}", f"-mx={mx}", str(compressed_path), str(file_path)]
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        compress_time = time.perf_counter() - start

        if result.returncode != 0:
            raise RuntimeError(f"7-Zip compression failed: {result.stderr.decode()}")

        compressed_size = compressed_path.stat().st_size

        # Decompress with 7-Zip
        start = time.perf_counter()
        cmd = [exe, "x", "-y", f"-o{decompressed_dir}", str(compressed_path)]
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        decompress_time = time.perf_counter() - start

        # Verify
        restored_file = decompressed_dir / file_path.name
        if restored_file.exists():
            verified = get_file_hash(str(restored_file)) == original_hash
        else:
            verified = False

        mode_str = f"{format_type}-{level}"
        return Result(
            name="7-Zip",
            mode=mode_str,
            original=original_size,
            compressed=compressed_size,
            compress_time=compress_time,
            decompress_time=decompress_time,
            verified=verified,
        )
    except Exception as e:
        print(f"  7-Zip error: {e}")
        return Result("7-Zip", f"{format_type}-{level}", 0, 0, 0, 0, False)


def print_banner():
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║     E8ZIP vs 7-Zip BENCHMARK                                                  ║
║     Real 7-Zip Comparison (7z and zip formats)                                ║
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
        f"{'Compressor':<12} {'Mode':<12} {'Ratio':>8} {'Size':>12} {'Compress':>10} {'Decompress':>10} {'OK':>6}"
    )
    print("─" * 80)

    sorted_results = sorted(results, key=lambda r: r.ratio, reverse=True)

    for r in sorted_results:
        if r.original == 0:
            continue
        verify_mark = "✓" if r.verified else "✗"
        print(
            f"{r.name:<12} {r.mode:<12} {r.ratio:>7.2f}x {format_size(r.compressed):>12} "
            f"{r.compress_time * 1000:>8.1f}ms {r.decompress_time * 1000:>8.1f}ms {verify_mark:>6}"
        )

    # Best results
    valid_results = [r for r in sorted_results if r.original > 0]
    if valid_results:
        best = valid_results[0]
        print(f"\n  🏆 BEST RATIO: {best.name} ({best.mode}) - {best.ratio:.2f}x")


def main():
    print_banner()

    # Find 7-Zip
    exe = find_7zip()
    if not exe:
        print("❌ 7-Zip not found!")
        print("   Please install 7-Zip from: https://www.7-zip.org/")
        print("   Or use benchmark_winzip.py for Windows built-in ZIP")
        return

    print(f"✓ 7-Zip found: {exe}\n")

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

            # 7-Zip with 7z format (best compression)
            print(f"Testing 7-Zip (7z format) on {file_path.name}...")
            for level in ["store", "fast", "normal", "ultra"]:
                result = benchmark_7zip(file_path, level, "7z", tmpdir, original_hash, exe)
                results.append(result)

            # 7-Zip with zip format
            print(f"Testing 7-Zip (zip format) on {file_path.name}...")
            for level in ["store", "fast", "normal"]:
                result = benchmark_7zip(file_path, level, "zip", tmpdir, original_hash, exe)
                results.append(result)

        print_results(file_path.name, file_size, results)

    print(f"\n{'═' * 80}")
    print("  BENCHMARK COMPLETE")
    print(f"{'═' * 80}\n")


if __name__ == "__main__":
    main()

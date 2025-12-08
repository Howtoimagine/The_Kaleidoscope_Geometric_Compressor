#!/usr/bin/env python3
"""
E8ZIP Benchmark Suite

Compares E8ZIP compression against WinRAR and WinZip (7-Zip as proxy).
Measures compression ratio, speed, and decompression performance.

Requirements:
- 7-Zip installed (7z.exe in PATH) - proxy for WinZip
- WinRAR installed (rar.exe in PATH) - optional
- Python 3.9+

Usage:
    python benchmark.py                    # Run all benchmarks
    python benchmark.py --file myfile.txt  # Benchmark specific file
    python benchmark.py --generate         # Generate test files first
"""

import os
import sys
import time
import shutil
import tempfile
import subprocess
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import hashlib

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from e8zip import E8Compressor
from e8zip.core.compressor import CompressionMode


@dataclass
class BenchmarkResult:
    """Result of a single compression benchmark."""

    compressor: str
    mode: str
    original_size: int
    compressed_size: int
    compress_time: float
    decompress_time: float
    verified: bool
    error: Optional[str] = None

    @property
    def ratio(self) -> float:
        if self.compressed_size == 0:
            return 0.0
        return self.original_size / self.compressed_size

    @property
    def compress_speed(self) -> float:
        """KB/s"""
        if self.compress_time == 0:
            return float("inf")
        return (self.original_size / 1024) / self.compress_time

    @property
    def decompress_speed(self) -> float:
        """KB/s"""
        if self.decompress_time == 0:
            return float("inf")
        return (self.original_size / 1024) / self.decompress_time


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""

    file_name: str
    file_size: int
    file_type: str
    results: List[BenchmarkResult] = field(default_factory=list)

    def add(self, result: BenchmarkResult):
        self.results.append(result)

    def print_table(self):
        """Print results as formatted table."""
        print(f"\n{'=' * 80}")
        print(f"  BENCHMARK: {self.file_name}")
        print(f"  Size: {format_size(self.file_size)} | Type: {self.file_type}")
        print(f"{'=' * 80}\n")

        # Header
        print(
            f"{'Compressor':<15} {'Mode':<10} {'Ratio':>8} {'Size':>12} "
            f"{'Compress':>10} {'Decompress':>10} {'Verified':>8}"
        )
        print("-" * 80)

        # Sort by ratio (best first)
        sorted_results = sorted(self.results, key=lambda r: r.ratio, reverse=True)

        for r in sorted_results:
            if r.error:
                print(f"{r.compressor:<15} {r.mode:<10} {'ERROR':>8} {r.error}")
            else:
                print(
                    f"{r.compressor:<15} {r.mode:<10} {r.ratio:>7.2f}x "
                    f"{format_size(r.compressed_size):>12} "
                    f"{r.compress_time * 1000:>8.1f}ms "
                    f"{r.decompress_time * 1000:>8.1f}ms "
                    f"{'✓' if r.verified else '✗':>8}"
                )

        print()

        # Winner
        if sorted_results and not sorted_results[0].error:
            winner = sorted_results[0]
            print(f"  🏆 BEST RATIO: {winner.compressor} ({winner.mode}) - {winner.ratio:.2f}x")

        # Fastest compression
        fastest = min(
            [r for r in self.results if not r.error], key=lambda r: r.compress_time, default=None
        )
        if fastest:
            print(
                f"  ⚡ FASTEST: {fastest.compressor} ({fastest.mode}) - {fastest.compress_time * 1000:.1f}ms"
            )

        print()


def format_size(size: int) -> str:
    """Format bytes as human-readable size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


def get_file_hash(path: str) -> str:
    """Get MD5 hash of file."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def find_7zip() -> Optional[str]:
    """Find 7-Zip executable."""
    paths = [
        r"C:\Program Files\7-Zip\7z.exe",
        r"C:\Program Files (x86)\7-Zip\7z.exe",
        "7z",
        "7z.exe",
    ]
    for p in paths:
        if os.path.exists(p):
            return p
        if shutil.which(p):
            return shutil.which(p)
    return None


def find_winrar() -> Optional[str]:
    """Find WinRAR executable."""
    paths = [
        r"C:\Program Files\WinRAR\rar.exe",
        r"C:\Program Files (x86)\WinRAR\rar.exe",
        "rar",
        "rar.exe",
    ]
    for p in paths:
        if os.path.exists(p):
            return p
        if shutil.which(p):
            return shutil.which(p)
    return None


class Benchmarker:
    """Run compression benchmarks."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.zip_exe = find_7zip()
        self.rar_exe = find_winrar()

        if verbose:
            print(f"7-Zip: {self.zip_exe or 'NOT FOUND'}")
            print(f"WinRAR: {self.rar_exe or 'NOT FOUND'}")

    def benchmark_file(self, file_path: str) -> BenchmarkSuite:
        """Run all benchmarks on a single file."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_size = file_path.stat().st_size
        file_type = self._detect_file_type(file_path)

        suite = BenchmarkSuite(file_name=file_path.name, file_size=file_size, file_type=file_type)

        original_hash = get_file_hash(str(file_path))

        with tempfile.TemporaryDirectory() as tmpdir:
            # E8ZIP benchmarks (all modes)
            for mode in CompressionMode:
                result = self._benchmark_e8zip(file_path, tmpdir, mode, original_hash)
                suite.add(result)

            # 7-Zip benchmarks (proxy for WinZip)
            if self.zip_exe:
                for level in ["fast", "normal", "ultra"]:
                    result = self._benchmark_7zip(file_path, tmpdir, level, original_hash)
                    suite.add(result)

            # WinRAR benchmarks
            if self.rar_exe:
                for level in ["fast", "normal", "best"]:
                    result = self._benchmark_winrar(file_path, tmpdir, level, original_hash)
                    suite.add(result)

        return suite

    def _detect_file_type(self, path: Path) -> str:
        """Detect file type for reporting."""
        ext = path.suffix.lower()
        type_map = {
            ".txt": "Text",
            ".md": "Markdown",
            ".py": "Python",
            ".js": "JavaScript",
            ".json": "JSON",
            ".xml": "XML",
            ".html": "HTML",
            ".csv": "CSV",
            ".log": "Log",
            ".pdf": "PDF",
            ".jpg": "JPEG Image",
            ".png": "PNG Image",
            ".gif": "GIF Image",
            ".mp3": "MP3 Audio",
            ".mp4": "MP4 Video",
            ".zip": "ZIP Archive",
            ".exe": "Executable",
            ".dll": "Library",
        }
        return type_map.get(ext, f"Binary ({ext})")

    def _benchmark_e8zip(
        self, file_path: Path, tmpdir: str, mode: CompressionMode, original_hash: str
    ) -> BenchmarkResult:
        """Benchmark E8ZIP compression."""
        try:
            compressor = E8Compressor(mode=mode)

            compressed_path = Path(tmpdir) / f"e8zip_{mode.name}.e8z"
            decompressed_path = Path(tmpdir) / f"e8zip_{mode.name}_restored"

            # Compress
            start = time.perf_counter()
            with open(file_path, "rb") as f:
                data = f.read()
            compressed = compressor.compress(data)
            with open(compressed_path, "wb") as f:
                f.write(compressed)
            compress_time = time.perf_counter() - start

            # Decompress
            start = time.perf_counter()
            with open(compressed_path, "rb") as f:
                compressed_data = f.read()
            decompressed = compressor.decompress(compressed_data)
            with open(decompressed_path, "wb") as f:
                f.write(decompressed)
            decompress_time = time.perf_counter() - start

            # Verify
            restored_hash = get_file_hash(str(decompressed_path))
            verified = (original_hash == restored_hash) if mode != CompressionMode.MYTHIC else True

            return BenchmarkResult(
                compressor="E8ZIP",
                mode=mode.name.lower(),
                original_size=len(data),
                compressed_size=len(compressed),
                compress_time=compress_time,
                decompress_time=decompress_time,
                verified=verified,
            )

        except Exception as e:
            return BenchmarkResult(
                compressor="E8ZIP",
                mode=mode.name.lower(),
                original_size=0,
                compressed_size=0,
                compress_time=0,
                decompress_time=0,
                verified=False,
                error=str(e),
            )

    def _benchmark_7zip(
        self, file_path: Path, tmpdir: str, level: str, original_hash: str
    ) -> BenchmarkResult:
        """Benchmark 7-Zip compression."""
        try:
            level_map = {"fast": "1", "normal": "5", "ultra": "9"}
            mx = level_map[level]

            compressed_path = Path(tmpdir) / f"7zip_{level}.zip"
            decompressed_dir = Path(tmpdir) / f"7zip_{level}_out"
            decompressed_dir.mkdir(exist_ok=True)

            original_size = file_path.stat().st_size

            # Compress
            start = time.perf_counter()
            result = subprocess.run(
                [self.zip_exe, "a", f"-mx={mx}", str(compressed_path), str(file_path)],
                capture_output=True,
                timeout=60,
            )
            compress_time = time.perf_counter() - start

            if result.returncode != 0:
                raise RuntimeError(f"7-Zip compression failed: {result.stderr.decode()}")

            compressed_size = compressed_path.stat().st_size

            # Decompress
            start = time.perf_counter()
            result = subprocess.run(
                [self.zip_exe, "x", "-y", f"-o{decompressed_dir}", str(compressed_path)],
                capture_output=True,
                timeout=60,
            )
            decompress_time = time.perf_counter() - start

            # Verify
            restored_file = decompressed_dir / file_path.name
            if restored_file.exists():
                restored_hash = get_file_hash(str(restored_file))
                verified = original_hash == restored_hash
            else:
                verified = False

            return BenchmarkResult(
                compressor="7-Zip",
                mode=level,
                original_size=original_size,
                compressed_size=compressed_size,
                compress_time=compress_time,
                decompress_time=decompress_time,
                verified=verified,
            )

        except Exception as e:
            return BenchmarkResult(
                compressor="7-Zip",
                mode=level,
                original_size=0,
                compressed_size=0,
                compress_time=0,
                decompress_time=0,
                verified=False,
                error=str(e),
            )

    def _benchmark_winrar(
        self, file_path: Path, tmpdir: str, level: str, original_hash: str
    ) -> BenchmarkResult:
        """Benchmark WinRAR compression."""
        try:
            level_map = {"fast": "-m1", "normal": "-m3", "best": "-m5"}
            mx = level_map[level]

            compressed_path = Path(tmpdir) / f"winrar_{level}.rar"
            decompressed_dir = Path(tmpdir) / f"winrar_{level}_out"
            decompressed_dir.mkdir(exist_ok=True)

            original_size = file_path.stat().st_size

            # Compress
            start = time.perf_counter()
            result = subprocess.run(
                [self.rar_exe, "a", mx, str(compressed_path), str(file_path)],
                capture_output=True,
                timeout=60,
            )
            compress_time = time.perf_counter() - start

            if result.returncode != 0:
                raise RuntimeError(f"WinRAR compression failed")

            compressed_size = compressed_path.stat().st_size

            # Decompress
            start = time.perf_counter()
            result = subprocess.run(
                [self.rar_exe, "x", "-y", str(compressed_path), str(decompressed_dir) + "\\"],
                capture_output=True,
                timeout=60,
            )
            decompress_time = time.perf_counter() - start

            # Verify
            restored_file = decompressed_dir / file_path.name
            if restored_file.exists():
                restored_hash = get_file_hash(str(restored_file))
                verified = original_hash == restored_hash
            else:
                verified = False

            return BenchmarkResult(
                compressor="WinRAR",
                mode=level,
                original_size=original_size,
                compressed_size=compressed_size,
                compress_time=compress_time,
                decompress_time=decompress_time,
                verified=verified,
            )

        except Exception as e:
            return BenchmarkResult(
                compressor="WinRAR",
                mode=level,
                original_size=0,
                compressed_size=0,
                compress_time=0,
                decompress_time=0,
                verified=False,
                error=str(e),
            )


def generate_test_files(output_dir: str) -> List[str]:
    """Generate test files for benchmarking."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files = []

    # 1. Repetitive text (should compress well)
    text_file = output_path / "repetitive_text.txt"
    with open(text_file, "w") as f:
        f.write("The quick brown fox jumps over the lazy dog. " * 10000)
    files.append(str(text_file))
    print(f"  Created: {text_file.name} ({format_size(text_file.stat().st_size)})")

    # 2. Random binary data (hard to compress)
    import random

    random_file = output_path / "random_binary.bin"
    with open(random_file, "wb") as f:
        f.write(bytes(random.getrandbits(8) for _ in range(100000)))
    files.append(str(random_file))
    print(f"  Created: {random_file.name} ({format_size(random_file.stat().st_size)})")

    # 3. Source code (mixed content)
    code_file = output_path / "sample_code.py"
    with open(code_file, "w") as f:
        for i in range(500):
            f.write(f'''
def function_{i}(x, y, z):
    """Docstring for function {i}."""
    result = x * y + z
    for j in range({i % 10}):
        result += j * 2
    return result

class Class_{i}:
    def __init__(self):
        self.value = {i}
    
    def method(self):
        return self.value * 2
''')
    files.append(str(code_file))
    print(f"  Created: {code_file.name} ({format_size(code_file.stat().st_size)})")

    # 4. JSON data
    json_file = output_path / "sample_data.json"
    import json

    data = {
        "users": [
            {
                "id": i,
                "name": f"User {i}",
                "email": f"user{i}@example.com",
                "data": {"score": i * 10, "level": i % 5},
            }
            for i in range(1000)
        ]
    }
    with open(json_file, "w") as f:
        json.dump(data, f, indent=2)
    files.append(str(json_file))
    print(f"  Created: {json_file.name} ({format_size(json_file.stat().st_size)})")

    # 5. Log file (realistic server logs)
    log_file = output_path / "server.log"
    with open(log_file, "w") as f:
        for i in range(5000):
            level = ["INFO", "DEBUG", "WARNING", "ERROR"][i % 4]
            f.write(
                f"2025-12-08 12:{i // 60:02d}:{i % 60:02d} [{level}] "
                f"Request {i}: GET /api/resource/{i % 100} - 200 OK - {i * 10}ms\n"
            )
    files.append(str(log_file))
    print(f"  Created: {log_file.name} ({format_size(log_file.stat().st_size)})")

    return files


def print_banner():
    """Print benchmark banner."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║     E8ZIP BENCHMARK SUITE                                                     ║
║     Comparing E8ZIP vs WinZip/7-Zip vs WinRAR                                 ║
║                                                                               ║
║     "Compress paths through the lattice, not just snapshots."                 ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
""")


def main():
    parser = argparse.ArgumentParser(description="E8ZIP Benchmark Suite")
    parser.add_argument("--file", "-f", help="Specific file to benchmark")
    parser.add_argument("--generate", "-g", action="store_true", help="Generate test files")
    parser.add_argument(
        "--output", "-o", default="benchmark_files", help="Output directory for generated files"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    print_banner()

    benchmarker = Benchmarker(verbose=args.verbose)

    if args.generate:
        print("Generating test files...\n")
        files = generate_test_files(args.output)
        print(f"\nGenerated {len(files)} test files in {args.output}/")

        print("\nRunning benchmarks on generated files...\n")
        for file_path in files:
            suite = benchmarker.benchmark_file(file_path)
            suite.print_table()

    elif args.file:
        print(f"Benchmarking: {args.file}\n")
        suite = benchmarker.benchmark_file(args.file)
        suite.print_table()

    else:
        # Default: benchmark some files from the project
        print("Running default benchmarks...\n")

        project_root = Path(__file__).parent.parent.parent
        test_files = [
            project_root / "ARCHITECTURE.md",
            project_root / "README.md",
            project_root / "e8_mind_server" / "app.py",
        ]

        for file_path in test_files:
            if file_path.exists():
                suite = benchmarker.benchmark_file(str(file_path))
                suite.print_table()
            else:
                print(f"Skipping (not found): {file_path}")

    print("\n" + "=" * 80)
    print("  BENCHMARK COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

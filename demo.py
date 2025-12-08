#!/usr/bin/env python3
"""
E8ZIP Demo Script

Demonstrates the E8 Geometric Lattice Compression library.
Run this script to see the compression in action.
"""

import os
import sys
import tempfile
import time

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from e8zip import E8Compressor, E8GeometricCodec, TrajectoryCompressor
from e8zip.core.compressor import CompressionMode
from e8zip.core.e8_lattice import E8Lattice
from e8zip.core.black_hole import BlackHoleCompressor


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def demo_e8_lattice():
    """Demonstrate E8 lattice properties."""
    print_header("E8 LATTICE DEMONSTRATION")

    lattice = E8Lattice()
    roots = lattice.roots

    print(f"E8 Lattice Properties:")
    print(f"  - Number of roots: {len(roots)}")
    print(f"  - Dimension: 8")
    print(f"  - Root norms: {np.linalg.norm(roots[0]):.4f} (all roots)")

    # Show some sample roots
    print(f"\n  Sample roots:")
    for i in range(3):
        print(f"    Root {i}: {roots[i]}")

    # Quantization demo
    test_vector = np.array([0.7, 0.8, 0.1, -0.2, 0.5, 0.5, 0.5, 0.5])
    print(f"\n  Test vector: {test_vector}")

    idx, nearest, distance = lattice.nearest_lattice_point(test_vector)
    print(f"  Nearest lattice point: {nearest}")
    print(f"  Distance: {distance:.4f}")


def demo_compression_modes():
    """Demonstrate different compression modes."""
    print_header("COMPRESSION MODES COMPARISON")

    # Create test data
    test_data = b"The E8 lattice is a beautiful 8-dimensional structure. " * 100
    print(f"Original data size: {len(test_data)} bytes\n")

    results = []

    for mode in CompressionMode:
        compressor = E8Compressor(mode=mode)

        start = time.time()
        compressed = compressor.compress(test_data)
        compress_time = time.time() - start

        ratio = len(test_data) / len(compressed)

        # Only decompress lossless modes
        if mode != CompressionMode.MYTHIC:
            start = time.time()
            decompressed = compressor.decompress(compressed)
            decompress_time = time.time() - start
            lossless = decompressed == test_data
        else:
            decompress_time = 0
            lossless = False  # MYTHIC is lossy

        results.append(
            {
                "mode": mode.name,
                "compressed_size": len(compressed),
                "ratio": ratio,
                "compress_time": compress_time,
                "decompress_time": decompress_time,
                "lossless": lossless,
            }
        )

        print(
            f"{mode.name:10s}: {len(compressed):6d} bytes | "
            f"Ratio: {ratio:5.2f}x | "
            f"Time: {compress_time * 1000:6.2f}ms | "
            f"Lossless: {'✓' if lossless else '✗'}"
        )


def demo_trajectory_compression():
    """Demonstrate trajectory compression."""
    print_header("TRAJECTORY COMPRESSION")

    # Create a trajectory (sequence of 8D vectors)
    np.random.seed(42)
    n_points = 50

    # Create a smooth trajectory with some noise
    t = np.linspace(0, 2 * np.pi, n_points)
    trajectory = []
    for i in range(n_points):
        vec = (
            np.array(
                [
                    np.sin(t[i]),
                    np.cos(t[i]),
                    np.sin(2 * t[i]),
                    np.cos(2 * t[i]),
                    t[i] / (2 * np.pi),
                    np.sin(t[i]) * np.cos(t[i]),
                    0.5,
                    0.5,
                ]
            )
            + np.random.randn(8) * 0.05
        )  # Add noise
        trajectory.append(vec)

    compressor = TrajectoryCompressor(hyperbolic=True)

    print(f"Original trajectory: {n_points} points × 8 dimensions")
    print(f"Original size: {n_points * 8 * 8} bytes\n")

    for tolerance in [0.05, 0.1, 0.2, 0.5]:
        compressed = compressor.compress(trajectory, tolerance=tolerance)
        compressed_bytes = compressed.to_bytes()

        print(
            f"Tolerance {tolerance:.2f}: "
            f"{len(compressed.perturbations):3d} perturbations | "
            f"Ratio: {compressed.compression_ratio:5.2f}x | "
            f"Fidelity: {compressed.semantic_fidelity:.3f} | "
            f"Size: {len(compressed_bytes)} bytes"
        )


def demo_black_hole_compression():
    """Demonstrate black hole compression."""
    print_header("BLACK HOLE COMPRESSION")

    bh = BlackHoleCompressor(dimension=24)

    print(f"Black Hole Properties (Initial):")
    print(f"  - Dimension: {bh.dimension}")
    print(f"  - Mass: {bh.mass:.4f}")
    print(f"  - Radius: {bh.radius:.4f}")
    print(f"  - Entropy: {bh.entropy:.4f}")
    print(f"  - Temperature: {bh.temperature:.4f}")

    # Absorb some vectors
    print(f"\nAbsorbing 100 random vectors...")

    for i in range(100):
        vec = np.random.randn(24) * 0.5
        bh.absorb(vec, metadata={"index": i})

    print(f"\nBlack Hole Properties (After absorption):")
    print(f"  - Mass: {bh.mass:.4f}")
    print(f"  - Radius: {bh.radius:.4f}")
    print(f"  - Entropy: {bh.entropy:.4f}")
    print(f"  - Temperature: {bh.temperature:.4f}")
    print(f"  - Absorbed items: {len(bh.absorbed_items)}")

    # Get horizon state
    horizon = bh.get_horizon_state()
    horizon_bytes = horizon.to_bytes()

    original_size = 100 * 24 * 8  # 100 vectors × 24 dims × 8 bytes
    print(f"\n  Original size: {original_size} bytes")
    print(f"  Horizon size: {len(horizon_bytes)} bytes")
    print(f"  Compression ratio: {original_size / len(horizon_bytes):.2f}x")


def demo_file_compression():
    """Demonstrate file compression."""
    print_header("FILE COMPRESSION DEMO")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test file
        input_path = os.path.join(tmpdir, "sample.txt")
        output_path = os.path.join(tmpdir, "sample.e8z")

        # Generate sample content
        content = []
        for i in range(100):
            content.append(f"Line {i}: The E8 lattice has 240 root vectors. " * 3)

        sample_text = "\n".join(content)

        with open(input_path, "w") as f:
            f.write(sample_text)

        original_size = len(sample_text.encode())
        print(f"Created sample file: {original_size} bytes")

        # Compress with normal mode
        compressor = E8Compressor(mode=CompressionMode.NORMAL)

        start = time.time()
        orig, comp = compressor.compress_file(input_path, output_path)
        elapsed = time.time() - start

        ratio = orig / comp

        print(f"\nCompression Results:")
        print(f"  - Original size: {orig} bytes")
        print(f"  - Compressed size: {comp} bytes")
        print(f"  - Ratio: {ratio:.2f}x")
        print(f"  - Time: {elapsed * 1000:.2f}ms")
        print(f"  - Speed: {orig / elapsed / 1024:.2f} KB/s")


def main():
    """Run all demos."""
    print("\n" + "🌀" * 30)
    print("\n           E8ZIP - GEOMETRIC LATTICE COMPRESSION")
    print("        Based on E8 Kaleidoscope Mind's Algorithms")
    print("\n" + "🌀" * 30)

    demo_e8_lattice()
    demo_compression_modes()
    demo_trajectory_compression()
    demo_black_hole_compression()
    demo_file_compression()

    print_header("DEMO COMPLETE")
    print("E8ZIP is ready for use!")
    print("\nUsage:")
    print("  e8zip compress myfile.txt")
    print("  e8zip decompress myfile.e8z")
    print("  e8zip info myfile.e8z")
    print('\n"Compress paths through the lattice, not just snapshots."')
    print()


if __name__ == "__main__":
    main()

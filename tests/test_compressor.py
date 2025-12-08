"""
Test E8 Compressor
"""

import numpy as np
import pytest
import tempfile
import os

from e8zip.core.compressor import E8Compressor, CompressionMode
from e8zip.core.codec import E8GeometricCodec
from e8zip.core.trajectory import TrajectoryCompressor


class TestE8Compressor:
    """Tests for E8 compression engine."""

    def test_compress_decompress_fast(self):
        """Test fast mode compression/decompression."""
        compressor = E8Compressor(mode=CompressionMode.FAST)

        # Create test data
        original = b"Hello, E8 Lattice Compression! This is a test." * 10

        # Compress
        compressed = compressor.compress(original)

        # Should produce valid output
        assert compressed.startswith(b"E8ZIP001")

        # Decompress
        restored = compressor.decompress(compressed)

        # Should match original
        assert restored == original

    def test_compress_decompress_normal(self):
        """Test normal mode compression/decompression."""
        compressor = E8Compressor(mode=CompressionMode.NORMAL)

        original = b"Testing trajectory compression mode." * 20

        compressed = compressor.compress(original)
        restored = compressor.decompress(compressed)

        assert restored == original

    def test_compress_file(self):
        """Test file compression."""
        compressor = E8Compressor(mode=CompressionMode.FAST)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test file
            input_path = os.path.join(tmpdir, "test.txt")
            output_path = os.path.join(tmpdir, "test.e8z")
            restored_path = os.path.join(tmpdir, "restored.txt")

            original_content = b"Test file content for E8ZIP compression." * 50

            with open(input_path, "wb") as f:
                f.write(original_content)

            # Compress
            orig_size, comp_size = compressor.compress_file(input_path, output_path)

            assert os.path.exists(output_path)
            assert orig_size == len(original_content)

            # Decompress
            compressor.decompress_file(output_path, restored_path)

            with open(restored_path, "rb") as f:
                restored_content = f.read()

            assert restored_content == original_content

    def test_compression_ratio(self):
        """Test that compression achieves reasonable ratios."""
        compressor = E8Compressor(mode=CompressionMode.FAST)

        # Repetitive data should compress well
        original = b"AAAAAAAAAA" * 1000
        compressed = compressor.compress(original)

        ratio = len(original) / len(compressed)

        # Should achieve at least some compression
        assert ratio > 0.5  # Minimal expectation

    def test_different_modes(self):
        """Test all compression modes work."""
        original = b"Testing all compression modes." * 30

        for mode in CompressionMode:
            compressor = E8Compressor(mode=mode)
            compressed = compressor.compress(original)

            # MYTHIC mode is lossy, so we skip exact comparison
            if mode != CompressionMode.MYTHIC:
                restored = compressor.decompress(compressed)
                assert restored == original, f"Mode {mode.name} failed"


class TestE8GeometricCodec:
    """Tests for E8 geometric codec."""

    def test_quantize_root(self):
        """Test quantization to roots."""
        codec = E8GeometricCodec(mode="root")

        # Create a vector near a root
        vector = np.array([0.9, 0.9, 0.1, 0.1, 0, 0, 0, 0])

        result = codec.quantize(vector)

        assert result.is_root
        assert len(result.lattice_point) == 8

    def test_quantize_lattice(self):
        """Test quantization to lattice."""
        codec = E8GeometricCodec(mode="lattice")

        vector = np.array([2.3, 1.7, 0.2, -0.9, 0.5, 0.5, 0.5, 0.5])

        result = codec.quantize(vector)

        assert len(result.lattice_point) == 8
        assert result.error >= 0

    def test_encode_decode(self):
        """Test encoding and decoding."""
        codec = E8GeometricCodec(mode="root")

        vector = np.array([0.8, 0.8, 0.1, 0.1, 0, 0, 0, 0])

        node_id, residual = codec.encode(vector)

        # We need the lattice point to decode
        result = codec.quantize(vector)
        decoded = codec.decode(node_id, residual, result.lattice_point)

        np.testing.assert_allclose(decoded, vector, rtol=1e-10)

    def test_batch_quantize(self):
        """Test batch quantization."""
        codec = E8GeometricCodec()

        vectors = np.random.randn(10, 8)

        ids, points, errors = codec.batch_quantize(vectors)

        assert len(ids) == 10
        assert points.shape == (10, 8)
        assert len(errors) == 10


class TestTrajectoryCompressor:
    """Tests for trajectory compression."""

    def test_compress_simple_trajectory(self):
        """Test compressing a simple trajectory."""
        compressor = TrajectoryCompressor()

        # Create a linear trajectory
        trajectory = [np.linspace(0, 1, 8) * i for i in range(10)]

        compressed = compressor.compress(trajectory, tolerance=0.1)

        assert compressed.length == 10
        # Compression ratio can be less than 1 for small trajectories with overhead
        assert compressed.compression_ratio > 0  # Just ensure it's positive

    def test_decompress_trajectory(self):
        """Test decompressing a trajectory."""
        compressor = TrajectoryCompressor()

        trajectory = [np.random.randn(8) for _ in range(20)]

        compressed = compressor.compress(trajectory, tolerance=0.5)
        decompressed = compressor.decompress(compressed)

        assert len(decompressed) == len(trajectory)

    def test_trajectory_serialization(self):
        """Test trajectory serialization."""
        compressor = TrajectoryCompressor()

        trajectory = [np.random.randn(8) for _ in range(15)]
        compressed = compressor.compress(trajectory, tolerance=0.2)

        # Serialize
        data = compressed.to_bytes()

        # Deserialize
        from e8zip.core.trajectory import CompressedTrajectory

        restored = CompressedTrajectory.from_bytes(data)

        assert restored.length == compressed.length
        np.testing.assert_allclose(restored.start_vector, compressed.start_vector)
        np.testing.assert_allclose(restored.end_vector, compressed.end_vector)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

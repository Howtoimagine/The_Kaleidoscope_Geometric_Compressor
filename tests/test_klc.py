"""
Tests for core/transforms.py and core/klc.py (the gain-shape progressive
lattice codec borrowed and extended from the Glass Network's KLC2) plus
the KGC2 header's Golay self-healing.
"""

import numpy as np
import pytest

from e8zip.core import transforms
from e8zip.core import klc
from e8zip.core.kgc import KGCCompressor


class TestTransforms:
    @pytest.mark.parametrize("kind,dim", [("hadamard", 128), ("rotation", 96), ("none", 64)])
    def test_orthogonal_roundtrip(self, kind, dim):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, (5, dim))
        fwd = transforms.forward(a, kind, seed=7)
        back = transforms.inverse(fwd, kind, seed=7)
        assert np.allclose(a, back, atol=1e-9)

    @pytest.mark.parametrize("kind,dim", [("hadamard", 128), ("rotation", 96)])
    def test_norm_preserving(self, kind, dim):
        rng = np.random.default_rng(1)
        a = rng.normal(0, 1, (8, dim))
        fwd = transforms.forward(a, kind, seed=3)
        assert np.allclose(
            np.linalg.norm(a, axis=1), np.linalg.norm(fwd, axis=1), atol=1e-9
        )

    def test_hadamard_rejects_non_power_of_two(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, (2, 100))
        with pytest.raises(Exception):
            transforms.forward(a, "hadamard", seed=1)

    def test_orthogonal_matrix_is_orthogonal(self):
        m = transforms.orthogonal_matrix(32, seed=42)
        assert np.allclose(m @ m.T, np.eye(32), atol=1e-9)


class TestKLC:
    def test_roundtrip_shapes(self):
        rng = np.random.default_rng(2)
        for shape in [(7, 13), (100,), (5, 5, 5), (48, 24)]:
            w = rng.normal(0, 0.1, shape).astype(np.float32)
            blob = klc.encode(w, n_layers=1)
            recon = klc.decode(blob)
            assert recon.shape == w.shape

    def test_progressive_fidelity_is_monotonic(self):
        rng = np.random.default_rng(3)
        w = rng.normal(0, 0.02, (16, 96)).astype(np.float32)
        blob = klc.encode(w, n_layers=3, shape_scale=16.0)
        mses = []
        for layer in range(4):
            recon = klc.decode(blob, up_to_layer=layer)
            mses.append(float(np.mean((w.astype(np.float64) - recon.astype(np.float64)) ** 2)))
        assert all(mses[i + 1] <= mses[i] + 1e-12 for i in range(len(mses) - 1))

    def test_layer_sizes_monotonic_and_bounded(self):
        rng = np.random.default_rng(4)
        w = rng.normal(0, 0.05, (8, 48)).astype(np.float32)
        blob = klc.encode(w, n_layers=2)
        sizes = klc.layer_sizes(blob)
        assert sizes["base"] < sizes["layer_1"] < sizes["layer_2"] == sizes["full"]

    def test_zero_blocks_reconstruct_exactly(self):
        z = np.zeros((6, 24), dtype=np.float32)
        blob = klc.encode(z, n_layers=2)
        recon = klc.decode(blob)
        assert np.allclose(recon, 0.0)

    def test_gain_recovers_magnitude(self):
        # A block with a large, clean gain should recover its norm closely
        rng = np.random.default_rng(5)
        block = rng.normal(0, 1, 24)
        block = block / np.linalg.norm(block) * 7.3
        blob = klc.encode(block, n_layers=2)
        recon = klc.decode(blob)
        assert abs(np.linalg.norm(recon) - 7.3) / 7.3 < 0.05

    def test_transform_options_all_roundtrip(self):
        rng = np.random.default_rng(6)
        w = rng.normal(0, 0.03, (8, 96)).astype(np.float32)
        for kind, dim in (("hadamard", 128), ("rotation", 96), ("none", 24)):
            blob = klc.encode(w, n_layers=1, transform=kind, transform_dim=dim)
            recon = klc.decode(blob)
            assert recon.shape == w.shape

    def test_more_layers_never_worse_snr(self):
        rng = np.random.default_rng(7)
        w = rng.normal(0, 0.02, (16, 48)).astype(np.float32)
        blob = klc.encode(w, n_layers=3, shape_scale=16.0)
        snrs = [klc.reconstruction_snr_db(w, klc.decode(blob, up_to_layer=L)) for L in range(4)]
        assert all(snrs[i + 1] >= snrs[i] - 1e-6 for i in range(3))


class TestKGCProgressiveFacade:
    def test_facade_matches_module(self):
        rng = np.random.default_rng(8)
        w = rng.normal(0, 0.02, (8, 48)).astype(np.float32)
        kgc = KGCCompressor()
        blob = kgc.compress_tensor_progressive(w, n_layers=2)
        recon = kgc.decompress_tensor_progressive(blob)
        assert recon.shape == w.shape
        sizes = kgc.tensor_progressive_layer_sizes(blob)
        assert sizes["full"] == len(blob)

    def test_tensor_transform_option(self):
        rng = np.random.default_rng(9)
        w = rng.normal(0, 0.02, (8, 384)).astype(np.float32)
        kgc = KGCCompressor()
        blob = kgc.compress_tensor(w, scale=4.0, transform="rotation")
        recon, info = kgc.decompress_tensor(blob)
        assert recon.shape == w.shape


class TestHeaderHealing:
    def setup_method(self):
        self.kgc = KGCCompressor()
        self.data = b"Golay protects the control block now." * 20

    def test_heals_one_to_three_bit_flips(self):
        blob = bytearray(self.kgc.compress(self.data, mode="fast"))
        rng = np.random.default_rng(0)
        for _ in range(20):
            corrupted = bytearray(blob)
            nbits = int(rng.integers(1, 4))
            positions = rng.choice(24, size=nbits, replace=False)
            for bp in positions:
                byte_idx = 4 + int(bp) // 8
                bit_idx = int(bp) % 8
                corrupted[byte_idx] ^= 1 << bit_idx
            out, info = self.kgc.decompress(bytes(corrupted))
            assert out == self.data

    def test_uncorrupted_header_unaffected(self):
        blob = self.kgc.compress(self.data, mode="fast")
        out, info = self.kgc.decompress(blob)
        assert out == self.data
        assert info["verified"]

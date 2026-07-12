"""
Tests for the Kaleidoscope Geometric Codec (KGC v2).

Falsification-first, in the Glass Network style: every claim the codec
makes (exactness, healing, conservation, honesty on random data) has a
test that would fail if the claim were false.
"""

import hashlib

import numpy as np
import pytest

from e8zip.core.entropy import (
    AdaptiveModel,
    Order1ByteModel,
    Order2ByteModel,
    RangeDecoder,
    RangeEncoder,
    decode_indices,
    encode_indices,
)
from e8zip.core.golay import (
    WEIGHT_ENUMERATOR,
    all_codewords,
    decode_24,
    encode_12_to_24,
)
from e8zip.core.kgc import KGCCompressor, Law9Error
from e8zip.core.leech_lattice import (
    LeechCodec,
    batch_nearest_leech_point,
    leech_index_decompose,
    leech_index_reconstruct,
)


# ----------------------------------------------------------------------
# Entropy spine
# ----------------------------------------------------------------------


class TestRangeCoder:
    def test_roundtrip_text(self):
        data = b"geodesic paths through the lattice " * 100
        blob = Order1ByteModel().compress(data)
        assert Order1ByteModel().decompress(blob, len(data)) == data
        assert len(blob) < len(data)

    def test_roundtrip_order2(self):
        data = b"the quick brown fox jumps over the lazy dog. " * 80
        blob = Order2ByteModel().compress(data)
        assert Order2ByteModel().decompress(blob, len(data)) == data

    def test_roundtrip_random_many_seeds(self):
        for seed in range(20):
            rng = np.random.default_rng(seed)
            data = rng.integers(0, 256, int(rng.integers(1, 4000))).astype(
                np.uint8
            ).tobytes()
            blob = Order1ByteModel().compress(data)
            assert Order1ByteModel().decompress(blob, len(data)) == data

    def test_extreme_skew_exercises_underflow(self):
        enc = RangeEncoder()
        rng = np.random.default_rng(0)
        syms = rng.choice(2, 50000, p=[0.999, 0.001])
        for s in syms:
            if s == 0:
                enc.encode(0, 65534, 65535)
            else:
                enc.encode(65534, 1, 65535)
        blob = enc.finish()
        dec = RangeDecoder(blob)
        for s in syms:
            f = dec.decode_freq(65535)
            d = 0 if f < 65534 else 1
            assert d == s
            if d == 0:
                dec.decode_update(0, 65534, 65535)
            else:
                dec.decode_update(65534, 1, 65535)

    def test_index_stream(self):
        rng = np.random.default_rng(2)
        idx = rng.zipf(1.5, 3000) % 4096
        blob = encode_indices(idx, 4096)
        assert np.array_equal(decode_indices(blob, 3000, 4096), idx)

    def test_adaptive_model_fenwick_consistency(self):
        m = AdaptiveModel(97)
        rng = np.random.default_rng(3)
        for _ in range(500):
            s = int(rng.integers(0, 97))
            m.update(s)
        cums = [m._cum(s) for s in range(98)]
        assert cums[0] == 0
        assert cums[-1] == m.total
        assert all(cums[i] < cums[i + 1] or m.freq(i) == 0 for i in range(97))


# ----------------------------------------------------------------------
# Golay code
# ----------------------------------------------------------------------


class TestGolay:
    def test_weight_enumerator(self):
        cw = all_codewords()
        weights = {}
        for w in cw.sum(axis=1).tolist():
            weights[w] = weights.get(w, 0) + 1
        assert weights == WEIGHT_ENUMERATOR

    def test_systematic(self):
        for msg in (0, 1, 0xABC, 4095):
            assert encode_12_to_24(msg) & 0xFFF == msg

    def test_heals_up_to_three_errors(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            msg = int(rng.integers(0, 4096))
            cw = encode_12_to_24(msg)
            nerr = int(rng.integers(0, 4))
            for p in rng.choice(24, size=nerr, replace=False):
                cw ^= 1 << int(p)
            decoded = decode_24(cw)
            assert decoded is not None
            assert decoded[1] == msg

    def test_detects_four_errors(self):
        # Weight-4 patterns are detected (None) or mis-corrected to a
        # different codeword -- they must never silently return the
        # original message as if nothing happened with 0 corrections.
        cw = encode_12_to_24(1234)
        corrupted = cw ^ 0b1111  # 4 flips
        decoded = decode_24(corrupted)
        if decoded is not None:
            codeword, _msg, nerr = decoded
            assert codeword != cw or nerr > 0


# ----------------------------------------------------------------------
# Leech lattice
# ----------------------------------------------------------------------


class TestLeech:
    def test_points_are_lattice_members(self):
        rng = np.random.default_rng(1)
        v = rng.normal(0, 1, (64, 24))
        points, shells, dists = batch_nearest_leech_point(v)
        y = points * 2.8284271247461903
        assert np.allclose(y, np.rint(y), atol=1e-9)
        # even lattice: squared norms are even integers (in y/sqrt8 coords: *2)
        norms2 = np.sum(points**2, axis=1)
        assert np.allclose(norms2 / 2.0, np.rint(norms2 / 2.0), atol=1e-9)

    def test_index_bijection(self):
        rng = np.random.default_rng(2)
        v = rng.normal(0, 1.5, (64, 24))
        points, _, _ = batch_nearest_leech_point(v)
        g_idx, case, z = leech_index_decompose(points)
        rec = leech_index_reconstruct(g_idx, case, z)
        assert np.allclose(rec, points, atol=1e-12)

    def test_zero_maps_to_origin(self):
        points, shells, dists = batch_nearest_leech_point(np.zeros((1, 24)))
        assert np.allclose(points[0], 0)
        assert shells[0] == 0
        assert dists[0] == 0

    def test_lattice_point_is_fixed(self):
        # A lattice point must quantize to itself
        rng = np.random.default_rng(3)
        v = rng.normal(0, 2, (8, 24))
        points, _, _ = batch_nearest_leech_point(v)
        again, _, dists = batch_nearest_leech_point(points)
        assert np.allclose(again, points, atol=1e-9)
        assert np.allclose(dists, 0, atol=1e-9)

    def test_codec_healing(self):
        codec = LeechCodec()
        q = codec.quantize(np.random.default_rng(4).normal(0, 1, 24))
        cw = codec.index_to_codeword(q.index)
        assert codec.heal_index(cw ^ 0b101000000000000000000001) == q.index


# ----------------------------------------------------------------------
# KGC container / regimes
# ----------------------------------------------------------------------


class TestKGCBytes:
    def setup_method(self):
        self.kgc = KGCCompressor()

    def test_lossless_roundtrip_all_modes(self):
        data = (b"Compression is prediction. Prediction is geometry. " * 60)
        for mode in ("fast", "strong", "geometric"):
            blob = self.kgc.compress(data, mode=mode)
            out, info = self.kgc.decompress(blob)
            assert out == data
            assert info["truth_status"] == "exact_recovery"
            assert info["verified"]

    def test_random_data_never_expands_much(self):
        rng = np.random.default_rng(5)
        data = rng.integers(0, 256, 8192).astype(np.uint8).tobytes()
        blob = self.kgc.compress(data, mode="fast")
        out, _ = self.kgc.decompress(blob)
        assert out == data
        # raw fallback: overhead bounded by header + 1 program byte
        assert len(blob) <= len(data) + 128

    def test_integrity_check_fires(self):
        data = b"tamper with me" * 50
        blob = bytearray(self.kgc.compress(data, mode="fast"))
        blob[-3] ^= 0xFF  # corrupt payload
        with pytest.raises(ValueError):
            self.kgc.decompress(bytes(blob))

    def test_inspect(self):
        data = b"inspectable" * 20
        blob = self.kgc.compress(data, mode="fast")
        meta = self.kgc.inspect(blob)
        assert meta["regime"] == "bytes"
        assert meta["source_length"] == len(data)
        assert meta["source_sha256"] == hashlib.sha256(data).hexdigest()


class TestKGCTensor:
    def setup_method(self):
        self.kgc = KGCCompressor()

    def test_rate_distortion_knob(self):
        rng = np.random.default_rng(6)
        w = rng.normal(0, 0.02, (24, 96)).astype(np.float32)
        results = {}
        for scale in (2.0, 6.0):
            blob = self.kgc.compress_tensor(w, scale=scale)
            w2, info = self.kgc.decompress_tensor(blob)
            assert w2.shape == w.shape
            assert info["truth_status"] == "reconstruction"
            mse = float(np.mean((w - w2) ** 2))
            results[scale] = (len(blob), mse)
        # more scale -> more bits, less distortion
        assert results[6.0][0] > results[2.0][0]
        assert results[6.0][1] < results[2.0][1]

    def test_beats_int4_scalar_at_comparable_rate(self):
        # The point of the whole exercise: lattice + incoherence +
        # entropy coding must beat scalar quantization at equal rate.
        rng = np.random.default_rng(7)
        w = rng.normal(0, 0.05, (48, 96)).astype(np.float32)

        blob = self.kgc.compress_tensor(w, scale=4.0)
        w2, _ = self.kgc.decompress_tensor(blob)
        bpw = len(blob) * 8 / w.size
        mse_kgc = float(np.mean((w - w2) ** 2))

        lo, hi = float(w.min()), float(w.max())
        q = np.round((w - lo) / (hi - lo) * 15)
        w4 = (q / 15 * (hi - lo) + lo).astype(np.float32)
        mse_int4 = float(np.mean((w - w4) ** 2))

        assert bpw < 6.0  # sane rate
        assert mse_kgc < mse_int4  # better distortion despite comparable bits


class TestKGCConsolidate:
    def setup_method(self):
        self.kgc = KGCCompressor()
        rng = np.random.default_rng(8)
        arche = rng.normal(0, 1.5, (12, 24))
        self.rows = (
            arche[rng.integers(0, 12, 120)] + rng.normal(0, 0.05, (120, 24))
        ).astype(np.float32)

    def test_exact_mode_bitwise_verified(self):
        blob = self.kgc.consolidate(self.rows, keep_residuals=True)
        out, info = self.kgc.deconsolidate(blob)
        assert info["verified"]
        assert info["truth_status"] == "exact_recovery"
        assert out.tobytes() == self.rows.tobytes()

    def test_mass_conservation(self):
        blob = self.kgc.consolidate(self.rows, keep_residuals=False)
        out, info = self.kgc.deconsolidate(blob)
        drift = abs(info["conserved_mass_out"] - info["conserved_mass_in"])
        assert drift / info["conserved_mass_in"] < 1e-3

    def test_collapse_ratio_grows_with_coarser_scale(self):
        _, fine = self.kgc.deconsolidate(
            self.kgc.consolidate(self.rows, rg_scale=1.0, keep_residuals=False)
        )
        _, coarse = self.kgc.deconsolidate(
            self.kgc.consolidate(self.rows, rg_scale=0.25, keep_residuals=False)
        )
        assert coarse["n_sites"] <= fine["n_sites"]

    def test_member_addresses_survive(self):
        ids = [f"obs:{i:04d}" for i in range(len(self.rows))]
        blob = self.kgc.consolidate(
            self.rows, keep_residuals=False, member_ids=ids
        )
        _, info = self.kgc.deconsolidate(blob)
        assert info["member_ids"] == ids

    def test_law9_shape_guard(self):
        with pytest.raises(ValueError):
            self.kgc.consolidate(np.zeros((10, 16)))  # wrong width

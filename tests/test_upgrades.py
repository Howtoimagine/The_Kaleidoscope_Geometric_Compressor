"""
Tests for the three v2.1 upgrades: theta-series shell prior, the
predictor (LLM socket) front-end, and the recall bake-off prototypes.
Falsification-first, as ever.
"""

import numpy as np
import pytest

from e8zip.core.entropy import AdaptiveModel, RangeDecoder, RangeEncoder
from e8zip.core.kgc import FLAG_SHELL_PRIOR, FLAG_Z_PRIOR, KGCCompressor, _parse_header
from e8zip.core.leech_lattice import THETA_SERIES, shell_prior, theta_series
from e8zip.core.predictor import (
    CallablePredictor,
    NGramMixPredictor,
    PredictorByteModel,
    get_predictor,
    register_predictor,
)
from e8zip.core.recall import (
    GemmRecall,
    LatticeCellRecall,
    ProjectionFilterRecall,
    measure_recall,
)


# ----------------------------------------------------------------------
# Theta series / shell prior
# ----------------------------------------------------------------------


class TestThetaSeries:
    def test_matches_tabulated_coefficients(self):
        assert theta_series(5) == THETA_SERIES

    def test_known_deep_coefficient(self):
        # N(2n) = (65520/691)(sigma_11(n) - tau(n)) must stay integral
        # (equivalent to Ramanujan's congruence); check a deep slice.
        coeffs = theta_series(50)
        assert all(isinstance(c, int) and c >= 0 for c in coeffs)
        assert coeffs[2] == 196560  # kissing number

    def test_shell_prior_is_distribution(self):
        p = shell_prior(300, 4.0)
        assert p.shape == (300,)
        assert abs(p.sum() - 1.0) < 1e-9
        assert np.all(p >= 0)
        # peak near 12*sigma^2 - 1 shells... theory: argmax ~ 11*sigma^2
        assert 120 <= int(np.argmax(p)) <= 200

    def test_shell_prior_empty_shell_gets_zero(self):
        p = shell_prior(4, 2.0)
        assert p[1] == 0.0  # no Leech points of norm 2


class TestAdaptiveModelPrior:
    def test_prior_roundtrip(self):
        prior = np.exp(-np.arange(64) / 3.0)
        rng = np.random.default_rng(0)
        syms = rng.integers(0, 8, 500)
        enc = RangeEncoder()
        m = AdaptiveModel(64, prior=prior)
        for s in syms:
            m.encode(enc, int(s))
        blob = enc.finish()
        dec = RangeDecoder(blob)
        m2 = AdaptiveModel(64, prior=prior)
        assert [m2.decode(dec) for _ in syms] == syms.tolist()

    def test_good_prior_beats_uniform(self):
        rng = np.random.default_rng(1)
        syms = np.minimum(rng.geometric(0.4, 2000) - 1, 63)

        def coded_size(prior):
            enc = RangeEncoder()
            m = AdaptiveModel(64, prior=prior)
            for s in syms:
                m.encode(enc, int(s))
            return len(enc.finish())

        good = 0.6 ** np.arange(64)
        assert coded_size(good) < coded_size(None)

    def test_bad_prior_rejected(self):
        with pytest.raises(ValueError):
            AdaptiveModel(4, prior=np.array([1.0, -1.0, 0.0, 0.0]))


class TestTensorShellPrior:
    def setup_method(self):
        self.kgc = KGCCompressor()
        rng = np.random.default_rng(6)
        self.w = rng.normal(0, 0.02, (24, 96)).astype(np.float32)

    def _roundtrip(self, **kw):
        blob = self.kgc.compress_tensor(self.w, scale=4.0, **kw)
        w2, info = self.kgc.decompress_tensor(blob)
        assert w2.shape == self.w.shape
        return blob, w2

    def test_all_flag_paths_roundtrip_identically(self):
        # same lattice points -> identical reconstruction whatever the
        # entropy-coding path; only the archive size may differ
        _, w_legacy = self._roundtrip(use_z_prior=False)
        _, w_zprior = self._roundtrip()
        _, w_shell = self._roundtrip(use_shell_prior=True)
        assert np.array_equal(w_legacy, w_zprior)
        assert np.array_equal(w_legacy, w_shell)

    def test_flags_recorded(self):
        blob, _ = self._roundtrip(use_shell_prior=True)
        _, _, flags, _, _ = _parse_header(blob)
        assert flags & FLAG_SHELL_PRIOR and flags & FLAG_Z_PRIOR
        blob, _ = self._roundtrip()
        _, _, flags, _, _ = _parse_header(blob)
        assert flags == FLAG_Z_PRIOR
        blob, _ = self._roundtrip(use_z_prior=False)
        _, _, flags, _, _ = _parse_header(blob)
        assert flags == 0  # legacy stream layout: old archives decode

    def test_z_prior_does_not_cost_rate(self):
        blob_legacy, _ = self._roundtrip(use_z_prior=False)
        blob_zprior, _ = self._roundtrip()
        assert len(blob_zprior) <= len(blob_legacy)


# ----------------------------------------------------------------------
# Predictor front-end (LLM socket)
# ----------------------------------------------------------------------


class TestPredictorFrontEnd:
    def test_ngram_mix_roundtrip_and_shrinks_text(self):
        data = b"the kaleidoscope turns and the lattice sings " * 200
        model = PredictorByteModel(NGramMixPredictor())
        blob = model.compress(data)
        assert PredictorByteModel(NGramMixPredictor()).decompress(
            blob, len(data)
        ) == data
        assert len(blob) < len(data) // 3

    def test_kgc_predictor_mode(self):
        kgc = KGCCompressor()
        data = b"compression is prediction; prediction is geometry. " * 100
        blob = kgc.compress(data, mode="predictor:ngram-mix")
        out, info = kgc.decompress(blob)
        assert out == data and info["verified"]

    def test_unknown_predictor_raises(self):
        with pytest.raises(KeyError):
            get_predictor("no-such-model")

    def test_callable_predictor_llm_socket(self):
        # deterministic stand-in for an LLM logits function
        def logits_fn(context: bytes) -> np.ndarray:
            logits = np.zeros(256)
            if context:
                logits[context[-1]] = 2.0  # "repeat last byte" model
            return logits

        register_predictor("toy-llm", lambda: CallablePredictor(logits_fn, 16))
        kgc = KGCCompressor()
        data = bytes([65] * 500 + [66] * 500)
        blob = kgc.compress(data, mode="predictor:toy-llm")
        out, _ = kgc.decompress(blob)
        assert out == data

    def test_random_data_never_expands(self):
        rng = np.random.default_rng(9)
        data = rng.integers(0, 256, 4096).astype(np.uint8).tobytes()
        kgc = KGCCompressor()
        blob = kgc.compress(data, mode="predictor")
        out, _ = kgc.decompress(blob)
        assert out == data
        assert len(blob) <= len(data) + 128  # raw fallback engaged


# ----------------------------------------------------------------------
# Recall prototypes
# ----------------------------------------------------------------------


class TestRecall:
    def setup_method(self):
        rng = np.random.default_rng(3)
        arche = rng.normal(0, 1.5, (16, 24))
        self.rows = arche[rng.integers(0, 16, 2000)] + rng.normal(
            0, 0.15, (2000, 24)
        )
        self.queries = self.rows[rng.integers(0, 2000, 16)] + rng.normal(
            0, 0.1, (16, 24)
        )

    def test_gemm_exact_self_query(self):
        g = GemmRecall(self.rows)
        idx, dist = g.query(self.rows[:5], k=1)
        assert np.array_equal(idx[:, 0], np.arange(5))
        assert np.allclose(dist[:, 0], 0.0, atol=1e-9)

    def test_lattice_cell_recall_quality(self):
        exact_idx, _ = GemmRecall(self.rows).query(self.queries, k=10)
        lat_idx, _ = LatticeCellRecall(self.rows).query(self.queries, k=10)
        assert measure_recall(lat_idx, exact_idx) > 0.5

    def test_projection_filter_recall_quality(self):
        exact_idx, _ = GemmRecall(self.rows).query(self.queries, k=10)
        proj = ProjectionFilterRecall(self.rows, radius=1.5, votes=2)
        p_idx, _ = proj.query(self.queries, k=10)
        assert measure_recall(p_idx, exact_idx) > 0.5

    def test_candidate_distances_are_exact(self):
        # whatever the filter surfaces, distances must be true distances
        proj = ProjectionFilterRecall(self.rows, radius=1.5, votes=2)
        idx, dist = proj.query(self.queries[:2], k=5)
        for j in range(2):
            for i, d in zip(idx[j], dist[j]):
                if i >= 0:
                    true = np.sum((self.rows[i] - self.queries[j]) ** 2)
                    assert abs(true - d) < 1e-6


# ----------------------------------------------------------------------
# GPU backends + neural predictor (skipped when torch is absent)
# ----------------------------------------------------------------------

torch = pytest.importorskip("torch")


class TestTorchRecall:
    def setup_method(self):
        rng = np.random.default_rng(4)
        arche = rng.normal(0, 1.5, (16, 24))
        self.rows = arche[rng.integers(0, 16, 3000)] + rng.normal(
            0, 0.15, (3000, 24)
        )
        self.queries = self.rows[rng.integers(0, 3000, 16)] + rng.normal(
            0, 0.1, (16, 24)
        )

    def test_torch_gemm_matches_cpu_exact(self):
        from e8zip.core.recall_gpu import TorchGemmRecall

        cpu_idx, cpu_d = GemmRecall(self.rows).query(self.queries, k=5)
        gpu_idx, gpu_d = TorchGemmRecall(self.rows, device="cpu").query(
            self.queries, k=5
        )
        assert np.array_equal(cpu_idx, gpu_idx)
        assert np.allclose(cpu_d, gpu_d, atol=1e-2)  # float32 device path

    def test_torch_projection_filter_recall_quality(self):
        from e8zip.core.recall_gpu import TorchProjectionFilterRecall

        exact_idx, _ = GemmRecall(self.rows).query(self.queries, k=10)
        proj = TorchProjectionFilterRecall(
            self.rows, radius=1.5, votes=2, device="cpu"
        )
        idx, dist = proj.query(self.queries, k=10)
        assert measure_recall(idx, exact_idx) > 0.9
        # surfaced distances are true 24-D distances
        for i, d in zip(idx[0], dist[0]):
            if i >= 0:
                true = np.sum((self.rows[i] - self.queries[0]) ** 2)
                assert abs(true - d) < 1e-2

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
    def test_cuda_paths_run(self):
        from e8zip.core.recall_gpu import TorchGemmRecall, TorchProjectionFilterRecall

        exact_idx, _ = TorchGemmRecall(self.rows).query(self.queries, k=10)
        proj = TorchProjectionFilterRecall(self.rows, radius=1.5, votes=2)
        idx, _ = proj.query(self.queries, k=10)
        assert measure_recall(idx, exact_idx) > 0.9

    def test_query_chunking_matches_unchunked(self):
        # bounded-memory chunking must not change results
        from e8zip.core.recall_gpu import TorchProjectionFilterRecall

        big = TorchProjectionFilterRecall(
            self.rows, radius=1.5, votes=2, device="cpu", query_chunk=4
        )
        small = TorchProjectionFilterRecall(
            self.rows, radius=1.5, votes=2, device="cpu", query_chunk=10_000
        )
        a_idx, a_d = big.query(self.queries, k=10)
        b_idx, b_d = small.query(self.queries, k=10)
        assert np.array_equal(a_idx, b_idx)
        assert np.allclose(a_d, b_d, atol=1e-3)


def _optix_available() -> bool:
    try:
        import cupy  # noqa: F401
        import optix  # noqa: F401
        from cuda.bindings import nvrtc  # noqa: F401

        return optix.deviceContextCreate is not None
    except Exception:
        return False


@pytest.mark.skipif(not _optix_available(), reason="OptiX/CuPy stack not installed")
class TestOptixRecall:
    def setup_method(self):
        rng = np.random.default_rng(5)
        arche = rng.normal(0, 1.5, (16, 24))
        self.rows = arche[rng.integers(0, 16, 3000)] + rng.normal(
            0, 0.15, (3000, 24)
        )
        self.queries = self.rows[rng.integers(0, 3000, 32)] + rng.normal(
            0, 0.1, (32, 24)
        )

    def test_rt_core_recall_and_exact_distances(self):
        from e8zip.core.recall import GemmRecall
        from e8zip.core.rt_optix import OptixProjectionFilterRecall

        exact_idx, _ = GemmRecall(self.rows).query(self.queries, k=10)
        rt = OptixProjectionFilterRecall(self.rows, radius=1.5, votes=2)
        idx, dist = rt.query(self.queries, k=10)
        # BVH containment is exact, so recall should be very high
        assert measure_recall(idx, exact_idx) > 0.95
        for i, d in zip(idx[0], dist[0]):
            if i >= 0:
                true = np.sum((self.rows[i] - self.queries[0]) ** 2)
                assert abs(true - d) < 1e-2

    def test_optix_query_chunking_matches(self):
        from e8zip.core.rt_optix import OptixProjectionFilterRecall

        rt = OptixProjectionFilterRecall(
            self.rows, radius=1.5, votes=2, query_chunk=8
        )
        chunked_idx, _ = rt.query(self.queries, k=10)
        whole_idx, _ = rt._query_chunk(
            np.asarray(self.queries, dtype=np.float32), 10
        )
        assert np.array_equal(chunked_idx, whole_idx)


class TestOnlineGRUPredictor:
    def test_deterministic_roundtrip_through_kgc(self):
        kgc = KGCCompressor()
        data = b"the model that predicts the stream becomes the stream. " * 40
        blob = kgc.compress(data, mode="predictor:gru-online")
        out, info = kgc.decompress(blob)
        assert out == data and info["verified"]

    def test_learns_within_a_stream(self):
        # coded size of the second half must beat the first half on
        # repetitive data: falsifies "the GRU never learns"
        from e8zip.core.predictor import OnlineGRUPredictor, PredictorByteModel

        data = b"abcdefgh" * 128
        half = len(data) // 2
        m = PredictorByteModel(OnlineGRUPredictor())
        full = len(m.compress(data))
        m2 = PredictorByteModel(OnlineGRUPredictor())
        first = len(m2.compress(data[:half]))
        assert full - first < first  # second half cheaper than first

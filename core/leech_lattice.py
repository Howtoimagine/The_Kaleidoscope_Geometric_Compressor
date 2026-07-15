"""
Leech Lattice (Λ24) - Exact Nearest-Point Quantizer via Golay Construction A

This replaces the old approximate/slow implementation with the exact
Conway-Sloane soft-decision decoder ported from the Glass Network KMind
(packages/kmind/leech.py, glass_windows branch), extended with the index
decomposition that makes the quantizer *codeable*:

    Every Leech point (scaled by sqrt(8)) is an integer vector
        y = 2*g + i*1 + 4*z
    with g one of the 4096 Golay codewords, i in {0,1} the coset case,
    and z in Z^24 the translation. (g_idx, i, z) is a bijective index
    of the lattice point -- 12 bits + 1 bit + small integers -- which is
    exactly what the range coder consumes. This also repairs the arity
    mismatch in the KMind's turbo_leech_packer (which expected this
    decomposition but the decoder never returned it).

Why Leech: densest lattice packing in 24 dimensions, kissing number
196560, coding gain 1.04 dB over scalar quantization (~0.173 bits/dim
saved at fixed distortion). Theta series Theta = E_12 - (65520/691)*Delta.

Based on:
- Conway & Sloane: "Sphere Packings, Lattices and Groups"
- Glass Network KMind leech.py / golay.py (glass_windows)
"""

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

import numpy as np

from e8zip.core.golay import all_codewords, decode_24, encode_12_to_24

SQRT8 = 2.8284271247461903

# Leech theta-series coefficients: number of lattice points of norm 2n.
# Theta_Lambda = E_12 - (65520/691) * Delta  (integrality <=> Ramanujan's
# tau(n) == sigma_11(n) mod 691). The shell populations are the natural
# maximum-entropy prior for shell-indexed entropy coding.
THETA_SERIES = (1, 0, 196560, 16773120, 398034000, 4629381120)


@lru_cache(maxsize=8)
def theta_series(n_max: int) -> Tuple[int, ...]:
    """
    Exact Leech theta coefficients N(2n) for n = 0..n_max (arbitrary n).

    Theta_Lambda = E_12 - (65520/691) * Delta, i.e. for n >= 1
        N(2n) = (65520/691) * (sigma_11(n) - tau(n)),
    with tau from Delta = q * prod_{m>=1} (1 - q^m)^24. The Euler product
    is expanded by the pentagonal number theorem, then raised to the 24th
    power by squaring; everything stays in exact Python integers.
    """
    n_terms = n_max + 1

    # Euler function prod (1 - q^m) mod q^n_terms (pentagonal number theorem)
    euler = [0] * n_terms
    euler[0] = 1
    k = 1
    while k * (3 * k - 1) // 2 < n_terms:
        sign = -1 if k % 2 else 1
        for g in (k * (3 * k - 1) // 2, k * (3 * k + 1) // 2):
            if g < n_terms:
                euler[g] = sign
        k += 1

    def _mul(a: List[int], b: List[int]) -> List[int]:
        out = [0] * n_terms
        for i, ai in enumerate(a):
            if ai:
                for j in range(min(len(b), n_terms - i)):
                    if b[j]:
                        out[i + j] += ai * b[j]
        return out

    # euler^24 = ((euler^2)^2 * euler^2)^... : 24 = 16 + 8
    e2 = _mul(euler, euler)
    e4 = _mul(e2, e2)
    e8 = _mul(e4, e4)
    e16 = _mul(e8, e8)
    eta24 = _mul(e16, e8)  # tau(n) = eta24[n-1]

    # sigma_11 by divisor sieve
    sigma11 = [0] * n_terms
    for d in range(1, n_terms):
        p = d**11
        for m in range(d, n_terms, d):
            sigma11[m] += p

    coeffs = [1]
    for n in range(1, n_terms):
        num = 65520 * (sigma11[n] - eta24[n - 1])
        q, r = divmod(num, 691)
        if r:
            raise ArithmeticError(f"theta coefficient {n} not integral")
        coeffs.append(q)
    return tuple(coeffs)


def shell_prior(n_shells: int, sigma: float) -> np.ndarray:
    """
    Maximum-entropy shell prior for a Gaussian source of per-coordinate
    std `sigma` quantized to the Leech lattice.

    P(shell n) ~ N(2n) * exp(-n / sigma^2): shell population (theta
    series) times the Gaussian radial density at squared norm 2n.
    Returns a normalized probability vector of length n_shells.
    """
    if n_shells < 1:
        raise ValueError("n_shells must be >= 1")
    coeffs = theta_series(n_shells - 1)
    s2 = max(float(sigma) ** 2, 1e-12)
    logs = np.array(
        [math.log(c) if c > 0 else -math.inf for c in coeffs], dtype=np.float64
    )
    logs -= np.arange(n_shells, dtype=np.float64) / s2
    logs -= logs[np.isfinite(logs)].max()
    w = np.exp(logs)
    total = w.sum()
    return w / total if total > 0 else np.full(n_shells, 1.0 / n_shells)


@dataclass
class LeechQuantization:
    """Result of Leech quantization with full codeable index."""

    lattice_point: np.ndarray
    index: int  # 12-bit Golay message index
    case: int  # coset case i in {0, 1}
    z: np.ndarray  # integer translation vector (24,)
    shell: int
    error: float


@lru_cache(maxsize=1)
def _codewords_np() -> np.ndarray:
    return all_codewords()  # (4096, 24) int32; row index == message


def batch_nearest_leech_point(
    v: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized exact nearest-Leech-point projection.

    Args:
        v: (N, 24) array in standard Leech coordinates.

    Returns:
        points (N, 24), shells (N,), distances (N,)
    """
    N = v.shape[0]
    if v.shape[1] != 24:
        raise ValueError("Leech vectors must have 24 dimensions")

    y = v * SQRT8
    golay = _codewords_np()  # (4096, 24)

    best_dist = np.full(N, np.inf, dtype=np.float64)
    best_c = np.zeros((N, 24), dtype=np.float64)

    for case in (0, 1):
        m0 = case
        m1 = (2 + case) % 4

        # Nearest integer congruent to m mod 4, per coordinate
        c0 = 4 * np.round((y - m0) / 4) + m0
        c1 = 4 * np.round((y - m1) / 4) + m1

        d0 = (y - c0) ** 2
        d1 = (y - c1) ** 2

        # Next-nearest candidates (one step of 4 toward the value)
        sign0 = np.where(y >= c0, 1.0, -1.0)
        c0_next = c0 + 4 * sign0
        p0 = (y - c0_next) ** 2 - d0

        sign1 = np.where(y >= c1, 1.0, -1.0)
        c1_next = c1 + 4 * sign1
        p1 = (y - c1_next) ** 2 - d1

        expected_mod = 0 if case == 0 else 4

        dist_base = d0.sum(axis=1)  # (N,)
        dist_diff = d1 - d0  # (N, 24)

        c_base = c0.sum(axis=1)  # (N,)
        c_diff = c1 - c0  # (N, 24)

        # Every array below this point has an (N, 4096) or (N, 4096, 24)
        # dimension - dist_k/c_sum_k/parity_match alone are ~5.7 GB EACH
        # at N=175k (a real 2048x2048 LLM layer), computed simultaneously
        # if done for the full N up front. This OOM-killed exactly that
        # layer on a 15 GB box. Chunking from here (not just the old
        # refinement-only chunk) bounds every (chunk, 4096, ...) array
        # regardless of chunk_size, so this can go back up from the
        # OOM-avoidance-era 64 - 64 added ~8x Python/numpy loop overhead
        # for no remaining memory benefit once dist_k/c_sum_k are chunked.
        chunk_size = 512
        for i_start in range(0, N, chunk_size):
            i_end = min(N, i_start + chunk_size)

            dist_k_chunk = (
                dist_base[i_start:i_end, None] + dist_diff[i_start:i_end] @ golay.T
            )  # (C, 4096)
            c_sum_k_chunk = (
                c_base[i_start:i_end, None] + c_diff[i_start:i_end] @ golay.T
            )  # (C, 4096)
            parity_chunk = (c_sum_k_chunk % 8) == expected_mod

            p0_chunk = p0[i_start:i_end, None, :]  # (C, 1, 24)
            p1_chunk = p1[i_start:i_end, None, :]
            g_exp = golay[None, :, :]  # (1, 4096, 24)

            p_chunk = np.where(g_exp == 0, p0_chunk, p1_chunk)
            min_p_k = p_chunk.min(axis=2)  # (C, 4096)
            best_i_k = p_chunk.argmin(axis=2)  # (C, 4096)

            dist_final = np.where(parity_chunk, dist_k_chunk, dist_k_chunk + min_p_k)

            best_k = dist_final.argmin(axis=1)  # (C,)
            rows = np.arange(i_end - i_start)
            best_dist_case = dist_final[rows, best_k]

            improved = best_dist_case < best_dist[i_start:i_end]
            if not np.any(improved):
                continue

            best_dist[i_start:i_end][improved] = best_dist_case[improved]

            k_improved = best_k[improved]
            g_best = golay[k_improved]  # (M, 24)

            c0_imp = c0[i_start:i_end][improved]
            c1_imp = c1[i_start:i_end][improved]
            c_k = np.where(g_best == 0, c0_imp, c1_imp)

            parity_best = parity_chunk[rows, best_k][improved]
            needs_fix = ~parity_best
            if np.any(needs_fix):
                fix_i = best_i_k[rows, best_k][improved][needs_fix]
                c0n = c0_next[i_start:i_end][improved][needs_fix]
                c1n = c1_next[i_start:i_end][improved][needs_fix]
                g_fix = g_best[needs_fix]
                c_next_k = np.where(g_fix == 0, c0n, c1n)
                fi = np.arange(len(fix_i))
                c_k[np.where(needs_fix)[0], fix_i] = c_next_k[fi, fix_i]

            best_c[i_start:i_end][improved] = c_k

    points = best_c / SQRT8
    squared_norms = np.sum(points**2, axis=1)
    # int64: shell = squared_norm/2 can legitimately exceed int32 range
    # for large-magnitude input (e.g. the legacy v1 LEECH-mode pathway,
    # which does not pre-scale its vectors the way KGC/KLC do).
    shells = np.nan_to_num(np.round(squared_norms / 2.0), nan=0.0, posinf=0.0).astype(
        np.int64
    )
    distances = np.sqrt(np.sum((v - points) ** 2, axis=1))

    return points, shells, distances


def nearest_leech_point(
    v: Sequence[float],
) -> Tuple[Tuple[float, ...], int, float]:
    """Single-vector convenience wrapper. Returns (point, shell, distance)."""
    arr = np.asarray(v, dtype=np.float64).reshape(1, 24)
    points, shells, distances = batch_nearest_leech_point(arr)
    return tuple(points[0].tolist()), int(shells[0]), float(distances[0])


def leech_index_decompose(
    points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompose Leech points into codeable indices: y = 2g + i*1 + 4z.

    Args:
        points: (N, 24) Leech points in standard coordinates.

    Returns:
        g_idx (N,) int64  - 12-bit Golay message indices
        case  (N,) int64  - coset case i in {0, 1}
        z     (N, 24) int64 - integer translations
    """
    y = np.rint(points * SQRT8).astype(np.int64)  # integral by construction
    # sum(y) = 4i (mod 8) determines the case
    case = ((y.sum(axis=1) % 8) // 4).astype(np.int64)
    # g = ((y - i) / 2) mod 2 must be a Golay codeword
    g_bits = (((y - case[:, None]) // 2) % 2).astype(np.int64)
    # systematic code: message = low 12 bits of the codeword
    powers = 1 << np.arange(12, dtype=np.int64)
    g_idx = (g_bits[:, :12] * powers).sum(axis=1)
    # z = (y - 2g - i) / 4
    z4 = y - 2 * g_bits - case[:, None]
    if not np.all(z4 % 4 == 0):
        raise ValueError("invalid Leech point: index decomposition failed")
    z = z4 // 4
    return g_idx, case, z


def leech_index_reconstruct(
    g_idx: np.ndarray, case: np.ndarray, z: np.ndarray
) -> np.ndarray:
    """Inverse of leech_index_decompose: (g_idx, i, z) -> points (N, 24)."""
    golay = _codewords_np()  # row index == message
    g_bits = golay[np.asarray(g_idx, dtype=np.int64)]  # (N, 24)
    y = (
        2 * g_bits
        + np.asarray(case, dtype=np.int64)[:, None]
        + 4 * np.asarray(z, dtype=np.int64)
    )
    return y.astype(np.float64) / SQRT8


class LeechLattice:
    """
    The Leech lattice for geometric compression (exact decoder).

    Backward-compatible surface for E8Compressor plus the new exact
    quantization and index-codec paths.
    """

    def __init__(self, cache_minimal: int = 0):
        # cache_minimal kept for API compatibility; the exact decoder
        # needs no sampled minimal-vector cache.
        pass

    def nearest_lattice_point(self, vector: np.ndarray) -> LeechQuantization:
        """Exact nearest Leech point with full index decomposition."""
        vector = self._ensure_24d(vector)
        points, shells, distances = batch_nearest_leech_point(vector.reshape(1, 24))
        g_idx, case, z = leech_index_decompose(points)
        return LeechQuantization(
            lattice_point=points[0],
            index=int(g_idx[0]),
            case=int(case[0]),
            z=z[0],
            shell=int(shells[0]),
            error=float(distances[0]),
        )

    def batch_quantize(
        self, vectors: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Exact batch quantization with index decomposition.

        Returns:
            points (N,24), g_idx (N,), case (N,), z (N,24), distances (N,)
        """
        vectors = np.asarray(vectors, dtype=np.float64)
        points, _, distances = batch_nearest_leech_point(vectors)
        g_idx, case, z = leech_index_decompose(points)
        return points, g_idx, case, z, distances

    def _ensure_24d(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float64).ravel()
        # Sanitize before the exact decoder: batch_nearest_leech_point has
        # no defensive clipping (it assumes well-scaled input, which every
        # KGC/KLC caller provides), but this OO wrapper is also the v1
        # LEECH-mode entry point, which can hand it NaN-sanitized-but-huge
        # values (core/compressor.py's byte pipeline clips only NaN/Inf,
        # not magnitude) - unclipped, squared_norms can overflow float64
        # and produce a NaN->int cast warning downstream. Matches the
        # nan_to_num/clip(-1e6, 1e6) convention used throughout e8_lattice.py.
        vector = np.nan_to_num(vector, nan=0.0, posinf=1e6, neginf=-1e6)
        vector = np.clip(vector, -1e6, 1e6)
        if len(vector) == 24:
            return vector
        result = np.zeros(24)
        n = min(len(vector), 24)
        result[:n] = vector[:n]
        return result

    def project_to_e8(self, vector: np.ndarray, layer: str) -> np.ndarray:
        """Extract one of three 8D layers from a 24D vector."""
        vector = self._ensure_24d(vector)
        idx = {"first": 0, "second": 1, "third": 2}.get(layer, 0)
        return vector[idx * 8 : (idx + 1) * 8]

    def embed_from_e8(self, e8_vector: np.ndarray, layer: str) -> np.ndarray:
        """Embed an 8D vector into one layer of a 24D vector."""
        result = np.zeros(24)
        idx = {"first": 0, "second": 1, "third": 2}.get(layer, 0)
        result[idx * 8 : (idx + 1) * 8] = np.asarray(e8_vector).ravel()[:8]
        return result

    def combine_e8_layers(
        self, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray
    ) -> np.ndarray:
        """Combine three 8D vectors into one 24D vector."""
        out = np.zeros(24)
        out[0:8] = np.asarray(v1).ravel()[:8]
        out[8:16] = np.asarray(v2).ravel()[:8]
        out[16:24] = np.asarray(v3).ravel()[:8]
        return out

    def theta_series_coefficient(self, n: int) -> int:
        """Number of Leech lattice points of norm 2n."""
        if 0 <= n < len(THETA_SERIES):
            return THETA_SERIES[n]
        raise ValueError(f"theta coefficient {n} not tabulated")


class LeechCodec:
    """Codec wrapper: quantize vectors, heal corrupted coset indices."""

    def __init__(self) -> None:
        self.lattice = LeechLattice()

    def quantize(self, vector: np.ndarray) -> LeechQuantization:
        return self.lattice.nearest_lattice_point(vector)

    def quantize_batch(self, vectors: List[np.ndarray]) -> List[LeechQuantization]:
        arr = np.array([self.lattice._ensure_24d(v) for v in vectors])
        points, g_idx, case, z, distances = self.lattice.batch_quantize(arr)
        shells = np.nan_to_num(
            np.round(np.sum(points**2, axis=1) / 2.0), nan=0.0, posinf=0.0
        ).astype(np.int64)
        return [
            LeechQuantization(
                lattice_point=points[i],
                index=int(g_idx[i]),
                case=int(case[i]),
                z=z[i],
                shell=int(shells[i]),
                error=float(distances[i]),
            )
            for i in range(len(vectors))
        ]

    @staticmethod
    def heal_index(corrupted_codeword: int) -> Optional[int]:
        """
        Error-correct a corrupted 24-bit Golay coset label.

        Up to 3 flipped bits are healed (returns the 12-bit message);
        4+ errors are detected and None is returned. This is the
        self-repair property the Glass Network uses to seal memory
        boundaries, applied to archive integrity.
        """
        decoded = decode_24(corrupted_codeword)
        return None if decoded is None else decoded[1]

    @staticmethod
    def index_to_codeword(g_idx: int) -> int:
        """Expand a 12-bit message to its protective 24-bit codeword."""
        return encode_12_to_24(g_idx)

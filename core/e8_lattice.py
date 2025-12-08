"""
E8 Lattice Core - Root System and Nearest Point Algorithms

The E8 lattice is an 8-dimensional even unimodular lattice with 240 root vectors.
It represents the densest sphere packing in 8D and has exceptional symmetry properties.

Mathematical Foundation:
- 240 root vectors of norm √2
- Two cosets: integer + half-integer coordinates
- Unique among lattices: optimal packing, kissing number 240

References:
- Gosset (1900): Discovery of E8 polytope
- Conway & Sloane: Sphere Packings, Lattices and Groups
"""

import numpy as np
from typing import Tuple, List, Optional
from functools import lru_cache
import hashlib
import itertools


# Safe norm function to avoid overflow
def _safe_norm(v: np.ndarray, axis: Optional[int] = None):
    """Compute norm safely, avoiding overflow."""
    v = np.asarray(v, dtype=np.float64)
    v = np.nan_to_num(v, nan=0.0, posinf=1e150, neginf=-1e150)

    if axis is None:
        max_val = np.max(np.abs(v))
        if max_val == 0:
            return 0.0
        if max_val > 1e150:
            scaled = v / max_val
            return min(max_val * np.sqrt(np.sum(scaled * scaled)), 1e300)
        return np.sqrt(np.sum(v * v))
    else:
        # For axis version, always use safe scaling to prevent overflow
        max_vals = np.max(np.abs(v), axis=axis, keepdims=True)
        max_vals = np.where(max_vals == 0, 1.0, max_vals)

        # Clip max_vals to prevent overflow in final multiply
        max_vals = np.clip(max_vals, 0, 1e100)

        # Always scale to prevent overflow in v*v
        scaled = v / max_vals
        # Compute result with overflow protection
        sqrt_sum = np.sqrt(np.sum(scaled * scaled, axis=axis))

        # Safe multiply: cap max_vals so multiply won't overflow
        max_squeezed = np.clip(max_vals.squeeze(axis), 0, 1e100)
        result = max_squeezed * sqrt_sum
        return np.clip(result, 0, 1e150)


class E8Lattice:
    """
    The E8 lattice for geometric compression.

    The 240 root vectors form the "alphabet" of fundamental concepts.
    Any 8D vector can be quantized to the nearest lattice point.
    """

    def __init__(self):
        self._roots = None
        self._root_norms = None

    @property
    def roots(self) -> np.ndarray:
        """Lazily generate and cache the 240 root vectors."""
        if self._roots is None:
            self._roots = self._generate_roots()
            self._root_norms = _safe_norm(self._roots, axis=1)
        return self._roots

    @staticmethod
    @lru_cache(maxsize=1)
    def _generate_roots_cached() -> np.ndarray:
        """Generate the 240 root vectors of E8 (cached)."""
        roots = []

        # Set 1: Permutations of (±1, ±1, 0, 0, 0, 0, 0, 0)
        # Choose 2 positions from 8, each with ±1 -> C(8,2) * 4 = 112 roots
        for i in range(8):
            for j in range(i + 1, 8):
                for s1 in [-1, 1]:
                    for s2 in [-1, 1]:
                        vec = np.zeros(8)
                        vec[i] = s1
                        vec[j] = s2
                        roots.append(vec)

        # Set 2: (±0.5, ±0.5, ..., ±0.5) with even number of minus signs
        # 2^8 = 256 combinations, half have even minus signs = 128 roots
        for signs in itertools.product([-0.5, 0.5], repeat=8):
            vec = np.array(signs)
            num_minus = np.sum(vec < 0)
            if num_minus % 2 == 0:
                roots.append(vec)

        return np.array(roots)

    def _generate_roots(self) -> np.ndarray:
        """Generate the 240 root vectors of E8."""
        return E8Lattice._generate_roots_cached()

    def nearest_root(self, vector: np.ndarray) -> Tuple[int, np.ndarray, float]:
        """
        Find the nearest E8 root to a vector.

        Args:
            vector: 8D numpy array

        Returns:
            (root_id, root_vector, distance)
        """
        vector = self._ensure_8d(vector)

        # Sanitize vector to prevent extreme values
        vector = np.nan_to_num(vector, nan=0.0, posinf=1e6, neginf=-1e6)
        vector = np.clip(vector, -1e6, 1e6)

        # Compute distances to all roots using safe norm
        diff = self.roots - vector
        # Clip the differences to prevent overflow in _safe_norm
        diff = np.clip(diff, -1e50, 1e50)
        distances = _safe_norm(diff, axis=1)

        # Find minimum
        idx = np.argmin(distances)

        return int(idx), self.roots[idx].copy(), float(distances[idx])

    def nearest_lattice_point(self, vector: np.ndarray) -> Tuple[int, np.ndarray, float]:
        """
        Find the nearest E8 lattice point (not just roots).

        Uses the parity-snap algorithm across two cosets:
        - Coset 0: Integer coordinates with even sum
        - Coset 1: Half-integer coordinates with even sum of (coord - 0.5)

        Args:
            vector: 8D numpy array

        Returns:
            (point_id, lattice_point, squared_distance)
        """
        vector = self._ensure_8d(vector)

        # Sanitize NaN/Inf values - be aggressive with large values
        vector = np.nan_to_num(vector, nan=0.0, posinf=1e6, neginf=-1e6)
        vector = np.clip(vector, -1e6, 1e6)

        # Coset 0: Round to nearest integer, enforce even sum
        y0 = np.rint(vector)
        y0_sum = np.sum(y0)
        if not np.isfinite(y0_sum):
            y0_sum = 0
        if int(y0_sum) % 2 != 0:
            # Flip the coordinate closest to half-integer
            j = int(np.argmax(np.abs(vector - y0)))
            y0[j] += 1 if vector[j] > y0[j] else -1

        # Coset 1: Round to nearest half-integer, enforce even sum
        y1 = np.rint(vector - 0.5) + 0.5
        y1_check = np.rint(y1 - 0.5)
        y1_sum = np.sum(y1_check)
        if not np.isfinite(y1_sum):
            y1_sum = 0
        if int(y1_sum) % 2 == 0:
            j = int(np.argmax(np.abs((vector - 0.5) - (y1 - 0.5))))
            y1[j] += 1 if (vector[j] - 0.5) > (y1[j] - 0.5) else -1

        # Choose closer coset
        d0 = float(np.clip(((vector - y0) ** 2).sum(), 0, 1e12))
        d1 = float(np.clip(((vector - y1) ** 2).sum(), 0, 1e12))

        if d0 <= d1:
            y, d = y0, d0
        else:
            y, d = y1, d1

        # Generate a unique ID from the lattice point
        point_id = self._hash_point(y)

        return point_id, y, d

    def _hash_point(self, point: np.ndarray) -> int:
        """Generate a unique integer ID for a lattice point."""
        # Convert to a hashable string
        point_bytes = point.tobytes()
        hash_hex = hashlib.md5(point_bytes).hexdigest()
        return int(hash_hex[:8], 16)

    def _ensure_8d(self, vector: np.ndarray) -> np.ndarray:
        """Ensure vector is 8D, padding or truncating as needed."""
        vector = np.asarray(vector, dtype=np.float64)

        if len(vector) == 8:
            return vector

        result = np.zeros(8)
        n = min(len(vector), 8)
        result[:n] = vector[:n]
        return result

    def batch_nearest_root(
        self, vectors: np.ndarray, use_gpu: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find nearest E8 root for a batch of vectors.

        GPU-accelerated when available and use_gpu=True.

        Args:
            vectors: (N, 8) array of vectors
            use_gpu: Whether to use GPU acceleration

        Returns:
            (indices, distances) - both of shape (N,)
        """
        vectors = np.asarray(vectors, dtype=np.float64)

        # Sanitize
        vectors = np.nan_to_num(vectors, nan=0.0, posinf=1e6, neginf=-1e6)
        vectors = np.clip(vectors, -1e6, 1e6)

        # Try GPU path
        if use_gpu:
            try:
                from e8zip.core.gpu_backend import get_gpu_ops, GPU_AVAILABLE

                if GPU_AVAILABLE:
                    gpu_ops = get_gpu_ops()
                    return gpu_ops.batch_nearest_root(vectors)
            except (ImportError, RuntimeError, Exception) as e:
                # GPU failed, fall back to CPU
                import logging

                logging.getLogger(__name__).debug(f"GPU fallback: {e}")

        # CPU path - vectorized (still fast for moderate sizes)
        return self._batch_nearest_root_cpu(vectors)

    def _batch_nearest_root_cpu(self, vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """CPU fallback for batch nearest root - vectorized."""
        roots = self.roots  # (240, 8)
        N = vectors.shape[0]

        # Process in chunks to avoid memory issues
        CHUNK_SIZE = 10000
        all_indices = []
        all_distances = []

        for i in range(0, N, CHUNK_SIZE):
            chunk = vectors[i : i + CHUNK_SIZE]

            # Compute distances: (chunk_size, 240)
            diff = chunk[:, None, :] - roots[None, :, :]
            distances = np.sqrt(np.sum(diff * diff, axis=2))

            indices = np.argmin(distances, axis=1)
            min_distances = np.min(distances, axis=1)

            all_indices.append(indices)
            all_distances.append(min_distances)

        return np.concatenate(all_indices), np.concatenate(all_distances)

    def batch_nearest_lattice_point(
        self, vectors: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Find nearest E8 lattice point for a batch of vectors.

        Uses vectorized parity-snap algorithm.

        Args:
            vectors: (N, 8) array of vectors

        Returns:
            (point_ids, lattice_points, distances)
        """
        vectors = np.asarray(vectors, dtype=np.float64)
        N = vectors.shape[0]

        # Sanitize
        vectors = np.nan_to_num(vectors, nan=0.0, posinf=1e6, neginf=-1e6)
        vectors = np.clip(vectors, -1e6, 1e6)

        # Coset 0: Round to nearest integer, enforce even sum
        y0 = np.rint(vectors)
        sums0 = np.sum(y0, axis=1)
        odd_mask = (sums0.astype(int) % 2) != 0

        # For odd sums, flip the coordinate closest to half-integer
        if np.any(odd_mask):
            residuals = np.abs(vectors[odd_mask] - y0[odd_mask])
            flip_idx = np.argmax(residuals, axis=1)
            for i, idx in enumerate(np.where(odd_mask)[0]):
                j = flip_idx[i]
                y0[idx, j] += 1 if vectors[idx, j] > y0[idx, j] else -1

        # Coset 1: Round to nearest half-integer, enforce even sum
        y1 = np.rint(vectors - 0.5) + 0.5
        y1_check = np.rint(y1 - 0.5)
        sums1 = np.sum(y1_check, axis=1)
        even_mask = (sums1.astype(int) % 2) == 0

        if np.any(even_mask):
            residuals = np.abs((vectors[even_mask] - 0.5) - (y1[even_mask] - 0.5))
            flip_idx = np.argmax(residuals, axis=1)
            for i, idx in enumerate(np.where(even_mask)[0]):
                j = flip_idx[i]
                y1[idx, j] += 1 if (vectors[idx, j] - 0.5) > (y1[idx, j] - 0.5) else -1

        # Compute distances
        d0 = np.sum((vectors - y0) ** 2, axis=1)
        d1 = np.sum((vectors - y1) ** 2, axis=1)

        # Choose closer coset
        use_y0 = d0 <= d1

        lattice_points = np.where(use_y0[:, None], y0, y1)
        distances = np.where(use_y0, d0, d1)

        # Generate point IDs
        point_ids = np.array([self._hash_point(p) for p in lattice_points])

        return point_ids, lattice_points, distances

    def project_to_8d(self, vector: np.ndarray) -> np.ndarray:
        """
        Project a high-dimensional vector to 8D.

        Uses a random projection (Johnson-Lindenstrauss style)
        that preserves distances with high probability.
        """
        vector = np.asarray(vector, dtype=np.float64)

        if len(vector) <= 8:
            return self._ensure_8d(vector)

        # Use a deterministic "random" projection based on dimension
        np.random.seed(42)  # Reproducible
        d = len(vector)
        projection_matrix = np.random.randn(8, d) / np.sqrt(d)

        return projection_matrix @ vector

    def compute_gap(self, vector: np.ndarray) -> float:
        """Compute squared distance to nearest E8 lattice point."""
        _, _, gap = self.nearest_lattice_point(vector)
        return gap

    def golden_ratio_weight(
        self, k: float, amplitude: float = 0.02, k_star: float = 0.05, phase: float = 0.0
    ) -> float:
        """
        Golden-ratio log-periodic multiplier.

        M(k) = 1 + A * cos[(2π/ln φ) * ln(k/k★) + φ0]

        This creates oscillations at golden-ratio intervals,
        useful for natural importance weighting.
        """
        phi = (1 + np.sqrt(5)) / 2.0  # Golden ratio
        omega = 2.0 * np.pi / np.log(phi)

        k = max(k, 1e-12)  # Avoid log(0)

        return 1.0 + amplitude * np.cos(omega * np.log(k / k_star) + phase)


# Convenience functions
_default_lattice = E8Lattice()


def nearest_e8_point(vector: np.ndarray) -> Tuple[int, np.ndarray, float]:
    """Find nearest E8 lattice point to vector."""
    return _default_lattice.nearest_lattice_point(vector)


def get_e8_roots() -> np.ndarray:
    """Get the 240 E8 root vectors."""
    return _default_lattice.roots.copy()


def compute_e8_gap(vector: np.ndarray) -> float:
    """Compute squared distance to nearest E8 lattice point."""
    return _default_lattice.compute_gap(vector)

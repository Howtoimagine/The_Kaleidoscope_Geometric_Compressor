"""
E8 Geometric Codec

Vector quantization using the E8 lattice.
Maps continuous vectors to discrete lattice points,
providing a "geometric alphabet" for compression.

The 240 E8 root vectors form the fundamental symbols.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

from e8zip.core.e8_lattice import E8Lattice

# Import safe norm to avoid overflow
try:
    from e8zip.utils.math_utils import safe_norm
except ImportError:

    def safe_norm(v, axis=None):
        v = np.nan_to_num(v, nan=0.0, posinf=1e150, neginf=-1e150)
        if axis is None:
            return np.sqrt(np.sum(v * v))
        return np.sqrt(np.sum(v * v, axis=axis))


@dataclass
class QuantizationResult:
    """Result of vector quantization."""

    node_id: int
    lattice_point: np.ndarray
    error: float
    is_root: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "lattice_point": self.lattice_point.tolist(),
            "error": self.error,
            "is_root": self.is_root,
        }


class E8GeometricCodec:
    """
    Vector quantization using the E8 lattice.

    Maps continuous 8D vectors to the nearest E8 lattice point.
    This provides a discrete "alphabet" of fundamental concepts.

    The codec supports multiple quantization modes:
    - root: Quantize to nearest of 240 roots only
    - lattice: Quantize to any E8 lattice point
    - scaled: Quantize with scale normalization
    """

    def __init__(self, mode: str = "lattice"):
        """
        Initialize the codec.

        Args:
            mode: 'root' for 240 roots only, 'lattice' for any lattice point
        """
        self.lattice = E8Lattice()
        self.mode = mode

        # Precompute root information
        self._roots = self.lattice.roots
        self._root_count = len(self._roots)

    def quantize(self, vector: np.ndarray) -> QuantizationResult:
        """
        Quantize vector to nearest E8 lattice point.

        Args:
            vector: Input vector (any dimension, will be projected to 8D)

        Returns:
            QuantizationResult with node_id, lattice_point, and error
        """
        # Project to 8D if needed
        if len(vector) != 8:
            vector = self.lattice.project_to_8d(vector)

        if self.mode == "root":
            return self._quantize_to_root(vector)
        else:
            return self._quantize_to_lattice(vector)

    def _quantize_to_root(self, vector: np.ndarray) -> QuantizationResult:
        """Quantize to nearest of 240 roots."""
        node_id, root, error = self.lattice.nearest_root(vector)
        return QuantizationResult(node_id=node_id, lattice_point=root, error=error, is_root=True)

    def _quantize_to_lattice(self, vector: np.ndarray) -> QuantizationResult:
        """Quantize to nearest lattice point."""
        node_id, point, error = self.lattice.nearest_lattice_point(vector)

        # Check if it's also a root
        is_root = any(np.allclose(point, root) for root in self._roots)

        return QuantizationResult(
            node_id=node_id, lattice_point=point, error=error, is_root=is_root
        )

    def encode(self, vector: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Encode a vector as (node_id, residual).

        The residual is the difference between the original vector
        and the quantized point, enabling reconstruction.
        """
        result = self.quantize(vector)

        # Ensure 8D for residual computation
        if len(vector) != 8:
            vector = self.lattice._ensure_8d(vector)

        residual = vector - result.lattice_point

        return result.node_id, residual

    def decode(
        self, node_id: int, residual: np.ndarray, lattice_point: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Decode (node_id, residual) back to vector.

        If lattice_point is provided, use it directly.
        Otherwise, we need to reconstruct it (requires stored mapping).
        """
        if lattice_point is not None:
            return lattice_point + residual
        else:
            # For roots, we can look up by ID
            if 0 <= node_id < self._root_count:
                return self._roots[node_id] + residual
            else:
                raise ValueError(f"Cannot decode node_id {node_id} without lattice_point")

    def conceptual_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Compute 'conceptual distance' between two vectors.

        This is the Euclidean distance in the lattice space.
        Could be extended to use lattice-aware metrics.
        """
        v1 = self.lattice._ensure_8d(v1)
        v2 = self.lattice._ensure_8d(v2)
        return float(safe_norm(v1 - v2))

    def batch_quantize(self, vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Quantize a batch of vectors.

        Args:
            vectors: Shape (N, D) array

        Returns:
            node_ids: Shape (N,) array of integers
            lattice_points: Shape (N, 8) array
            errors: Shape (N,) array of floats
        """
        n = len(vectors)
        node_ids = np.zeros(n, dtype=np.int64)
        lattice_points = np.zeros((n, 8), dtype=np.float64)
        errors = np.zeros(n, dtype=np.float64)

        for i, vec in enumerate(vectors):
            result = self.quantize(vec)
            node_ids[i] = result.node_id
            lattice_points[i] = result.lattice_point
            errors[i] = result.error

        return node_ids, lattice_points, errors


class SemanticLossyCodec:
    """
    Semantic lossy compression with controlled "mythic degradation".

    Allows trading fidelity for compression by interpolating
    between the literal vector and its archetypal (lattice) form.

    High temperature = more loss allowed (more 'mythic', less literal)
    Low temperature = strict fidelity
    """

    def __init__(self, base_codec: Optional[E8GeometricCodec] = None):
        self.codec = base_codec or E8GeometricCodec()

    def compress_with_temperature(
        self, vector: np.ndarray, mythic_temperature: float = 0.5
    ) -> np.ndarray:
        """
        Compress a vector with mythic temperature.

        Args:
            vector: Input vector
            mythic_temperature: 0.0 = keep original, 1.0 = snap to lattice

        Returns:
            Compressed vector (interpolation between original and archetype)
        """
        vector = self.codec.lattice._ensure_8d(vector)

        # Find the archetype (nearest lattice point)
        result = self.codec.quantize(vector)
        archetype = result.lattice_point

        # Interpolate based on temperature
        # T=0 -> Literal (Original)
        # T=1 -> Archetypal (Lattice Point)
        compressed = (1 - mythic_temperature) * vector + mythic_temperature * archetype

        return compressed

    def batch_compress_with_temperature(
        self, vectors: np.ndarray, mythic_temperature: float = 0.5
    ) -> np.ndarray:
        """
        Batch version of compress_with_temperature.

        Args:
            vectors: (N, 8) array of vectors
            mythic_temperature: 0.0 = keep original, 1.0 = snap to lattice

        Returns:
            Compressed vectors (N, 8)
        """
        # Find archetypes (nearest lattice points) - vectorized
        _, archetypes, _ = self.codec.lattice.batch_nearest_lattice_point(vectors)

        # Interpolate
        # vectors: (N, 8), archetypes: (N, 8)
        compressed = (1 - mythic_temperature) * vectors + mythic_temperature * archetypes

        return compressed

    def extract_invariants(self, vector: np.ndarray) -> Dict[str, float]:
        """
        Extract topological invariants that should be preserved.

        These are properties that remain constant regardless of
        mythic temperature.
        """
        vector = self.codec.lattice._ensure_8d(vector)

        norm = float(safe_norm(vector))
        projection = float(np.sum(vector))  # Projection onto main diagonal

        # Nearest root information
        result = self.codec.quantize(vector)

        return {
            "norm": norm,
            "projection": projection,
            "nearest_root_id": result.node_id,
            "quantization_error": result.error,
        }

    def adaptive_temperature(self, vectors: np.ndarray, target_ratio: float = 5.0) -> float:
        """
        Compute optimal mythic temperature for target compression ratio.

        Args:
            vectors: Batch of vectors to analyze
            target_ratio: Desired compression ratio

        Returns:
            Optimal temperature value
        """
        # Analyze the variance of the vectors
        variance = np.var(vectors)

        # Higher variance = need more temperature for same compression
        # This is a heuristic formula
        base_temp = 0.5
        variance_factor = np.log1p(variance) / 10
        ratio_factor = np.log(target_ratio) / np.log(100)  # Scale to [0, 1]

        temperature = np.clip(base_temp * ratio_factor + variance_factor, 0.0, 1.0)

        return float(temperature)

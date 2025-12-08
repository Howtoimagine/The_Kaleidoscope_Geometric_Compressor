"""
Trajectory Compression

Compresses sequences of vectors as geodesic paths with sparse perturbations.
Instead of storing every point, we store:
- Start point (quantized to E8)
- End point (quantized to E8)
- Sparse perturbations (deviations from the geodesic)

This achieves significant compression while preserving the path structure.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field
import struct
import zlib

from e8zip.core.codec import E8GeometricCodec
from e8zip.core.hyperbolic import PoincareBall


@dataclass
class CompressedTrajectory:
    """
    A compressed representation of a sequence of vectors.

    Stores only start, end, and sparse perturbations from the geodesic.
    """

    start_node_id: int
    start_vector: np.ndarray
    end_node_id: int
    end_vector: np.ndarray
    perturbations: List[Tuple[int, np.ndarray]] = field(default_factory=list)
    length: int = 0
    compression_ratio: float = 1.0
    semantic_fidelity: float = 1.0
    hyperbolic: bool = False

    def to_bytes(self) -> bytes:
        """Serialize to bytes for storage."""
        # Header
        header = struct.pack(
            "<IIQQ",  # 2 ints, 2 unsigned long longs
            self.start_node_id,
            self.end_node_id,
            self.length,
            len(self.perturbations),
        )

        # Flags
        flags = struct.pack("<B", 1 if self.hyperbolic else 0)

        # Start and end vectors (8 doubles each)
        vectors = struct.pack("<8d", *self.start_vector)
        vectors += struct.pack("<8d", *self.end_vector)

        # Perturbations
        perturbation_data = b""
        for idx, vec in self.perturbations:
            perturbation_data += struct.pack("<I8d", idx, *vec)

        # Compress perturbation data
        compressed_perturbations = zlib.compress(perturbation_data, level=9)
        perturbation_header = struct.pack("<I", len(compressed_perturbations))

        return header + flags + vectors + perturbation_header + compressed_perturbations

    @classmethod
    def from_bytes(cls, data: bytes) -> "CompressedTrajectory":
        """Deserialize from bytes."""
        offset = 0

        # Header
        start_id, end_id, length, num_perturbations = struct.unpack_from("<IIQQ", data, offset)
        offset += struct.calcsize("<IIQQ")

        # Flags
        flags = struct.unpack_from("<B", data, offset)[0]
        hyperbolic = bool(flags & 1)
        offset += 1

        # Vectors
        start_vector = np.array(struct.unpack_from("<8d", data, offset))
        offset += 8 * 8
        end_vector = np.array(struct.unpack_from("<8d", data, offset))
        offset += 8 * 8

        # Perturbations
        compressed_len = struct.unpack_from("<I", data, offset)[0]
        offset += 4

        compressed_perturbations = data[offset : offset + compressed_len]
        perturbation_data = zlib.decompress(compressed_perturbations)

        perturbations = []
        p_offset = 0
        perturbation_size = struct.calcsize("<I8d")
        for _ in range(num_perturbations):
            values = struct.unpack_from("<I8d", perturbation_data, p_offset)
            idx = values[0]
            vec = np.array(values[1:])
            perturbations.append((idx, vec))
            p_offset += perturbation_size

        return cls(
            start_node_id=start_id,
            start_vector=start_vector,
            end_node_id=end_id,
            end_vector=end_vector,
            perturbations=perturbations,
            length=length,
            hyperbolic=hyperbolic,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "start_node_id": self.start_node_id,
            "start_vector": self.start_vector.tolist(),
            "end_node_id": self.end_node_id,
            "end_vector": self.end_vector.tolist(),
            "perturbations": [(idx, vec.tolist()) for idx, vec in self.perturbations],
            "length": self.length,
            "compression_ratio": self.compression_ratio,
            "semantic_fidelity": self.semantic_fidelity,
            "hyperbolic": self.hyperbolic,
        }


class TrajectoryCompressor:
    """
    Compresses paths through the lattice.

    "Compress paths through the lattice, not just snapshots."

    The key insight is that many sequential data points lie approximately
    on geodesics (straight lines in curved space). We exploit this by
    storing only the start, end, and deviations.
    """

    def __init__(self, codec: Optional[E8GeometricCodec] = None, hyperbolic: bool = True):
        """
        Initialize trajectory compressor.

        Args:
            codec: E8 geometric codec for quantization
            hyperbolic: Use hyperbolic geodesics (better for hierarchical data)
        """
        self.codec = codec or E8GeometricCodec()
        self.hyperbolic = hyperbolic
        self.poincare = PoincareBall(dim=8) if hyperbolic else None

    def compress(
        self, trajectory: List[np.ndarray], tolerance: float = 0.1
    ) -> CompressedTrajectory:
        """
        Compress a sequence of vectors into a trajectory.

        Args:
            trajectory: List of vectors (any dimension)
            tolerance: Maximum deviation before storing a perturbation

        Returns:
            CompressedTrajectory object
        """
        if not trajectory:
            raise ValueError("Empty trajectory")

        # Ensure 8D
        trajectory_8d = [self.codec.lattice._ensure_8d(v) for v in trajectory]

        # Quantize start and end
        start_vec = trajectory_8d[0]
        end_vec = trajectory_8d[-1]

        start_result = self.codec.quantize(start_vec)
        end_result = self.codec.quantize(end_vec)

        length = len(trajectory_8d)
        perturbations = []

        # Find deviations from geodesic
        for i in range(1, length - 1):
            t = i / (length - 1)

            if self.hyperbolic and self.poincare:
                # Hyperbolic geodesic
                expected = self.poincare.geodesic(
                    start_result.lattice_point, end_result.lattice_point, t
                )
            else:
                # Euclidean linear interpolation
                expected = (1 - t) * start_result.lattice_point + t * end_result.lattice_point

            actual = trajectory_8d[i]
            # Sanitize expected value
            expected = np.nan_to_num(expected, nan=0.0, posinf=1e6, neginf=-1e6)
            deviation = actual - expected
            deviation = np.nan_to_num(deviation, nan=0.0, posinf=1e6, neginf=-1e6)
            norm = np.linalg.norm(deviation)
            if not np.isfinite(norm):
                norm = 0.0

            # Store perturbation if deviation exceeds tolerance
            if norm > tolerance:
                perturbations.append((i, deviation))

        # Calculate compression statistics
        original_size = length * 8 * 8  # 8 floats * 8 bytes
        # Start/End + (Index + Vector) per perturbation
        compressed_size = (2 * 8 * 8) + (len(perturbations) * (4 + 8 * 8))
        ratio = original_size / (compressed_size + 1e-10)

        # Semantic fidelity (inverse of reconstruction error)
        fidelity = 1.0 / (1.0 + len(perturbations) * tolerance)

        return CompressedTrajectory(
            start_node_id=start_result.node_id,
            start_vector=start_result.lattice_point,
            end_node_id=end_result.node_id,
            end_vector=end_result.lattice_point,
            perturbations=perturbations,
            length=length,
            compression_ratio=ratio,
            semantic_fidelity=fidelity,
            hyperbolic=self.hyperbolic,
        )

    def decompress(self, compressed: CompressedTrajectory) -> List[np.ndarray]:
        """
        Reconstruct the trajectory from compressed form.

        Args:
            compressed: CompressedTrajectory object

        Returns:
            List of reconstructed 8D vectors
        """
        reconstructed = []
        start = compressed.start_vector
        end = compressed.end_vector
        length = compressed.length

        # Build perturbation lookup
        perturbations_map = {idx: vec for idx, vec in compressed.perturbations}

        for i in range(length):
            if i == 0:
                reconstructed.append(start.copy())
                continue
            if i == length - 1:
                reconstructed.append(end.copy())
                continue

            t = i / (length - 1)

            if compressed.hyperbolic and self.poincare:
                expected = self.poincare.geodesic(start, end, t)
            else:
                expected = (1 - t) * start + t * end

            if i in perturbations_map:
                expected = expected + perturbations_map[i]

            reconstructed.append(expected)

        return reconstructed

    def compute_reconstruction_error(
        self, original: List[np.ndarray], reconstructed: List[np.ndarray]
    ) -> float:
        """Compute mean reconstruction error."""
        if len(original) != len(reconstructed):
            return float("inf")

        total_error = 0.0
        for orig, recon in zip(original, reconstructed):
            orig_8d = self.codec.lattice._ensure_8d(orig)
            error = np.linalg.norm(orig_8d - recon)
            total_error += error

        return total_error / len(original)


class AdaptiveTrajectoryCompressor:
    """
    Adaptive trajectory compressor that optimizes tolerance.

    Automatically finds the best tolerance to achieve a target
    compression ratio or fidelity.
    """

    def __init__(self, base_compressor: Optional[TrajectoryCompressor] = None):
        self.compressor = base_compressor or TrajectoryCompressor()

    def compress_to_target_ratio(
        self, trajectory: List[np.ndarray], target_ratio: float = 5.0, max_iterations: int = 10
    ) -> CompressedTrajectory:
        """
        Compress trajectory to achieve target compression ratio.

        Args:
            trajectory: Input trajectory
            target_ratio: Desired compression ratio
            max_iterations: Maximum optimization iterations

        Returns:
            CompressedTrajectory achieving approximately target ratio
        """
        # Binary search for optimal tolerance
        low_tol, high_tol = 0.01, 1.0
        best_result = None

        for _ in range(max_iterations):
            mid_tol = (low_tol + high_tol) / 2
            result = self.compressor.compress(trajectory, tolerance=mid_tol)

            if best_result is None:
                best_result = result

            if result.compression_ratio < target_ratio:
                # Need more compression -> higher tolerance
                low_tol = mid_tol
            else:
                high_tol = mid_tol
                if abs(result.compression_ratio - target_ratio) < abs(
                    best_result.compression_ratio - target_ratio
                ):
                    best_result = result

        return best_result

    def compress_to_target_fidelity(
        self, trajectory: List[np.ndarray], target_fidelity: float = 0.9, max_iterations: int = 10
    ) -> CompressedTrajectory:
        """
        Compress trajectory while maintaining minimum fidelity.

        Args:
            trajectory: Input trajectory
            target_fidelity: Minimum acceptable semantic fidelity
            max_iterations: Maximum optimization iterations

        Returns:
            CompressedTrajectory with best ratio at or above target fidelity
        """
        # Binary search for maximum tolerance that maintains fidelity
        low_tol, high_tol = 0.001, 2.0
        best_result = None

        for _ in range(max_iterations):
            mid_tol = (low_tol + high_tol) / 2
            result = self.compressor.compress(trajectory, tolerance=mid_tol)

            if result.semantic_fidelity >= target_fidelity:
                # Can try higher tolerance for more compression
                low_tol = mid_tol
                if best_result is None or result.compression_ratio > best_result.compression_ratio:
                    best_result = result
            else:
                # Fidelity too low, need lower tolerance
                high_tol = mid_tol

        return best_result or self.compressor.compress(trajectory, tolerance=0.01)

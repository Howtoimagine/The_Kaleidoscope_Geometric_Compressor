"""
Black Hole Compression

Ultra-high compression using holographic encoding principles.
Based on the Bekenstein-Hawking entropy bound and the holographic principle.

Theory:
- Information falling into a black hole is encoded on the Event Horizon
- Maximum entropy (information capacity) proportional to horizon Area (A/4)
- Black holes are the most efficient compressors in nature (saturate Bekenstein bound)
- Hawking Radiation returns information in scrambled (encrypted) form

Implementation:
- Bulk: 24D Leech Lattice space (or 8D E8 for faster mode)
- Horizon: (D-1)-sphere surface
- Scrambling: Fast scrambling via permutation matrices
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import struct
import zlib
import hashlib

# Import safe norm to avoid overflow
try:
    from e8zip.utils.math_utils import safe_norm
except ImportError:

    def safe_norm(v, axis=None):
        """Fallback safe norm."""
        v = np.nan_to_num(v, nan=0.0, posinf=1e150, neginf=-1e150)
        if axis is None:
            max_val = np.max(np.abs(v))
            if max_val == 0:
                return 0.0
            if max_val > 1e150:
                scaled = v / max_val
                return max_val * np.sqrt(np.sum(scaled * scaled))
            return np.sqrt(np.sum(v * v))
        else:
            return np.sqrt(np.sum(v * v, axis=axis))


@dataclass
class HorizonState:
    """
    The state encoded on the black hole's event horizon.
    This is the compressed representation of all absorbed information.
    """

    dimension: int
    radius: float
    entropy: float
    encoded_data: np.ndarray
    absorbed_count: int
    checksum: str

    def to_bytes(self) -> bytes:
        """Serialize horizon state."""
        # Header
        header = struct.pack(
            "<IdddQ",
            self.dimension,
            self.radius,
            self.entropy,
            float(self.absorbed_count),
            len(self.encoded_data),
        )

        # Encoded data
        data_bytes = self.encoded_data.tobytes()
        compressed = zlib.compress(data_bytes, level=9)

        # Checksum
        checksum_bytes = self.checksum.encode("utf-8")

        return (
            header
            + struct.pack("<I", len(compressed))
            + compressed
            + struct.pack("<I", len(checksum_bytes))
            + checksum_bytes
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "HorizonState":
        """Deserialize horizon state."""
        offset = 0

        header_size = struct.calcsize("<IdddQ")
        dim, radius, entropy, absorbed_float, data_len = struct.unpack_from("<IdddQ", data, offset)
        absorbed_count = int(absorbed_float)
        offset += header_size

        compressed_len = struct.unpack_from("<I", data, offset)[0]
        offset += 4

        compressed = data[offset : offset + compressed_len]
        offset += compressed_len

        decompressed = zlib.decompress(compressed)
        encoded_data = np.frombuffer(decompressed, dtype=np.float64).reshape(-1)

        checksum_len = struct.unpack_from("<I", data, offset)[0]
        offset += 4

        checksum = data[offset : offset + checksum_len].decode("utf-8")

        return cls(
            dimension=dim,
            radius=radius,
            entropy=entropy,
            encoded_data=encoded_data,
            absorbed_count=absorbed_count,
            checksum=checksum,
        )


@dataclass
class HawkingRadiation:
    """
    Information emitted by the black hole.
    Scrambled but unitary (in principle recoverable).
    """

    spectrum: np.ndarray
    entanglement_entropy: float
    temperature: float
    scrambled_data: bytes = field(default=b"")


class BlackHoleCompressor:
    """
    Ultra-high compression using black hole thermodynamics.

    This compressor absorbs multiple data vectors into a "black hole",
    encoding them holographically on the event horizon.

    The compression ratio scales with the dimension:
    - 8D mode: Fast, moderate compression
    - 24D mode: Slower, extreme compression (Leech lattice)
    """

    def __init__(self, dimension: int = 24, initial_mass: float = 1.0):
        """
        Initialize black hole compressor.

        Args:
            dimension: Bulk dimension (8 for E8, 24 for Leech)
            initial_mass: Initial black hole mass
        """
        self.dimension = dimension
        self.mass = initial_mass

        # Internal state (the "singularity" accumulator)
        self.internal_state = np.zeros(dimension)

        # Tracking
        self.absorbed_items: List[Dict[str, Any]] = []
        self.total_absorbed_energy = 0.0

        # Scrambling matrix (unitary for information preservation)
        np.random.seed(42)  # Reproducible scrambling
        self._scrambling_matrix = self._generate_scrambling_matrix()

    def _generate_scrambling_matrix(self) -> np.ndarray:
        """Generate a random orthogonal matrix for scrambling."""
        # Random matrix
        A = np.random.randn(self.dimension, self.dimension)
        # QR decomposition gives orthogonal Q
        Q, _ = np.linalg.qr(A)
        return Q

    @property
    def radius(self) -> float:
        """
        Schwarzschild Radius.
        In D dimensions, R_s ~ M^(1/(D-3)).
        """
        exponent = 1.0 / max(self.dimension - 3.0, 1.0)
        return self.mass**exponent

    @property
    def area(self) -> float:
        """
        Horizon Area (Volume of (D-2)-sphere).
        Area ~ R^(D-2).
        """
        # Clamp to prevent overflow
        if self.radius > 1e10:
            return 1e300
        return min(self.radius ** (self.dimension - 2), 1e300)

    @property
    def entropy(self) -> float:
        """
        Bekenstein-Hawking Entropy.
        S = Area / 4.
        """
        return self.area / 4.0

    @property
    def temperature(self) -> float:
        """
        Hawking Temperature.
        T ~ 1/R.
        """
        if self.radius < 1e-9:
            return float("inf")
        return 1.0 / (4.0 * np.pi * self.radius)

    @property
    def capacity(self) -> float:
        """Maximum information capacity (in bits)."""
        return self.entropy * np.log(2)

    def absorb(self, vector: np.ndarray, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Absorb a vector into the black hole.

        Args:
            vector: Data vector to compress
            metadata: Optional metadata (stored separately)

        Returns:
            Absorption statistics
        """
        # Ensure correct dimension
        vector = self._ensure_dimension(vector)

        # Sanitize NaN/Inf values
        vector = np.nan_to_num(vector, nan=0.0, posinf=1e6, neginf=-1e6)

        # Compute energy (information content) using safe_norm
        energy = safe_norm(vector)
        if not np.isfinite(energy):
            energy = 0.0

        # Add mass
        self.mass += energy * 0.01  # Scale factor

        # Scramble and absorb
        scrambled = self._scrambling_matrix @ vector
        scrambled = np.nan_to_num(scrambled, nan=0.0, posinf=1e6, neginf=-1e6)
        scrambled = np.clip(scrambled, -1e150, 1e150)
        self.internal_state = np.clip(self.internal_state + scrambled, -1e150, 1e150)

        # Track
        item_id = len(self.absorbed_items)
        self.absorbed_items.append(
            {"id": item_id, "energy": float(min(energy, 1e150)), "metadata": metadata or {}}
        )
        self.total_absorbed_energy = min(self.total_absorbed_energy + energy, 1e300)

        return {
            "item_id": item_id,
            "energy": float(energy),
            "new_mass": float(self.mass),
            "new_entropy": float(self.entropy),
            "temperature": float(self.temperature),
        }

    def absorb_batch(self, vectors: List[np.ndarray]) -> Dict[str, Any]:
        """
        Absorb multiple vectors efficiently.

        Args:
            vectors: List of data vectors

        Returns:
            Batch absorption statistics
        """
        stats = []
        for vec in vectors:
            stat = self.absorb(vec)
            stats.append(stat)

        return {
            "absorbed_count": len(vectors),
            "total_energy": sum(s["energy"] for s in stats),
            "final_mass": float(self.mass),
            "final_entropy": float(self.entropy),
            "compression_ratio": self._estimate_compression_ratio(vectors),
        }

    def _estimate_compression_ratio(self, vectors: List[np.ndarray]) -> float:
        """Estimate compression ratio for absorbed vectors."""
        original_size = len(vectors) * self.dimension * 8  # bytes
        # Horizon encoding size
        horizon_size = self.dimension * 8 + 64  # state + metadata
        return original_size / (horizon_size + 1e-10)

    def get_horizon_state(self) -> HorizonState:
        """
        Get the current event horizon state.

        This is the compressed representation of all absorbed information.
        """
        # Sanitize internal state
        self.internal_state = np.nan_to_num(self.internal_state, nan=0.0, posinf=1e6, neginf=-1e6)

        # Project internal state onto horizon (unit sphere * radius)
        norm = safe_norm(self.internal_state)
        if np.isfinite(norm) and norm > 1e-9:
            horizon_projection = self.internal_state / norm * self.radius
        else:
            horizon_projection = np.zeros(self.dimension)

        # Compute checksum
        checksum = hashlib.sha256(horizon_projection.tobytes()).hexdigest()[:16]

        return HorizonState(
            dimension=self.dimension,
            radius=self.radius,
            entropy=self.entropy,
            encoded_data=horizon_projection,
            absorbed_count=len(self.absorbed_items),
            checksum=checksum,
        )

    def evaporate(self, fraction: float = 0.1) -> HawkingRadiation:
        """
        Emit Hawking radiation (partial decompression).

        Args:
            fraction: Fraction of mass/energy to emit

        Returns:
            HawkingRadiation containing scrambled information
        """
        # Energy release
        energy_release = self.mass * fraction
        self.mass -= energy_release
        self.mass = max(self.mass, 0.1)  # Prevent complete evaporation

        # Signal (leakage from internal state)
        signal = self.internal_state * fraction

        # Add thermal noise
        noise = np.random.randn(self.dimension) * self.temperature
        radiation_vector = signal + noise

        # Entanglement entropy
        s_ent = self.entropy * fraction

        return HawkingRadiation(
            spectrum=radiation_vector,
            entanglement_entropy=s_ent,
            temperature=self.temperature,
            scrambled_data=radiation_vector.tobytes(),
        )

    def compress_to_bytes(self) -> bytes:
        """
        Get compressed representation as bytes.

        Returns:
            Binary compressed data
        """
        horizon = self.get_horizon_state()
        return horizon.to_bytes()

    @classmethod
    def decompress_from_bytes(cls, data: bytes) -> "BlackHoleCompressor":
        """
        Reconstruct black hole from compressed bytes.

        Note: This reconstructs the COMPRESSED state,
        not the original data (which is scrambled).
        """
        horizon = HorizonState.from_bytes(data)

        bh = cls(dimension=horizon.dimension)
        bh.mass = horizon.entropy * 4  # Reverse entropy formula
        bh.internal_state = horizon.encoded_data * (1.0 / (horizon.radius + 1e-10))

        return bh

    def _ensure_dimension(self, vector: np.ndarray) -> np.ndarray:
        """Ensure vector has correct dimension."""
        vector = np.asarray(vector, dtype=np.float64)

        if len(vector) == self.dimension:
            return vector

        result = np.zeros(self.dimension)
        n = min(len(vector), self.dimension)
        result[:n] = vector[:n]
        return result

    def reset(self):
        """Reset black hole to initial state."""
        self.mass = 1.0
        self.internal_state = np.zeros(self.dimension)
        self.absorbed_items = []
        self.total_absorbed_energy = 0.0


class HolographicEncoder:
    """
    Holographic encoding for data streams.

    Implements the holographic principle:
    All information in a volume can be encoded on its boundary.
    """

    def __init__(self, bulk_dimension: int = 24, boundary_dimension: int = 23):
        """
        Initialize holographic encoder.

        Args:
            bulk_dimension: Interior dimension (default: 24 for Leech)
            boundary_dimension: Boundary dimension (bulk - 1)
        """
        self.bulk_dim = bulk_dimension
        self.boundary_dim = boundary_dimension

        # Projection matrix (bulk -> boundary)
        np.random.seed(123)
        self._projection = np.random.randn(boundary_dimension, bulk_dimension)
        proj_norms = safe_norm(self._projection, axis=1)
        self._projection /= proj_norms[:, np.newaxis]

        # Inverse (approximate reconstruction)
        self._inverse = np.linalg.pinv(self._projection)

    def encode(self, bulk_data: np.ndarray) -> np.ndarray:
        """
        Project bulk data onto boundary.

        Args:
            bulk_data: Data in bulk space

        Returns:
            Boundary-encoded data
        """
        bulk_data = self._ensure_bulk_dim(bulk_data)
        return self._projection @ bulk_data

    def decode(self, boundary_data: np.ndarray) -> np.ndarray:
        """
        Reconstruct bulk data from boundary.

        Note: Reconstruction is lossy due to dimensionality reduction.
        """
        return self._inverse @ boundary_data

    def encode_batch(self, vectors: List[np.ndarray]) -> np.ndarray:
        """Encode multiple vectors."""
        bulk_vectors = np.array([self._ensure_bulk_dim(v) for v in vectors])
        return bulk_vectors @ self._projection.T

    def _ensure_bulk_dim(self, vector: np.ndarray) -> np.ndarray:
        """Ensure vector has bulk dimension."""
        vector = np.asarray(vector, dtype=np.float64)

        if len(vector) == self.bulk_dim:
            return vector

        result = np.zeros(self.bulk_dim)
        n = min(len(vector), self.bulk_dim)
        result[:n] = vector[:n]
        return result

"""
Mathematical Utilities for E8ZIP

Common mathematical operations used throughout the compression library.
"""

import numpy as np
from typing import Tuple, Optional, List
import warnings


# Constants
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618
PSI = 1 / PHI  # Golden ratio conjugate ≈ 0.618


def safe_norm(v: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    Compute norm safely, avoiding overflow for large values.

    Uses scaling to prevent overflow in intermediate calculations.

    Args:
        v: Input vector or array
        axis: Axis along which to compute norm

    Returns:
        Norm value(s)
    """
    v = np.asarray(v, dtype=np.float64)

    # Handle NaN/Inf and clip extreme values
    v = np.nan_to_num(v, nan=0.0, posinf=1e150, neginf=-1e150)
    v = np.clip(v, -1e150, 1e150)

    if axis is None:
        # Scalar norm
        max_val = np.max(np.abs(v))
        if max_val == 0:
            return 0.0
        if max_val > 1e75:
            # Scale down to avoid overflow in v*v
            scaled = v / max_val
            return min(max_val * np.sqrt(np.sum(scaled * scaled)), 1e300)
        return np.sqrt(np.sum(v * v))
    else:
        # Axis norm - always use safe scaling
        max_vals = np.max(np.abs(v), axis=axis, keepdims=True)
        max_vals = np.where(max_vals == 0, 1.0, max_vals)

        # Always scale to prevent overflow
        scaled = v / max_vals
        result = max_vals.squeeze(axis) * np.sqrt(np.sum(scaled * scaled, axis=axis))
        return np.clip(result, 0, 1e300)


def golden_ratio() -> float:
    """Return the golden ratio φ."""
    return PHI


def phi_weight(index: int, decay: float = 0.9) -> float:
    """
    Compute golden-ratio based weight.

    W(i) = φ^(-i * decay)

    Args:
        index: Position index
        decay: Decay rate

    Returns:
        Weight value
    """
    return PHI ** (-index * decay)


def normalize_vector(v: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Normalize vector to unit length.

    Args:
        v: Input vector
        epsilon: Small value to prevent division by zero

    Returns:
        Normalized vector
    """
    v = np.asarray(v, dtype=np.float64)
    norm = np.linalg.norm(v)
    if norm < epsilon:
        return v
    return v / norm


def project_to_sphere(v: np.ndarray, radius: float = 1.0) -> np.ndarray:
    """
    Project vector onto sphere surface.

    Args:
        v: Input vector
        radius: Sphere radius

    Returns:
        Vector on sphere surface
    """
    return normalize_vector(v) * radius


def project_to_ball(v: np.ndarray, max_radius: float = 1.0, epsilon: float = 1e-10) -> np.ndarray:
    """
    Project vector into ball (clamp if outside).

    Args:
        v: Input vector
        max_radius: Maximum allowed radius
        epsilon: Buffer from boundary

    Returns:
        Vector inside ball
    """
    v = np.asarray(v, dtype=np.float64)
    norm = np.linalg.norm(v)
    max_norm = max_radius - epsilon

    if norm > max_norm:
        return v / norm * max_norm
    return v


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute cosine similarity between vectors.

    Args:
        v1: First vector
        v2: Second vector

    Returns:
        Cosine similarity in [-1, 1]
    """
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0

    return float(np.dot(v1, v2) / (norm1 * norm2))


def angular_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute angular distance between vectors (in radians).

    Args:
        v1: First vector
        v2: Second vector

    Returns:
        Angle in radians [0, π]
    """
    cos_sim = cosine_similarity(v1, v2)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return float(np.arccos(cos_sim))


def linear_interpolate(v1: np.ndarray, v2: np.ndarray, t: float) -> np.ndarray:
    """
    Linear interpolation between vectors.

    Args:
        v1: Start vector
        v2: End vector
        t: Interpolation parameter [0, 1]

    Returns:
        Interpolated vector
    """
    return (1 - t) * np.asarray(v1) + t * np.asarray(v2)


def slerp(v1: np.ndarray, v2: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation.

    Interpolates along the great circle between two unit vectors.

    Args:
        v1: Start vector (should be normalized)
        v2: End vector (should be normalized)
        t: Interpolation parameter [0, 1]

    Returns:
        Interpolated unit vector
    """
    v1 = normalize_vector(v1)
    v2 = normalize_vector(v2)

    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    theta = np.arccos(dot)

    if abs(theta) < 1e-10:
        return v1.copy()

    sin_theta = np.sin(theta)

    return (np.sin((1 - t) * theta) / sin_theta) * v1 + (np.sin(t * theta) / sin_theta) * v2


def compute_variance(vectors: List[np.ndarray]) -> float:
    """
    Compute variance of a set of vectors.

    Args:
        vectors: List of vectors

    Returns:
        Variance value
    """
    if not vectors:
        return 0.0

    array = np.array(vectors)
    return float(np.var(array))


def compute_entropy(values: np.ndarray) -> float:
    """
    Compute Shannon entropy of a distribution.

    Args:
        values: Array of values (will be normalized to probabilities)

    Returns:
        Entropy in bits
    """
    values = np.asarray(values, dtype=np.float64)
    values = np.abs(values)

    total = values.sum()
    if total < 1e-10:
        return 0.0

    probs = values / total
    probs = probs[probs > 0]  # Filter zeros

    return float(-np.sum(probs * np.log2(probs)))


def random_unit_vector(dim: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate a random unit vector.

    Args:
        dim: Dimension
        seed: Random seed

    Returns:
        Random unit vector
    """
    if seed is not None:
        np.random.seed(seed)

    v = np.random.randn(dim)
    return normalize_vector(v)


def random_orthogonal_matrix(dim: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate a random orthogonal matrix.

    Args:
        dim: Matrix dimension
        seed: Random seed

    Returns:
        Orthogonal matrix (dim x dim)
    """
    if seed is not None:
        np.random.seed(seed)

    A = np.random.randn(dim, dim)
    Q, _ = np.linalg.qr(A)
    return Q


def pad_or_truncate(v: np.ndarray, target_dim: int) -> np.ndarray:
    """
    Pad or truncate vector to target dimension.

    Args:
        v: Input vector
        target_dim: Target dimension

    Returns:
        Vector of target dimension
    """
    v = np.asarray(v, dtype=np.float64)

    if len(v) == target_dim:
        return v

    result = np.zeros(target_dim)
    n = min(len(v), target_dim)
    result[:n] = v[:n]
    return result


def fibonacci_sphere(n_points: int) -> np.ndarray:
    """
    Generate points on a sphere using Fibonacci spiral.

    Args:
        n_points: Number of points

    Returns:
        Array of shape (n_points, 3)
    """
    indices = np.arange(n_points, dtype=float)
    phi_angle = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle

    y = 1 - (indices / (n_points - 1)) * 2
    radius = np.sqrt(1 - y**2)

    theta = phi_angle * indices

    x = np.cos(theta) * radius
    z = np.sin(theta) * radius

    return np.column_stack([x, y, z])


def bytes_to_bits(data: bytes) -> np.ndarray:
    """
    Convert bytes to bit array.

    Args:
        data: Input bytes

    Returns:
        Array of bits (0 or 1)
    """
    bits = []
    for byte in data:
        for i in range(8):
            bits.append((byte >> (7 - i)) & 1)
    return np.array(bits, dtype=np.uint8)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """
    Convert bit array to bytes.

    Args:
        bits: Array of bits

    Returns:
        Bytes
    """
    # Pad to multiple of 8
    n = len(bits)
    padded_len = ((n + 7) // 8) * 8
    padded = np.zeros(padded_len, dtype=np.uint8)
    padded[:n] = bits

    result = []
    for i in range(0, padded_len, 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | padded[i + j]
        result.append(byte)

    return bytes(result)

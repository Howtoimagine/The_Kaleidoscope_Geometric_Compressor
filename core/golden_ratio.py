"""
Golden Ratio Weighting for E8ZIP

Implements quasicrystal-inspired φ-weighting for improved compression.

Based on Hypothesis H2 from Kaleidoscope experiments:
"E8 root vectors optimally pack according to quasicrystal (Penrose/phi)
tiling rules in 8D projection."

Theory:
- Golden ratio φ = (1 + √5)/2 ≈ 1.618
- Fibonacci sequence appears naturally in optimal packings
- Quasicrystals use aperiodic tilings with φ-based spacing
- φ-weighted distances reduce total packing energy

Weight function: w(d) = φ^(-d)
- Nearby vectors get high weight
- Distance decay follows golden ratio
- More efficient representation of local structure

References:
- Experiment 3: E8 Quasicrystal Packing (Cycle 2)
- Penrose tiling theory
- Icosahedral quasicrystals
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


# ============================================================
# GOLDEN RATIO CONSTANTS
# ============================================================

PHI = (1 + np.sqrt(5)) / 2  # ≈ 1.618033988749895
PHI_INV = PHI - 1  # = 1/φ ≈ 0.618033988749895
PHI_SQ = PHI**2  # = φ + 1 ≈ 2.618033988749895

# Fibonacci sequence (first 20 terms)
FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765]


# ============================================================
# GOLDEN RATIO WEIGHTING FUNCTIONS
# ============================================================


def phi_weight(distance: float, power: float = 1.0) -> float:
    """
    Golden ratio distance weighting.

    w(d) = φ^(-d)

    Args:
        distance: Distance value
        power: Power modifier (default 1.0)

    Returns:
        Weight value in (0, 1]
    """
    return PHI ** (-distance * power)


def inverse_phi_weight(distance: float) -> float:
    """
    Inverse golden ratio weighting (for emphasis on far points).

    w(d) = 1 - φ^(-d)
    """
    return 1.0 - PHI ** (-distance)


def fibonacci_quantize(value: float, max_level: int = 15) -> Tuple[int, float]:
    """
    Quantize a value to the nearest Fibonacci number.

    Used for discrete golden-ratio-based encoding.

    Args:
        value: Value to quantize
        max_level: Maximum Fibonacci index to consider

    Returns:
        (fibonacci_index, residual)
    """
    abs_val = abs(value)
    sign = np.sign(value) if value != 0 else 1

    # Find nearest Fibonacci number
    best_idx = 0
    best_dist = float("inf")

    for i, fib in enumerate(FIBONACCI[:max_level]):
        dist = abs(abs_val - fib)
        if dist < best_dist:
            best_dist = dist
            best_idx = i

    residual = value - sign * FIBONACCI[best_idx]
    return (best_idx if sign >= 0 else -best_idx), residual


def golden_angle_encode(value: float, n_bits: int = 8) -> int:
    """
    Encode a value using golden angle distribution.

    The golden angle (≈137.5°) creates optimal distribution patterns.
    Used in phyllotaxis (leaf arrangement in plants).

    Args:
        value: Value in [0, 1] to encode
        n_bits: Number of bits for encoding

    Returns:
        Integer encoding
    """
    golden_angle = 2 * np.pi * PHI_INV  # ≈ 2.399963...
    max_val = 2**n_bits

    # Map value to golden angle sector
    angle = value * 2 * np.pi
    sector = int((angle / golden_angle) % max_val)

    return sector


# ============================================================
# QUASICRYSTAL WEIGHTING CODEC
# ============================================================


@dataclass
class WeightedQuantization:
    """Result of golden-ratio weighted quantization."""

    value: float
    weight: float
    fibonacci_level: int
    residual: float


class GoldenRatioCodec:
    """
    Codec using golden ratio weighting for compression.

    Features:
    - φ-weighted distance metrics
    - Fibonacci-based quantization
    - Quasicrystal-inspired encoding
    """

    def __init__(self, dimension: int = 8):
        """
        Initialize golden ratio codec.

        Args:
            dimension: Vector dimension
        """
        self.dimension = dimension

        # Precompute φ powers for efficiency
        self.phi_powers = np.array([PHI**i for i in range(64)])
        self.phi_neg_powers = np.array([PHI ** (-i) for i in range(64)])

    def weighted_distance(self, v1: np.ndarray, v2: np.ndarray, use_golden: bool = True) -> float:
        """
        Compute golden-ratio weighted distance.

        Args:
            v1, v2: Input vectors
            use_golden: Whether to use φ-weighting

        Returns:
            Weighted distance
        """
        diff = v1 - v2

        if not use_golden:
            return np.linalg.norm(diff)

        # Weight each dimension by Fibonacci-like sequence
        weights = np.array([FIBONACCI[i % len(FIBONACCI)] for i in range(len(diff))])
        weights = weights / weights.sum()  # Normalize

        weighted_diff = diff * np.sqrt(weights)
        return np.linalg.norm(weighted_diff)

    def phi_quantize(self, value: float, precision: int = 10) -> Tuple[List[int], float]:
        """
        Quantize value using φ-based representation.

        Similar to Zeckendorf representation (sum of non-consecutive Fibonacci numbers).

        Args:
            value: Value to quantize
            precision: Number of Fibonacci terms to use

        Returns:
            (list of Fibonacci indices, residual)
        """
        abs_val = abs(value)
        sign = 1 if value >= 0 else -1

        indices = []
        remaining = abs_val

        # Greedy Zeckendorf decomposition
        for i in range(min(precision, len(FIBONACCI) - 1), -1, -1):
            if FIBONACCI[i] <= remaining:
                indices.append(i if sign > 0 else -i)
                remaining -= FIBONACCI[i]

                # Skip consecutive (Zeckendorf property)
                if indices and abs(indices[-1]) == i + 1:
                    continue

        return indices, remaining * sign

    def encode_vector_golden(self, vector: np.ndarray) -> Dict:
        """
        Encode vector using golden ratio techniques.

        Args:
            vector: Input vector

        Returns:
            Encoded representation
        """
        encoded = {
            "dimension": len(vector),
            "scale": float(np.linalg.norm(vector)),
            "fibonacci_indices": [],
            "golden_angles": [],
            "residuals": [],
        }

        # Normalize
        if encoded["scale"] > 1e-10:
            normalized = vector / encoded["scale"]
        else:
            normalized = vector

        for val in normalized:
            # Fibonacci encoding
            indices, residual = self.phi_quantize(val, precision=8)
            encoded["fibonacci_indices"].append(indices)
            encoded["residuals"].append(residual)

            # Golden angle encoding
            angle_code = golden_angle_encode((val + 1) / 2, n_bits=8)
            encoded["golden_angles"].append(angle_code)

        return encoded

    def decode_vector_golden(self, encoded: Dict) -> np.ndarray:
        """
        Decode vector from golden ratio encoding.

        Args:
            encoded: Encoded representation

        Returns:
            Reconstructed vector
        """
        vector = np.zeros(encoded["dimension"])

        for i, (indices, residual) in enumerate(
            zip(encoded["fibonacci_indices"], encoded["residuals"])
        ):
            # Reconstruct from Fibonacci representation
            val = sum(FIBONACCI[abs(idx)] * (1 if idx >= 0 else -1) for idx in indices)
            vector[i] = val + residual

        # Apply scale
        vector *= encoded["scale"]

        return vector


# ============================================================
# QUASICRYSTAL PACKING ENERGY
# ============================================================


def packing_energy(
    vectors: np.ndarray, use_phi_weight: bool = True, repulsion_power: float = 2.0
) -> Dict:
    """
    Compute packing energy for a set of vectors.

    Based on Experiment 3: E8 Quasicrystal Packing.

    Energy = Σ_i Σ_j≠i w(d_ij) · f(d_ij)
    where w(d) = φ^(-d) (golden ratio decay)
          f(d) = 1/d² (Coulomb-like repulsion)

    Lower energy = better packing.

    Args:
        vectors: Array of vectors (N x D)
        use_phi_weight: Whether to use golden ratio weighting
        repulsion_power: Power for repulsion term

    Returns:
        Energy statistics
    """
    n = len(vectors)
    total_energy = 0.0
    min_dist = float("inf")
    max_dist = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(vectors[i] - vectors[j])

            if dist < 1e-10:
                continue

            min_dist = min(min_dist, dist)
            max_dist = max(max_dist, dist)

            # Repulsion
            repulsion = 1.0 / (dist**repulsion_power)

            # Weight
            if use_phi_weight:
                weight = phi_weight(dist)
            else:
                weight = 1.0

            total_energy += weight * repulsion

    return {
        "total_energy": total_energy,
        "normalized_energy": total_energy / (n * (n - 1) / 2) if n > 1 else 0,
        "min_distance": min_dist,
        "max_distance": max_dist,
        "phi_weighted": use_phi_weight,
    }


def optimize_golden_packing(
    vectors: np.ndarray, n_iterations: int = 100, learning_rate: float = 0.01
) -> np.ndarray:
    """
    Optimize vector packing using golden ratio gradient descent.

    Args:
        vectors: Initial vectors (N x D)
        n_iterations: Number of optimization steps
        learning_rate: Step size

    Returns:
        Optimized vectors
    """
    vectors = vectors.copy()
    n, d = vectors.shape

    for iteration in range(n_iterations):
        gradients = np.zeros_like(vectors)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                diff = vectors[i] - vectors[j]
                dist = np.linalg.norm(diff)

                if dist < 1e-10:
                    continue

                # Gradient of φ-weighted energy
                weight = phi_weight(dist)
                grad_weight = -np.log(PHI) * weight

                repulsion_grad = -2 / (dist**3) * diff

                gradient = weight * repulsion_grad + grad_weight * (1 / dist**2) * (diff / dist)
                gradients[i] += gradient

        # Update with golden ratio step
        vectors -= learning_rate * PHI_INV * gradients

    return vectors


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("GOLDEN RATIO WEIGHTING DEMO")
    print("Quasicrystal-Inspired Compression")
    print("=" * 60)

    print(f"\n1. Golden Ratio Constants")
    print(f"   φ = {PHI:.10f}")
    print(f"   1/φ = {PHI_INV:.10f}")
    print(f"   φ² = {PHI_SQ:.10f}")
    print(f"   φ² - φ - 1 = {PHI_SQ - PHI - 1:.10f} (should be 0)")

    print(f"\n   Fibonacci sequence: {FIBONACCI[:10]}")
    print(f"   Ratio F(n)/F(n-1) → φ: {[FIBONACCI[i + 1] / FIBONACCI[i] for i in range(5, 10)]}")

    print("\n2. φ-Weighting Test")
    for d in [0.5, 1.0, 2.0, 5.0, 10.0]:
        w = phi_weight(d)
        print(f"   Distance {d:5.1f} → Weight {w:.6f}")

    print("\n3. Fibonacci Quantization")
    codec = GoldenRatioCodec()

    for val in [1.5, 5.0, 13.3, 21.7, 100.0]:
        indices, residual = codec.phi_quantize(val)
        reconstructed = sum(FIBONACCI[abs(i)] * (1 if i >= 0 else -1) for i in indices)
        print(f"   {val:6.1f} → Fib{indices} = {reconstructed:.0f} + {residual:.3f}")

    print("\n4. Vector Encoding")
    np.random.seed(42)
    test_vec = np.random.randn(8) * 10

    encoded = codec.encode_vector_golden(test_vec)
    decoded = codec.decode_vector_golden(encoded)

    print(f"   Original:    {test_vec[:4]}...")
    print(f"   Decoded:     {decoded[:4]}...")
    print(f"   Error:       {np.max(np.abs(test_vec - decoded)):.6f}")

    print("\n5. Packing Energy Comparison")
    np.random.seed(42)
    random_vectors = np.random.randn(50, 8)

    uniform_energy = packing_energy(random_vectors, use_phi_weight=False)
    golden_energy = packing_energy(random_vectors, use_phi_weight=True)

    print(f"   Uniform weighting: {uniform_energy['normalized_energy']:.4f}")
    print(f"   Golden weighting:  {golden_energy['normalized_energy']:.4f}")

    print("\n6. Golden Packing Optimization")
    optimized = optimize_golden_packing(random_vectors, n_iterations=50)
    opt_energy = packing_energy(optimized, use_phi_weight=True)

    print(f"   Before: {golden_energy['normalized_energy']:.4f}")
    print(f"   After:  {opt_energy['normalized_energy']:.4f}")
    print(
        f"   Improvement: {(1 - opt_energy['normalized_energy'] / golden_energy['normalized_energy']) * 100:.1f}%"
    )

    print("\n" + "=" * 60)
    print("GOLDEN RATIO COMPLETE")
    print("=" * 60)

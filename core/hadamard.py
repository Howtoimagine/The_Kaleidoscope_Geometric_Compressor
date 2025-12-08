"""
Hadamard Transform for E8ZIP

Implementation of the Fast Walsh-Hadamard Transform for preprocessing data
before E8 lattice quantization.

Based on QuIP# (ICML 2024) which uses randomized Hadamard Transform
for incoherence processing to achieve optimal compression.

Theory:
- Hadamard transform spreads information uniformly across all dimensions
- After transform, data follows sub-Gaussian distribution
- This makes E8 lattice quantization more efficient
- Complexity: O(n log n) vs O(n²) for dense matrix multiplication

The Walsh-Hadamard matrix H_n is defined recursively:
    H_1 = [1]
    H_2n = [H_n   H_n ]
           [H_n  -H_n ]

Properties:
- Orthogonal: H^T H = n I
- Self-inverse: H^-1 = H/n
- All entries ±1 (no multiplication needed!)
"""

import numpy as np
from typing import Tuple, Optional
from functools import lru_cache


class HadamardTransform:
    """
    Fast Walsh-Hadamard Transform for preprocessing.

    Features:
    - O(n log n) complexity
    - Randomized version for better incoherence (QuIP# style)
    - Support for arbitrary dimension (pads to power of 2)
    """

    def __init__(self, dimension: int = 8, randomized: bool = True, seed: int = 42):
        """
        Initialize Hadamard Transform.

        Args:
            dimension: Input dimension (will be padded to power of 2)
            randomized: Use randomized signs for better incoherence
            seed: Random seed for reproducibility
        """
        self.original_dim = dimension
        self.padded_dim = self._next_power_of_2(dimension)
        self.randomized = randomized
        self.seed = seed

        # Generate random signs for randomized Hadamard
        if randomized:
            np.random.seed(seed)
            self.random_signs = np.random.choice([-1, 1], size=self.padded_dim)
        else:
            self.random_signs = np.ones(self.padded_dim)

    def _next_power_of_2(self, n: int) -> int:
        """Find smallest power of 2 >= n."""
        return 1 << (n - 1).bit_length()

    def transform(self, data: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Apply Fast Walsh-Hadamard Transform.

        Args:
            data: Input data (1D or 2D array)
            normalize: Whether to normalize by 1/sqrt(n)

        Returns:
            Transformed data
        """
        if data.ndim == 1:
            return self._transform_1d(data, normalize)
        elif data.ndim == 2:
            # Transform each row
            return np.array([self._transform_1d(row, normalize) for row in data])
        else:
            raise ValueError(f"Unsupported array dimension: {data.ndim}")

    def _transform_1d(self, data: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Apply FWHT to 1D array."""
        n = len(data)

        # Pad to power of 2
        if n < self.padded_dim:
            padded = np.zeros(self.padded_dim)
            padded[:n] = data
        else:
            padded = data[: self.padded_dim].copy()

        # Apply random signs (for randomized Hadamard)
        if self.randomized:
            padded = padded * self.random_signs

        # Fast Walsh-Hadamard Transform (iterative in-place)
        # Pre-clip and normalize to prevent overflow during transform
        max_val = np.max(np.abs(padded))
        scale_factor = 1.0
        if max_val > 1e10:
            scale_factor = max_val / 1e10
            padded = padded / scale_factor

        h = 1
        while h < self.padded_dim:
            for i in range(0, self.padded_dim, h * 2):
                for j in range(i, i + h):
                    x = padded[j]
                    y = padded[j + h]
                    # Safe addition with clipping
                    sum_val = x + y
                    diff_val = x - y
                    # Clip extreme values
                    if abs(sum_val) > 1e100:
                        sum_val = np.sign(sum_val) * 1e100
                    if abs(diff_val) > 1e100:
                        diff_val = np.sign(diff_val) * 1e100
                    padded[j] = sum_val
                    padded[j + h] = diff_val
            h *= 2

        # Restore scale if we applied one (with overflow protection)
        if scale_factor != 1.0:
            with np.errstate(over="ignore"):
                padded = padded * scale_factor
            padded = np.nan_to_num(padded, nan=0.0, posinf=1e150, neginf=-1e150)

        # Normalize
        if normalize:
            padded = padded / np.sqrt(self.padded_dim)

        # Final cleanup and return original dimension
        return np.clip(padded[: self.original_dim], -1e150, 1e150)

    def inverse_transform(self, data: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Apply inverse Fast Walsh-Hadamard Transform.

        The Hadamard transform is self-inverse (up to scaling).

        Args:
            data: Transformed data
            normalize: Whether to normalize

        Returns:
            Original data
        """
        if data.ndim == 1:
            return self._inverse_1d(data, normalize)
        elif data.ndim == 2:
            return np.array([self._inverse_1d(row, normalize) for row in data])
        else:
            raise ValueError(f"Unsupported array dimension: {data.ndim}")

    def _inverse_1d(self, data: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Apply inverse FWHT to 1D array."""
        n = len(data)

        # Pad to power of 2
        if n < self.padded_dim:
            padded = np.zeros(self.padded_dim)
            padded[:n] = data
        else:
            padded = data[: self.padded_dim].copy()

        # FWHT is self-inverse (with normalization)
        h = 1
        while h < self.padded_dim:
            for i in range(0, self.padded_dim, h * 2):
                for j in range(i, i + h):
                    x = padded[j]
                    y = padded[j + h]
                    padded[j] = x + y
                    padded[j + h] = x - y
            h *= 2

        # Normalize
        if normalize:
            padded = padded / np.sqrt(self.padded_dim)

        # Undo random signs
        if self.randomized:
            padded = padded * self.random_signs

        return padded[: self.original_dim]


class IncoherenceProcessor:
    """
    Full incoherence processing pipeline based on QuIP#.

    Steps:
    1. Randomized Hadamard Transform (spreads information)
    2. Optional: Apply learned rotation matrix
    3. Optional: Scale by importance weights
    """

    def __init__(self, dimension: int = 8, seed: int = 42):
        """
        Initialize incoherence processor.

        Args:
            dimension: Data dimension
            seed: Random seed
        """
        self.dimension = dimension
        self.hadamard = HadamardTransform(dimension, randomized=True, seed=seed)

        # Optional learned rotation
        self.rotation_matrix = None

    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Apply incoherence processing.

        Args:
            data: Input vectors

        Returns:
            Incoherent vectors (better for lattice quantization)
        """
        # Step 1: Hadamard transform
        transformed = self.hadamard.transform(data)

        # Step 2: Optional rotation
        if self.rotation_matrix is not None:
            if data.ndim == 1:
                transformed = self.rotation_matrix @ transformed
            else:
                transformed = transformed @ self.rotation_matrix.T

        return transformed

    def inverse_process(self, data: np.ndarray) -> np.ndarray:
        """
        Undo incoherence processing.

        Args:
            data: Incoherent vectors

        Returns:
            Original-space vectors
        """
        result = data

        # Undo rotation
        if self.rotation_matrix is not None:
            if data.ndim == 1:
                result = self.rotation_matrix.T @ result
            else:
                result = result @ self.rotation_matrix

        # Inverse Hadamard
        result = self.hadamard.inverse_transform(result)

        return result

    def learn_rotation(self, samples: np.ndarray, n_iter: int = 100) -> np.ndarray:
        """
        Learn optimal rotation matrix from data samples.

        Uses PCA-like alignment to find principal directions.

        Args:
            samples: Training data samples
            n_iter: Number of iterations

        Returns:
            Learned rotation matrix
        """
        # Apply Hadamard first
        transformed = self.hadamard.transform(samples)

        # Compute covariance
        cov = np.cov(transformed.T)

        # Eigendecomposition for PCA-like rotation
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        self.rotation_matrix = eigenvectors[:, idx].T

        return self.rotation_matrix


def compute_incoherence_metric(data: np.ndarray) -> float:
    """
    Compute incoherence metric (lower is more coherent/concentrated).

    Based on the L-infinity to L2 ratio.

    Args:
        data: Input vectors

    Returns:
        Incoherence score (higher means more spread out)
    """
    if data.ndim == 1:
        l_inf = np.max(np.abs(data))
        l2 = np.linalg.norm(data)
        return l2 / (l_inf + 1e-10) / np.sqrt(len(data))
    else:
        scores = []
        for row in data:
            l_inf = np.max(np.abs(row))
            l2 = np.linalg.norm(row)
            scores.append(l2 / (l_inf + 1e-10) / np.sqrt(len(row)))
        return np.mean(scores)


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("HADAMARD TRANSFORM DEMO")
    print("Based on QuIP# (ICML 2024) - E8 Lattice Quantization")
    print("=" * 60)

    # Test basic transform
    print("\n1. Basic Hadamard Transform")
    ht = HadamardTransform(dimension=8, randomized=False)

    # Concentrated vector (bad for quantization)
    concentrated = np.array([10.0, 0, 0, 0, 0, 0, 0, 0])
    print(f"   Original:    {concentrated}")

    transformed = ht.transform(concentrated)
    print(f"   Transformed: {transformed}")

    recovered = ht.inverse_transform(transformed)
    print(f"   Recovered:   {recovered}")
    print(f"   Match: {np.allclose(concentrated, recovered)}")

    # Test incoherence
    print("\n2. Incoherence Processing (Randomized)")
    processor = IncoherenceProcessor(dimension=8)

    # Generate test data
    np.random.seed(42)
    test_data = np.random.randn(100, 8)

    # Make some dimensions concentrated (bad)
    test_data[:, 0] *= 10

    print(f"   Original incoherence: {compute_incoherence_metric(test_data):.4f}")

    processed = processor.process(test_data)
    print(f"   After processing:     {compute_incoherence_metric(processed):.4f}")

    recovered = processor.inverse_process(processed)
    print(f"   Recovery error:       {np.max(np.abs(test_data - recovered)):.6f}")

    print("\n3. Effect on Quantization")
    from e8_lattice import E8Lattice

    e8 = E8Lattice()

    # Quantize original vs processed
    original_errors = []
    processed_errors = []

    for i in range(min(10, len(test_data))):
        # Original
        _, _, err_orig = e8.nearest_lattice_point(test_data[i])
        original_errors.append(err_orig)

        # Processed (then quantize in transformed space)
        _, _, err_proc = e8.nearest_lattice_point(processed[i])
        processed_errors.append(err_proc)

    print(f"   Avg quantization error (original):  {np.mean(original_errors):.4f}")
    print(f"   Avg quantization error (processed): {np.mean(processed_errors):.4f}")
    print(
        f"   Improvement: {(1 - np.mean(processed_errors) / np.mean(original_errors)) * 100:.1f}%"
    )

    print("\n" + "=" * 60)
    print("HADAMARD TRANSFORM COMPLETE")
    print("=" * 60)

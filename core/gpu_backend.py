"""
E8ZIP GPU Backend (PyTorch Edition)

NVIDIA CUDA acceleration via PyTorch for E8 lattice operations.
Falls back to NumPy when GPU is not available.

Key accelerated operations:
- Distance calculations (nearest_root, nearest_lattice_point)
- Hadamard transforms
- Black hole matrix operations
- Batch vector processing
"""

import os
import warnings
from typing import Optional, Tuple, Union
import logging
import numpy as np

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
#                              GPU DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

GPU_AVAILABLE = False
GPU_DEVICE_NAME = "None"
GPU_MEMORY_GB = 0
BACKEND_NAME = "NumPy (CPU)"

try:
    import torch

    if torch.cuda.is_available():
        GPU_AVAILABLE = True
        GPU_DEVICE_NAME = torch.cuda.get_device_name(0)
        # Convert bytes to GB
        total_mem = torch.cuda.get_device_properties(0).total_memory
        GPU_MEMORY_GB = total_mem / (1024**3)
        BACKEND_NAME = "PyTorch (CUDA)"
        logger.info(f"GPU detected: {GPU_DEVICE_NAME} ({GPU_MEMORY_GB:.1f} GB)")

        # Warmup check
        try:
            # Simple tensor operation to verify CUDA is actually working
            _x = torch.tensor([1.0], device="cuda")
            _y = _x + 1.0
            _z = _y.cpu()
        except Exception as e:
            logger.warning(f"GPU detected but failed warmup: {e}. Falling back to CPU.")
            GPU_AVAILABLE = False
            BACKEND_NAME = "NumPy (CPU)"
    else:
        logger.info("PyTorch available but no CUDA devices found. Using NumPy.")

except ImportError:
    logger.info("PyTorch not installed. Using NumPy. Install with: pip install torch")
except Exception as e:
    logger.warning(f"GPU initialization failed: {e}. Using NumPy.")


# ═══════════════════════════════════════════════════════════════════════════════
#                              ARRAY PROVIDER
# ═══════════════════════════════════════════════════════════════════════════════


class ArrayProvider:
    """
    Unified interface for GPU (PyTorch) and CPU (NumPy) array operations.
    Automatically uses GPU when available and beneficial.
    """

    def __init__(self, force_cpu: bool = False):
        """
        Initialize array provider.

        Args:
            force_cpu: If True, always use NumPy even if GPU available
        """
        self.use_gpu = GPU_AVAILABLE and not force_cpu
        self.device = torch.device("cuda") if self.use_gpu else torch.device("cpu")

        # Minimum array size to use GPU (smaller arrays have too much overhead)
        self.gpu_threshold = 5000  # Elements

    @property
    def backend_name(self) -> str:
        return BACKEND_NAME if self.use_gpu else "NumPy (CPU)"

    def to_gpu(self, arr: np.ndarray) -> Union[np.ndarray, "torch.Tensor"]:
        """Transfer numpy array to GPU if available."""
        if self.use_gpu:
            # Convert to tensor and move to device
            # Ensure float64 for precision if needed, or float32 for speed
            # E8ZIP uses float64 mostly
            if isinstance(arr, np.ndarray):
                return torch.from_numpy(arr).to(self.device)
            elif isinstance(arr, torch.Tensor):
                return arr.to(self.device)
        return arr

    def to_cpu(self, arr) -> np.ndarray:
        """Transfer array to CPU (NumPy)."""
        if self.use_gpu and isinstance(arr, torch.Tensor):
            return arr.cpu().numpy()
        return np.asarray(arr)

    def should_use_gpu(self, size: int) -> bool:
        """Check if array is large enough to benefit from GPU."""
        return self.use_gpu and size >= self.gpu_threshold


# ═══════════════════════════════════════════════════════════════════════════════
#                           GPU-ACCELERATED OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════


class E8GPUOps:
    """
    GPU-accelerated E8 lattice operations using PyTorch.
    """

    def __init__(self, provider: Optional[ArrayProvider] = None):
        self.provider = provider or ArrayProvider()

        # Pre-compute E8 roots on GPU if available
        self._roots = None

    def _get_e8_roots(self):
        """Get E8 root vectors (lazily loaded to GPU)."""
        if self._roots is None:
            # Generate 240 E8 roots
            roots = []

            # Type 1: ±1 in two positions (112 vectors)
            for i in range(8):
                for j in range(i + 1, 8):
                    for si in [-1, 1]:
                        for sj in [-1, 1]:
                            v = [0.0] * 8
                            v[i] = si
                            v[j] = sj
                            roots.append(v)

            # Type 2: ±1/2 in all positions with even number of minuses (128 vectors)
            for mask in range(256):
                if bin(mask).count("1") % 2 == 0:
                    v = [0.5 if not (mask & (1 << i)) else -0.5 for i in range(8)]
                    roots.append(v)

            roots_np = np.array(roots, dtype=np.float64)
            self._roots = self.provider.to_gpu(roots_np)

        return self._roots

    def batch_nearest_root(self, vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find nearest E8 root for a batch of vectors.

        GPU-accelerated when vectors.shape[0] > threshold.

        Args:
            vectors: (N, 8) array of vectors

        Returns:
            (indices, distances) - both of shape (N,)
        """
        N = vectors.shape[0]

        if not self.provider.should_use_gpu(N * 240):
            # CPU path for small batches
            return self._batch_nearest_root_cpu(vectors)

        # GPU path
        roots = self._get_e8_roots()  # (240, 8)
        vectors_gpu = self.provider.to_gpu(vectors)  # (N, 8)

        # Ensure types match (float64 usually)
        if vectors_gpu.dtype != roots.dtype:
            vectors_gpu = vectors_gpu.to(roots.dtype)

        # Compute pairwise distances using torch.cdist
        # vectors: (N, 8), roots: (240, 8) -> distances: (N, 240)
        # cdist computes p-norm (default p=2, Euclidean distance)
        distances = torch.cdist(vectors_gpu, roots)

        # Find minimum for each vector
        min_distances, indices = torch.min(distances, dim=1)

        return self.provider.to_cpu(indices), self.provider.to_cpu(min_distances)

    def _batch_nearest_root_cpu(self, vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """CPU fallback for small batches."""
        roots = self.provider.to_cpu(self._get_e8_roots())

        # Vectorized distance calculation
        diff = vectors[:, None, :] - roots[None, :, :]
        distances = np.sqrt(np.sum(diff * diff, axis=2))

        indices = np.argmin(distances, axis=1)
        min_distances = np.min(distances, axis=1)

        return indices, min_distances

    def batch_hadamard_transform(self, data: np.ndarray) -> np.ndarray:
        """
        GPU-accelerated Fast Walsh-Hadamard Transform.

        Args:
            data: Input data array

        Returns:
            Transformed data
        """
        if not self.provider.should_use_gpu(data.size):
            return self._hadamard_cpu(data)

        data_gpu = self.provider.to_gpu(data.astype(np.float64))

        # Pad to power of 2
        n = len(data_gpu)
        padded_dim = 1
        while padded_dim < n:
            padded_dim *= 2

        # Create padded tensor
        padded = torch.zeros(padded_dim, dtype=data_gpu.dtype, device=self.provider.device)
        padded[:n] = data_gpu

        # Pre-normalize to prevent overflow
        max_val = torch.max(torch.abs(padded))
        scale = 1.0
        if max_val > 1e10:
            scale = float(max_val) / 1e10
            padded = padded / scale

        # Fast Walsh-Hadamard Transform (iterative)
        h = 1
        while h < padded_dim:
            temp = padded.view(-1, 2, h)
            x = temp[:, 0, :]
            y = temp[:, 1, :]

            sum_val = x + y
            diff_val = x - y

            padded = torch.stack((sum_val, diff_val), dim=1).view(-1)
            h *= 2

        # Restore scale and normalize
        if scale != 1.0:
            padded = padded * scale
        padded = padded / torch.sqrt(torch.tensor(padded_dim, device=self.provider.device))

        result = self.provider.to_cpu(padded[:n])
        return np.clip(result, -1e150, 1e150)

    def _hadamard_cpu(self, data: np.ndarray) -> np.ndarray:
        """CPU fallback for Hadamard transform."""
        # Use existing implementation
        n = len(data)
        padded_dim = 1
        while padded_dim < n:
            padded_dim *= 2

        padded = np.zeros(padded_dim, dtype=np.float64)
        padded[:n] = data

        h = 1
        while h < padded_dim:
            for i in range(0, padded_dim, h * 2):
                for j in range(i, i + h):
                    x, y = padded[j], padded[j + h]
                    padded[j], padded[j + h] = x + y, x - y
            h *= 2

        padded = padded / np.sqrt(padded_dim)
        return padded[:n]


# ═══════════════════════════════════════════════════════════════════════════════
#                              GLOBAL INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

# Default provider and ops (lazy initialization)
_default_provider: Optional[ArrayProvider] = None
_default_ops: Optional[E8GPUOps] = None


def get_provider(force_cpu: bool = False) -> ArrayProvider:
    """Get the default array provider."""
    global _default_provider
    if _default_provider is None or force_cpu:
        _default_provider = ArrayProvider(force_cpu=force_cpu)
    return _default_provider


def get_gpu_ops(force_cpu: bool = False) -> E8GPUOps:
    """Get the default GPU operations instance."""
    global _default_ops
    if _default_ops is None or force_cpu:
        _default_ops = E8GPUOps(get_provider(force_cpu))
    return _default_ops


def gpu_info() -> dict:
    """Get GPU information."""
    return {
        "available": GPU_AVAILABLE,
        "device_name": GPU_DEVICE_NAME,
        "memory_gb": GPU_MEMORY_GB,
        "backend": BACKEND_NAME,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#                              UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def sync_gpu():
    """Synchronize GPU operations (wait for completion)."""
    if GPU_AVAILABLE:
        torch.cuda.synchronize()


def free_gpu_memory():
    """Free unused GPU memory."""
    if GPU_AVAILABLE:
        torch.cuda.empty_cache()

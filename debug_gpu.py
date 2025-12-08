import sys
import os
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

print("Python executable:", sys.executable)
print("Python version:", sys.version)

print("\n--- Checking PyTorch ---")
try:
    import torch

    print("PyTorch imported successfully.")
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("CUDA Device Count:", torch.cuda.device_count())
        print("Device 0:", torch.cuda.get_device_name(0))

        # Try a simple operation
        print("Testing simple GPU operation...")
        a_gpu = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        b_gpu = torch.tensor([4.0, 5.0, 6.0], device="cuda")
        c_gpu = a_gpu + b_gpu
        print("Result:", c_gpu)
        print("Simple operation successful.")
    else:
        print("No CUDA devices found.")

except ImportError as e:
    print(f"PyTorch import failed: {e}")
except Exception as e:
    print(f"PyTorch check failed: {e}")

print("\n--- Checking E8ZIP GPU Backend ---")
try:
    # Add parent directory to path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from e8zip.core.gpu_backend import GPU_AVAILABLE, GPU_DEVICE_NAME, get_gpu_ops, BACKEND_NAME

    print(f"GPU_AVAILABLE: {GPU_AVAILABLE}")
    print(f"GPU_DEVICE_NAME: {GPU_DEVICE_NAME}")
    print(f"BACKEND_NAME: {BACKEND_NAME}")

    if GPU_AVAILABLE:
        print("Testing E8GPUOps...")
        ops = get_gpu_ops()

        # Create dummy vectors
        N = 1000
        vectors = np.random.randn(N, 8).astype(np.float64)

        print(f"Running batch_nearest_root on {N} vectors...")
        indices, distances = ops.batch_nearest_root(vectors)

        print(f"Output shapes: indices={indices.shape}, distances={distances.shape}")
        print("E8GPUOps test successful.")
    else:
        print("GPU backend reports GPU not available.")

except Exception as e:
    print(f"E8ZIP GPU Backend check failed: {e}")
    import traceback

    traceback.print_exc()

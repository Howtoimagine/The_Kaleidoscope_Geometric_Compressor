import sys
import os
import time
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from e8zip.core.compressor import E8Compressor, CompressionMode
from e8zip.core.gpu_backend import GPU_AVAILABLE, BACKEND_NAME

print(f"GPU Available: {GPU_AVAILABLE}")
print(f"Backend: {BACKEND_NAME}")

if not GPU_AVAILABLE:
    print("WARNING: GPU not available. Test will run on CPU.")

# Create compressor
compressor = E8Compressor(mode=CompressionMode.MYTHIC, verbose=True)

# Create dummy data (10MB)
data_size = 10 * 1024 * 1024
data = os.urandom(data_size)

print(f"Compressing {data_size} bytes...")
start_time = time.time()
compressed = compressor.compress(data)
end_time = time.time()

print(f"Compression time: {end_time - start_time:.4f} seconds")
print(f"Compressed size: {len(compressed)} bytes")
print(f"Ratio: {len(data) / len(compressed):.2f}x")

# Verify decompression
print("Decompressing...")
decompressed = compressor.decompress(compressed)
print(f"Decompressed size: {len(decompressed)} bytes")

# Check if roughly similar (lossy)
# Mythic is lossy, so exact match isn't expected, but length should match
print(f"Length match: {len(data) == len(decompressed)}")

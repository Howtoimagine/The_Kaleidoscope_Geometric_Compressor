import sys
import os
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from e8zip.core.compressor import E8Compressor, CompressionMode
from e8zip.core.streaming import StreamingCompressor
from e8zip.core.gpu_backend import GPU_AVAILABLE, BACKEND_NAME

print(f"GPU Available: {GPU_AVAILABLE}")
print(f"Backend: {BACKEND_NAME}")

# Create a large dummy file (120MB) to trigger streaming threshold if we were using CLI/GUI logic
# But here we test StreamingCompressor directly.
FILE_SIZE = 120 * 1024 * 1024
INPUT_FILE = "test_large_input.bin"
OUTPUT_FILE = "test_large_output.e8z"
RESTORED_FILE = "test_large_restored.bin"

print(f"Generating {FILE_SIZE / 1024 / 1024:.1f} MB test file...")
with open(INPUT_FILE, "wb") as f:
    # Write random data in chunks
    chunk = os.urandom(1024 * 1024)  # 1MB
    for _ in range(120):
        f.write(chunk)

print("Compressing with StreamingCompressor (MYTHIC mode)...")
compressor = E8Compressor(mode=CompressionMode.MYTHIC, verbose=True)
streamer = StreamingCompressor(compressor)


def progress_callback(stats):
    sys.stdout.write(f"\rProgress: {stats.progress:.1f}% | {stats.speed_mb_s:.1f} MB/s")
    sys.stdout.flush()


start_time = time.time()
stats = streamer.compress_file(INPUT_FILE, OUTPUT_FILE, progress_callback=progress_callback)
end_time = time.time()
print()

print(f"Compression complete in {end_time - start_time:.2f}s")
print(f"Average Speed: {stats.speed_mb_s:.2f} MB/s")
print(f"Ratio: {stats.ratio:.2f}x")

print("Decompressing...")
start_time = time.time()
stats_decomp = streamer.decompress_file(
    OUTPUT_FILE, RESTORED_FILE, progress_callback=progress_callback
)
end_time = time.time()
print()

print(f"Decompression complete in {end_time - start_time:.2f}s")

# Verify size
orig_size = os.path.getsize(INPUT_FILE)
restored_size = os.path.getsize(RESTORED_FILE)
print(f"Original size: {orig_size}")
print(f"Restored size: {restored_size}")
print(f"Match: {orig_size == restored_size}")

# Cleanup
# os.remove(INPUT_FILE)
# os.remove(OUTPUT_FILE)
# os.remove(RESTORED_FILE)

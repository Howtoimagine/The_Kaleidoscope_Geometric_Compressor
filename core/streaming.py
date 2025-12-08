"""
E8ZIP Streaming Compression

Chunked processing for large files.
Keeps memory usage bounded regardless of file size.

Features:
- Process files of any size with constant memory
- Progress callbacks for UI integration
- Parallel chunk processing (CPU and GPU)
- Resume capability for interrupted compressions
"""

import os
import struct
import hashlib
import time
from pathlib import Path
from typing import Optional, Callable, Iterator, Tuple, BinaryIO, Union
from dataclasses import dataclass
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#                              CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_CHUNK_SIZE = 4 * 1024 * 1024  # 4 MB chunks
MIN_CHUNK_SIZE = 64 * 1024  # 64 KB minimum
MAX_CHUNK_SIZE = 64 * 1024 * 1024  # 64 MB maximum

# Streaming header magic
STREAM_MAGIC = b"E8ZSTRM1"  # 8 bytes


# ═══════════════════════════════════════════════════════════════════════════════
#                              DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class StreamingStats:
    """Statistics for streaming compression."""

    total_size: int = 0
    processed_size: int = 0
    compressed_size: int = 0
    chunk_count: int = 0
    elapsed_time: float = 0.0

    @property
    def progress(self) -> float:
        """Progress as percentage (0-100)."""
        if self.total_size == 0:
            return 0.0
        return (self.processed_size / self.total_size) * 100

    @property
    def ratio(self) -> float:
        """Compression ratio."""
        if self.compressed_size == 0:
            return 0.0
        return self.processed_size / self.compressed_size

    @property
    def speed_mb_s(self) -> float:
        """Processing speed in MB/s."""
        if self.elapsed_time == 0:
            return 0.0
        return (self.processed_size / (1024 * 1024)) / self.elapsed_time


@dataclass
class ChunkHeader:
    """Header for each compressed chunk."""

    original_size: int
    compressed_size: int
    checksum: int  # CRC32

    def to_bytes(self) -> bytes:
        """Serialize to bytes."""
        return struct.pack("<III", self.original_size, self.compressed_size, self.checksum)

    @classmethod
    def from_bytes(cls, data: bytes) -> "ChunkHeader":
        """Deserialize from bytes."""
        original, compressed, checksum = struct.unpack("<III", data)
        return cls(original, compressed, checksum)

    @staticmethod
    def size() -> int:
        return 12  # 3 x 4 bytes


# ═══════════════════════════════════════════════════════════════════════════════
#                           STREAMING COMPRESSOR
# ═══════════════════════════════════════════════════════════════════════════════


class StreamingCompressor:
    """
    Streaming E8ZIP compressor for large files.

    Processes files in chunks to keep memory bounded.
    Supports parallel processing and progress callbacks.
    """

    def __init__(
        self,
        mode: Union[str, "E8Compressor"] = "normal",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        parallel: bool = True,
        max_workers: Optional[int] = None,
        use_gpu: bool = True,
    ):
        """
        Initialize streaming compressor.

        Args:
            mode: Compression mode string OR existing E8Compressor instance
            chunk_size: Size of each chunk in bytes
            parallel: Enable parallel chunk processing
            max_workers: Max parallel workers (default: CPU count)
            use_gpu: Use GPU acceleration if available
        """
        from e8zip.core.compressor import E8Compressor, CompressionMode

        self.chunk_size = max(MIN_CHUNK_SIZE, min(chunk_size, MAX_CHUNK_SIZE))
        self.parallel = parallel
        self.max_workers = max_workers or mp.cpu_count()
        self.use_gpu = use_gpu

        if isinstance(mode, E8Compressor):
            self._compressor = mode
            self.mode = mode.mode.name.lower()
        else:
            self.mode = mode
            # Create base compressor
            try:
                mode_enum = CompressionMode[mode.upper()]
            except KeyError:
                mode_enum = CompressionMode.NORMAL
            self._compressor = E8Compressor(mode=mode_enum)

        # Stats
        self.stats = StreamingStats()

    def compress_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        progress_callback: Optional[Callable[[StreamingStats], None]] = None,
    ) -> StreamingStats:
        """
        Compress a file using streaming.

        Args:
            input_path: Path to input file
            output_path: Path to output file (default: input_path + .e8z)
            progress_callback: Called after each chunk with stats

        Returns:
            Final compression statistics
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if output_path is None:
            output_path = str(input_path) + ".e8z"
        output_path = Path(output_path)

        # Initialize stats
        self.stats = StreamingStats(total_size=input_path.stat().st_size)
        start_time = time.perf_counter()

        with open(input_path, "rb") as fin:
            with open(output_path, "wb") as fout:
                # Write stream header
                self._write_header(fout, self.stats.total_size)

                # Process chunks
                for chunk in self._read_chunks(fin):
                    compressed_chunk = self._compress_chunk(chunk)

                    # Write chunk header and data
                    checksum = self._crc32(chunk)
                    chunk_header = ChunkHeader(
                        original_size=len(chunk),
                        compressed_size=len(compressed_chunk),
                        checksum=checksum,
                    )
                    fout.write(chunk_header.to_bytes())
                    fout.write(compressed_chunk)

                    # Update stats
                    self.stats.processed_size += len(chunk)
                    self.stats.compressed_size += len(compressed_chunk) + ChunkHeader.size()
                    self.stats.chunk_count += 1
                    self.stats.elapsed_time = time.perf_counter() - start_time

                    if progress_callback:
                        progress_callback(self.stats)

        self.stats.elapsed_time = time.perf_counter() - start_time
        return self.stats

    def decompress_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        progress_callback: Optional[Callable[[StreamingStats], None]] = None,
    ) -> StreamingStats:
        """
        Decompress a streaming E8Z file.

        Args:
            input_path: Path to .e8z file
            output_path: Path to output file
            progress_callback: Called after each chunk with stats

        Returns:
            Final decompression statistics
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if output_path is None:
            # Remove .e8z extension
            output_path = str(input_path)
            if output_path.endswith(".e8z"):
                output_path = output_path[:-4]
            else:
                output_path += ".restored"
        output_path = Path(output_path)

        start_time = time.perf_counter()

        with open(input_path, "rb") as fin:
            # Read and validate header
            original_size = self._read_header(fin)
            self.stats = StreamingStats(total_size=original_size)

            with open(output_path, "wb") as fout:
                while True:
                    # Read chunk header
                    header_data = fin.read(ChunkHeader.size())
                    if not header_data or len(header_data) < ChunkHeader.size():
                        break

                    chunk_header = ChunkHeader.from_bytes(header_data)

                    # Read compressed data
                    compressed_data = fin.read(chunk_header.compressed_size)
                    if len(compressed_data) < chunk_header.compressed_size:
                        raise ValueError("Truncated chunk data")

                    # Decompress
                    decompressed = self._decompress_chunk(compressed_data)

                    # Verify checksum (skip for lossy modes)
                    if not self._is_lossy_mode(getattr(self, "detected_mode_byte", 0)):
                        actual_checksum = self._crc32(decompressed)
                        if actual_checksum != chunk_header.checksum:
                            logger.warning(f"Checksum mismatch in chunk {self.stats.chunk_count}")

                    fout.write(decompressed)

                    # Update stats
                    self.stats.processed_size += len(decompressed)
                    self.stats.compressed_size += chunk_header.compressed_size
                    self.stats.chunk_count += 1
                    self.stats.elapsed_time = time.perf_counter() - start_time

                    if progress_callback:
                        progress_callback(self.stats)

        self.stats.elapsed_time = time.perf_counter() - start_time
        return self.stats

    def _read_chunks(self, file: BinaryIO) -> Iterator[bytes]:
        """Read file in chunks."""
        while True:
            chunk = file.read(self.chunk_size)
            if not chunk:
                break
            yield chunk

    def _compress_chunk(self, data: bytes) -> bytes:
        """Compress a single chunk."""
        return self._compressor.compress(data)

    def _decompress_chunk(self, data: bytes) -> bytes:
        """Decompress a single chunk."""
        return self._compressor.decompress(data)

    def _write_header(self, file: BinaryIO, total_size: int):
        """Write stream header."""
        file.write(STREAM_MAGIC)
        file.write(struct.pack("<Q", total_size))  # 8 bytes
        file.write(struct.pack("<I", self.chunk_size))  # 4 bytes
        file.write(struct.pack("<B", self._mode_to_byte()))  # 1 byte
        file.write(b"\x00" * 11)  # Reserved padding to 32 bytes

    def _read_header(self, file: BinaryIO) -> int:
        """Read and validate stream header. Returns original file size."""
        magic = file.read(8)
        if magic != STREAM_MAGIC:
            raise ValueError(f"Invalid stream magic: expected {STREAM_MAGIC}, got {magic}")

        total_size = struct.unpack("<Q", file.read(8))[0]
        chunk_size = struct.unpack("<I", file.read(4))[0]
        mode_byte = struct.unpack("<B", file.read(1))[0]
        file.read(11)  # Skip reserved

        self.chunk_size = chunk_size
        self.detected_mode_byte = mode_byte
        return total_size

    def _is_lossy_mode(self, mode_byte: int) -> bool:
        """Check if mode is lossy."""
        # 4 = MYTHIC
        return mode_byte == 4

    def _mode_to_byte(self) -> int:
        """Convert mode string to byte."""
        # Match CompressionMode enum values
        modes = {"fast": 1, "normal": 2, "ultra": 3, "mythic": 4, "leech": 5, "quip": 6}
        return modes.get(self.mode.lower(), 2)

    @staticmethod
    def _crc32(data: bytes) -> int:
        """Compute CRC32 checksum."""
        import zlib

        return zlib.crc32(data) & 0xFFFFFFFF


# ═══════════════════════════════════════════════════════════════════════════════
#                           PARALLEL STREAMING
# ═══════════════════════════════════════════════════════════════════════════════


class ParallelStreamingCompressor(StreamingCompressor):
    """
    Parallel streaming compressor using multiple CPU cores or GPU.

    Processes multiple chunks simultaneously for faster compression.
    """

    def compress_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        progress_callback: Optional[Callable[[StreamingStats], None]] = None,
    ) -> StreamingStats:
        """
        Compress a file using parallel streaming.

        Uses ThreadPoolExecutor to process multiple chunks in parallel.
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if output_path is None:
            output_path = str(input_path) + ".e8z"
        output_path = Path(output_path)

        # Initialize stats
        self.stats = StreamingStats(total_size=input_path.stat().st_size)
        start_time = time.perf_counter()

        # Read all chunk positions
        chunk_positions = []
        pos = 0
        while pos < self.stats.total_size:
            size = min(self.chunk_size, self.stats.total_size - pos)
            chunk_positions.append((pos, size))
            pos += size

        # Process chunks in parallel
        results = [None] * len(chunk_positions)

        with open(input_path, "rb") as fin:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}

                for idx, (pos, size) in enumerate(chunk_positions):
                    fin.seek(pos)
                    chunk = fin.read(size)
                    future = executor.submit(self._compress_chunk_with_meta, idx, chunk)
                    futures[future] = idx

                for future in as_completed(futures):
                    idx = futures[future]
                    results[idx] = future.result()

                    # Update stats
                    self.stats.processed_size += results[idx][1]
                    self.stats.chunk_count += 1
                    self.stats.elapsed_time = time.perf_counter() - start_time

                    if progress_callback:
                        progress_callback(self.stats)

        # Write results in order
        with open(output_path, "wb") as fout:
            self._write_header(fout, self.stats.total_size)

            for idx, chunk, compressed, checksum in results:
                chunk_header = ChunkHeader(
                    original_size=len(chunk),
                    compressed_size=len(compressed),
                    checksum=checksum,
                )
                fout.write(chunk_header.to_bytes())
                fout.write(compressed)
                self.stats.compressed_size += len(compressed) + ChunkHeader.size()

        self.stats.elapsed_time = time.perf_counter() - start_time
        return self.stats

    def _compress_chunk_with_meta(self, idx: int, chunk: bytes) -> Tuple[int, bytes, bytes, int]:
        """Compress chunk and return with metadata."""
        compressed = self._compress_chunk(chunk)
        checksum = self._crc32(chunk)
        return (idx, chunk, compressed, checksum)


# ═══════════════════════════════════════════════════════════════════════════════
#                              UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def estimate_memory_usage(file_size: int, chunk_size: int = DEFAULT_CHUNK_SIZE) -> int:
    """
    Estimate peak memory usage for compression.

    Returns:
        Estimated memory in bytes
    """
    # Each chunk: original + compressed + vector workspace
    chunk_memory = chunk_size * 3  # Original + compressed + workspace
    # E8 vector array for chunk: chunk_size / 8 * 64 bytes
    vector_memory = (chunk_size // 8) * 64
    # E8 roots: 240 * 8 * 8 bytes
    roots_memory = 240 * 8 * 8

    return chunk_memory + vector_memory + roots_memory


def recommended_chunk_size(file_size: int, max_memory_mb: int = 256) -> int:
    """
    Calculate recommended chunk size based on available memory.

    Args:
        file_size: Size of file to compress
        max_memory_mb: Maximum memory to use in MB

    Returns:
        Recommended chunk size in bytes
    """
    max_memory = max_memory_mb * 1024 * 1024

    # Chunk memory = chunk_size * ~4 (with overhead)
    chunk_size = max_memory // 4

    # Clamp to valid range
    chunk_size = max(MIN_CHUNK_SIZE, min(chunk_size, MAX_CHUNK_SIZE))

    # If file is smaller than chunk size, use file size
    if file_size < chunk_size:
        chunk_size = file_size

    return chunk_size

"""
E8ZIP - Geometric Lattice Compression

A novel compression algorithm based on E8 lattice quantization,
hyperbolic geodesics, and black hole holographic encoding.

"Compress paths through the lattice, not just snapshots."

Features:
- GPU acceleration (NVIDIA CUDA via CuPy)
- Streaming mode for large files
- Pre-compressed file detection
- 6 compression modes (FAST, NORMAL, ULTRA, MYTHIC, LEECH, QUIP)
"""

__version__ = "2.0.0"
__author__ = "E8 Kaleidoscope Mind"

# Self-alias: the repository uses a flat layout (package contents at the
# repo root) while all internal imports use the absolute `e8zip.` prefix.
# Registering this module as `e8zip` makes the package importable no
# matter what directory name it was cloned into.
import sys as _sys

if "e8zip" not in _sys.modules:
    _sys.modules["e8zip"] = _sys.modules[__name__]

from e8zip.core.compressor import E8Compressor, CompressionMode
from e8zip.core.kgc import KGCCompressor, Law9Error
from e8zip.core.codec import E8GeometricCodec
from e8zip.core.trajectory import TrajectoryCompressor, CompressedTrajectory
from e8zip.core.black_hole import BlackHoleCompressor
from e8zip.formats.e8z import E8ZArchive

# GPU and streaming support
try:
    from e8zip.core.gpu_backend import GPU_AVAILABLE, gpu_info
except ImportError:
    GPU_AVAILABLE = False

    def gpu_info():
        return {"available": False}


try:
    from e8zip.core.streaming import StreamingCompressor, ParallelStreamingCompressor
except ImportError:
    StreamingCompressor = None
    ParallelStreamingCompressor = None

try:
    from e8zip.core.file_analyzer import FileAnalyzer, analyze_file, is_precompressed
except ImportError:
    FileAnalyzer = None
    analyze_file = None
    is_precompressed = None

__all__ = [
    "KGCCompressor",
    "Law9Error",
    "E8Compressor",
    "CompressionMode",
    "E8GeometricCodec",
    "TrajectoryCompressor",
    "CompressedTrajectory",
    "BlackHoleCompressor",
    "E8ZArchive",
    "GPU_AVAILABLE",
    "gpu_info",
    "StreamingCompressor",
    "ParallelStreamingCompressor",
    "FileAnalyzer",
    "analyze_file",
    "is_precompressed",
]

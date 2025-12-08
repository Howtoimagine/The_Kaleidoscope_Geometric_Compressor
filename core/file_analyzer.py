"""
E8ZIP File Analyzer

Detects pre-compressed files and recommends optimal compression settings.
Prevents wasted effort on already-compressed data.
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from enum import Enum
import struct


# ═══════════════════════════════════════════════════════════════════════════════
#                              FILE CATEGORIES
# ═══════════════════════════════════════════════════════════════════════════════


class FileCategory(Enum):
    """Categories of files based on compression potential."""

    # High compression potential
    TEXT = "text"  # Plain text, source code
    STRUCTURED = "structured"  # JSON, XML, CSV
    BINARY = "binary"  # Executables, raw data

    # Low compression potential (already compressed)
    COMPRESSED_IMAGE = "compressed_image"  # JPEG, PNG, WebP
    COMPRESSED_VIDEO = "compressed_video"  # MP4, MKV, WebM
    COMPRESSED_AUDIO = "compressed_audio"  # MP3, AAC, OGG
    COMPRESSED_ARCHIVE = "compressed_archive"  # ZIP, RAR, 7Z, GZ

    # Unknown
    UNKNOWN = "unknown"


@dataclass
class FileAnalysis:
    """Results of file analysis."""

    path: str
    size: int
    category: FileCategory
    mime_type: str
    is_precompressed: bool
    recommended_mode: str
    expected_ratio: Tuple[float, float]  # (min, max)
    warning: Optional[str] = None

    @property
    def size_mb(self) -> float:
        return self.size / (1024 * 1024)

    @property
    def size_human(self) -> str:
        """Human-readable size."""
        if self.size < 1024:
            return f"{self.size} B"
        elif self.size < 1024 * 1024:
            return f"{self.size / 1024:.1f} KB"
        elif self.size < 1024 * 1024 * 1024:
            return f"{self.size / (1024 * 1024):.1f} MB"
        else:
            return f"{self.size / (1024 * 1024 * 1024):.2f} GB"


# ═══════════════════════════════════════════════════════════════════════════════
#                           MAGIC BYTE SIGNATURES
# ═══════════════════════════════════════════════════════════════════════════════

# (magic_bytes, offset, mime_type, category)
MAGIC_SIGNATURES: List[Tuple[bytes, int, str, FileCategory]] = [
    # Images (compressed)
    (b"\xff\xd8\xff", 0, "image/jpeg", FileCategory.COMPRESSED_IMAGE),
    (b"\x89PNG\r\n\x1a\n", 0, "image/png", FileCategory.COMPRESSED_IMAGE),
    (b"GIF87a", 0, "image/gif", FileCategory.COMPRESSED_IMAGE),
    (b"GIF89a", 0, "image/gif", FileCategory.COMPRESSED_IMAGE),
    (b"RIFF", 0, "image/webp", FileCategory.COMPRESSED_IMAGE),  # Also AVI, WAV
    # Video (compressed)
    (b"\x00\x00\x00\x1cftyp", 0, "video/mp4", FileCategory.COMPRESSED_VIDEO),
    (b"\x00\x00\x00\x20ftyp", 0, "video/mp4", FileCategory.COMPRESSED_VIDEO),
    (b"\x00\x00\x00\x18ftyp", 0, "video/mp4", FileCategory.COMPRESSED_VIDEO),
    (b"\x1aE\xdf\xa3", 0, "video/webm", FileCategory.COMPRESSED_VIDEO),  # Also MKV
    (b"FLV\x01", 0, "video/x-flv", FileCategory.COMPRESSED_VIDEO),
    # Audio (compressed)
    (b"ID3", 0, "audio/mpeg", FileCategory.COMPRESSED_AUDIO),
    (b"\xff\xfb", 0, "audio/mpeg", FileCategory.COMPRESSED_AUDIO),  # MP3 frame sync
    (b"\xff\xfa", 0, "audio/mpeg", FileCategory.COMPRESSED_AUDIO),
    (b"OggS", 0, "audio/ogg", FileCategory.COMPRESSED_AUDIO),
    (b"fLaC", 0, "audio/flac", FileCategory.COMPRESSED_AUDIO),
    # Archives (compressed)
    (b"PK\x03\x04", 0, "application/zip", FileCategory.COMPRESSED_ARCHIVE),
    (b"Rar!\x1a\x07", 0, "application/x-rar", FileCategory.COMPRESSED_ARCHIVE),
    (b"7z\xbc\xaf\x27\x1c", 0, "application/x-7z", FileCategory.COMPRESSED_ARCHIVE),
    (b"\x1f\x8b", 0, "application/gzip", FileCategory.COMPRESSED_ARCHIVE),
    (b"BZh", 0, "application/x-bzip2", FileCategory.COMPRESSED_ARCHIVE),
    (b"\xfd7zXZ\x00", 0, "application/x-xz", FileCategory.COMPRESSED_ARCHIVE),
    (b"ZSTD", 0, "application/zstd", FileCategory.COMPRESSED_ARCHIVE),
    # E8ZIP (already ours)
    (b"E8ZIP001", 0, "application/x-e8zip", FileCategory.COMPRESSED_ARCHIVE),
    (b"E8ZSTRM1", 0, "application/x-e8zip-stream", FileCategory.COMPRESSED_ARCHIVE),
    # Text/Structured
    (b"<?xml", 0, "text/xml", FileCategory.STRUCTURED),
    (b"<!DOCTYPE", 0, "text/html", FileCategory.STRUCTURED),
    (b"<html", 0, "text/html", FileCategory.STRUCTURED),
    # Binary
    (b"MZ", 0, "application/x-msdownload", FileCategory.BINARY),  # Windows EXE
    (b"\x7fELF", 0, "application/x-elf", FileCategory.BINARY),  # Linux ELF
]

# Extension-based fallback
EXTENSION_CATEGORIES: Dict[str, Tuple[str, FileCategory]] = {
    # Compressed images
    ".jpg": ("image/jpeg", FileCategory.COMPRESSED_IMAGE),
    ".jpeg": ("image/jpeg", FileCategory.COMPRESSED_IMAGE),
    ".png": ("image/png", FileCategory.COMPRESSED_IMAGE),
    ".gif": ("image/gif", FileCategory.COMPRESSED_IMAGE),
    ".webp": ("image/webp", FileCategory.COMPRESSED_IMAGE),
    ".avif": ("image/avif", FileCategory.COMPRESSED_IMAGE),
    ".heic": ("image/heic", FileCategory.COMPRESSED_IMAGE),
    # Compressed video
    ".mp4": ("video/mp4", FileCategory.COMPRESSED_VIDEO),
    ".mkv": ("video/x-matroska", FileCategory.COMPRESSED_VIDEO),
    ".webm": ("video/webm", FileCategory.COMPRESSED_VIDEO),
    ".avi": ("video/x-msvideo", FileCategory.COMPRESSED_VIDEO),
    ".mov": ("video/quicktime", FileCategory.COMPRESSED_VIDEO),
    ".wmv": ("video/x-ms-wmv", FileCategory.COMPRESSED_VIDEO),
    ".flv": ("video/x-flv", FileCategory.COMPRESSED_VIDEO),
    ".m4v": ("video/x-m4v", FileCategory.COMPRESSED_VIDEO),
    # Compressed audio
    ".mp3": ("audio/mpeg", FileCategory.COMPRESSED_AUDIO),
    ".aac": ("audio/aac", FileCategory.COMPRESSED_AUDIO),
    ".ogg": ("audio/ogg", FileCategory.COMPRESSED_AUDIO),
    ".flac": ("audio/flac", FileCategory.COMPRESSED_AUDIO),
    ".m4a": ("audio/mp4", FileCategory.COMPRESSED_AUDIO),
    ".wma": ("audio/x-ms-wma", FileCategory.COMPRESSED_AUDIO),
    ".opus": ("audio/opus", FileCategory.COMPRESSED_AUDIO),
    # Archives
    ".zip": ("application/zip", FileCategory.COMPRESSED_ARCHIVE),
    ".rar": ("application/x-rar", FileCategory.COMPRESSED_ARCHIVE),
    ".7z": ("application/x-7z", FileCategory.COMPRESSED_ARCHIVE),
    ".gz": ("application/gzip", FileCategory.COMPRESSED_ARCHIVE),
    ".bz2": ("application/x-bzip2", FileCategory.COMPRESSED_ARCHIVE),
    ".xz": ("application/x-xz", FileCategory.COMPRESSED_ARCHIVE),
    ".tar": ("application/x-tar", FileCategory.BINARY),  # tar itself isn't compressed
    ".zst": ("application/zstd", FileCategory.COMPRESSED_ARCHIVE),
    ".e8z": ("application/x-e8zip", FileCategory.COMPRESSED_ARCHIVE),
    # Text
    ".txt": ("text/plain", FileCategory.TEXT),
    ".md": ("text/markdown", FileCategory.TEXT),
    ".rst": ("text/x-rst", FileCategory.TEXT),
    ".log": ("text/plain", FileCategory.TEXT),
    ".csv": ("text/csv", FileCategory.STRUCTURED),
    # Code
    ".py": ("text/x-python", FileCategory.TEXT),
    ".js": ("text/javascript", FileCategory.TEXT),
    ".ts": ("text/typescript", FileCategory.TEXT),
    ".java": ("text/x-java", FileCategory.TEXT),
    ".c": ("text/x-c", FileCategory.TEXT),
    ".cpp": ("text/x-c++", FileCategory.TEXT),
    ".h": ("text/x-c", FileCategory.TEXT),
    ".rs": ("text/x-rust", FileCategory.TEXT),
    ".go": ("text/x-go", FileCategory.TEXT),
    ".rb": ("text/x-ruby", FileCategory.TEXT),
    ".php": ("text/x-php", FileCategory.TEXT),
    ".cs": ("text/x-csharp", FileCategory.TEXT),
    ".swift": ("text/x-swift", FileCategory.TEXT),
    ".kt": ("text/x-kotlin", FileCategory.TEXT),
    # Structured
    ".json": ("application/json", FileCategory.STRUCTURED),
    ".xml": ("text/xml", FileCategory.STRUCTURED),
    ".yaml": ("text/yaml", FileCategory.STRUCTURED),
    ".yml": ("text/yaml", FileCategory.STRUCTURED),
    ".toml": ("text/toml", FileCategory.STRUCTURED),
    ".ini": ("text/plain", FileCategory.STRUCTURED),
    ".html": ("text/html", FileCategory.STRUCTURED),
    ".htm": ("text/html", FileCategory.STRUCTURED),
    ".css": ("text/css", FileCategory.STRUCTURED),
    # Binary
    ".exe": ("application/x-msdownload", FileCategory.BINARY),
    ".dll": ("application/x-msdownload", FileCategory.BINARY),
    ".so": ("application/x-sharedlib", FileCategory.BINARY),
    ".bin": ("application/octet-stream", FileCategory.BINARY),
    ".dat": ("application/octet-stream", FileCategory.BINARY),
}


# ═══════════════════════════════════════════════════════════════════════════════
#                              FILE ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════


class FileAnalyzer:
    """
    Analyzes files to determine optimal compression strategy.
    """

    def analyze(self, path: str) -> FileAnalysis:
        """
        Analyze a file and return compression recommendations.

        Args:
            path: Path to file

        Returns:
            FileAnalysis with recommendations
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        size = path.stat().st_size

        # Try magic bytes first
        mime_type, category = self._detect_by_magic(path)

        # Fallback to extension
        if category == FileCategory.UNKNOWN:
            mime_type, category = self._detect_by_extension(path)

        # Determine if pre-compressed
        is_precompressed = category in (
            FileCategory.COMPRESSED_IMAGE,
            FileCategory.COMPRESSED_VIDEO,
            FileCategory.COMPRESSED_AUDIO,
            FileCategory.COMPRESSED_ARCHIVE,
        )

        # Get recommendations
        recommended_mode, expected_ratio, warning = self._get_recommendations(
            category, is_precompressed, size
        )

        return FileAnalysis(
            path=str(path),
            size=size,
            category=category,
            mime_type=mime_type,
            is_precompressed=is_precompressed,
            recommended_mode=recommended_mode,
            expected_ratio=expected_ratio,
            warning=warning,
        )

    def _detect_by_magic(self, path: Path) -> Tuple[str, FileCategory]:
        """Detect file type by magic bytes."""
        try:
            with open(path, "rb") as f:
                header = f.read(32)

            for magic, offset, mime, category in MAGIC_SIGNATURES:
                if len(header) > offset + len(magic):
                    if header[offset : offset + len(magic)] == magic:
                        return mime, category
        except Exception:
            pass

        return "application/octet-stream", FileCategory.UNKNOWN

    def _detect_by_extension(self, path: Path) -> Tuple[str, FileCategory]:
        """Detect file type by extension."""
        ext = path.suffix.lower()

        if ext in EXTENSION_CATEGORIES:
            return EXTENSION_CATEGORIES[ext]

        return "application/octet-stream", FileCategory.UNKNOWN

    def _get_recommendations(
        self,
        category: FileCategory,
        is_precompressed: bool,
        size: int,
    ) -> Tuple[str, Tuple[float, float], Optional[str]]:
        """
        Get compression recommendations based on file analysis.

        Returns:
            (recommended_mode, (min_ratio, max_ratio), warning_message)
        """
        if is_precompressed:
            warning = (
                f"This file appears to be already compressed ({category.value}). "
                f"E8ZIP typically achieves < 1.05x ratio on pre-compressed data. "
                f"Consider using FAST mode to minimize processing time."
            )
            return "fast", (0.95, 1.05), warning

        if category == FileCategory.TEXT:
            return "mythic", (5.0, 50.0), None

        if category == FileCategory.STRUCTURED:
            return "ultra", (3.0, 20.0), None

        if category == FileCategory.BINARY:
            return "normal", (1.5, 5.0), None

        # Unknown - use normal with modest expectations
        return "normal", (1.0, 3.0), None

    def should_warn(self, analysis: FileAnalysis) -> bool:
        """Check if user should be warned about this file."""
        return analysis.is_precompressed or analysis.warning is not None


# ═══════════════════════════════════════════════════════════════════════════════
#                              UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_file(path: str) -> FileAnalysis:
    """Convenience function to analyze a single file."""
    return FileAnalyzer().analyze(path)


def is_precompressed(path: str) -> bool:
    """Quick check if file is already compressed."""
    try:
        analysis = analyze_file(path)
        return analysis.is_precompressed
    except Exception:
        return False


def get_recommended_mode(path: str) -> str:
    """Get recommended compression mode for a file."""
    try:
        analysis = analyze_file(path)
        return analysis.recommended_mode
    except Exception:
        return "normal"


def format_warning(analysis: FileAnalysis) -> str:
    """Format a user-friendly warning message."""
    if not analysis.warning:
        return ""

    lines = [
        f"⚠ Warning: {analysis.warning}",
        "",
        f"  File: {analysis.path}",
        f"  Size: {analysis.size_human}",
        f"  Type: {analysis.mime_type}",
        f"  Expected ratio: {analysis.expected_ratio[0]:.2f}x - {analysis.expected_ratio[1]:.2f}x",
        "",
    ]
    return "\n".join(lines)

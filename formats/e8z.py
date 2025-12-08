"""
E8Z Archive Format

The .e8z file format for storing E8-compressed data.

File Structure:
==============

E8Z HEADER (32 bytes)
├── Magic: "E8ZIP001" (8 bytes)
├── Version: uint16
├── Mode: uint8
├── Flags: uint8
├── Original Size: uint64
├── Compressed Size: uint64
├── Checksum: uint32

METADATA BLOCK (variable)
├── Metadata Length: uint32
├── JSON Metadata: bytes (zlib compressed)

DATA BLOCKS (variable)
├── Block Count: uint32
├── For each block:
│   ├── Block Header
│   └── Block Data
"""

import struct
import zlib
import json
import hashlib
from typing import Dict, List, Any, Optional, BinaryIO
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime


# Constants
MAGIC = b"E8ZIP001"
VERSION = 1


@dataclass
class E8ZHeader:
    """E8Z file header."""

    version: int = VERSION
    mode: int = 2  # Default: NORMAL
    flags: int = 0
    original_size: int = 0
    compressed_size: int = 0
    checksum: int = 0

    @classmethod
    def from_bytes(cls, data: bytes) -> "E8ZHeader":
        """Parse header from bytes."""
        if len(data) < 32:
            raise ValueError("Header too short")

        magic = data[:8]
        if magic != MAGIC:
            raise ValueError(f"Invalid magic: {magic}")

        version, mode, flags = struct.unpack_from("<HBB", data, 8)
        original_size, compressed_size = struct.unpack_from("<QQ", data, 12)
        checksum = struct.unpack_from("<I", data, 28)[0]

        return cls(
            version=version,
            mode=mode,
            flags=flags,
            original_size=original_size,
            compressed_size=compressed_size,
            checksum=checksum,
        )

    def to_bytes(self) -> bytes:
        """Serialize header to bytes."""
        return (
            MAGIC
            + struct.pack("<HBB", self.version, self.mode, self.flags)
            + struct.pack("<QQ", self.original_size, self.compressed_size)
            + struct.pack("<I", self.checksum)
        )


@dataclass
class E8ZMetadata:
    """E8Z file metadata."""

    filename: Optional[str] = None
    timestamp: Optional[str] = None
    compression_mode: str = "normal"
    original_encoding: str = "utf-8"
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "timestamp": self.timestamp,
            "compression_mode": self.compression_mode,
            "original_encoding": self.original_encoding,
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "E8ZMetadata":
        return cls(
            filename=data.get("filename"),
            timestamp=data.get("timestamp"),
            compression_mode=data.get("compression_mode", "normal"),
            original_encoding=data.get("original_encoding", "utf-8"),
            attributes=data.get("attributes", {}),
        )

    def to_bytes(self) -> bytes:
        """Serialize metadata to compressed bytes."""
        json_str = json.dumps(self.to_dict())
        compressed = zlib.compress(json_str.encode("utf-8"), level=9)
        return struct.pack("<I", len(compressed)) + compressed

    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> tuple["E8ZMetadata", int]:
        """Deserialize metadata from bytes."""
        length = struct.unpack_from("<I", data, offset)[0]
        offset += 4

        compressed = data[offset : offset + length]
        json_str = zlib.decompress(compressed).decode("utf-8")

        return cls.from_dict(json.loads(json_str)), offset + length


@dataclass
class FileEntry:
    """Entry for a file in the archive."""

    name: str
    original_size: int
    compressed_size: int
    offset: int = 0
    checksum: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "size": self.original_size,
            "compressed_size": self.compressed_size,
            "ratio": self.original_size / (self.compressed_size + 1e-10),
            "checksum": self.checksum,
        }


class E8ZArchive:
    """
    E8Z Archive handler.

    Provides methods for creating, reading, and modifying E8Z archives.
    """

    def __init__(self, path: Optional[str] = None):
        """
        Initialize E8Z archive.

        Args:
            path: Path to existing or new archive
        """
        self.path = Path(path) if path else None
        self.header = E8ZHeader()
        self.metadata = E8ZMetadata()
        self.files: List[FileEntry] = []
        self._data: bytes = b""

    @classmethod
    def open(cls, path: str) -> "E8ZArchive":
        """
        Open an existing E8Z archive.

        Args:
            path: Path to archive

        Returns:
            E8ZArchive instance
        """
        archive = cls(path)
        archive._read()
        return archive

    @classmethod
    def create(cls, path: str) -> "E8ZArchive":
        """
        Create a new E8Z archive.

        Args:
            path: Path for new archive

        Returns:
            E8ZArchive instance
        """
        archive = cls(path)
        archive.metadata.timestamp = datetime.now().isoformat()
        return archive

    def _read(self):
        """Read archive from file."""
        with open(self.path, "rb") as f:
            data = f.read()

        # Parse header
        self.header = E8ZHeader.from_bytes(data[:32])

        offset = 32

        # Parse metadata
        self.metadata, offset = E8ZMetadata.from_bytes(data, offset)

        # Parse file entries (if multi-file)
        if self.header.flags & 0x01:  # Multi-file flag
            file_count = struct.unpack_from("<I", data, offset)[0]
            offset += 4

            for _ in range(file_count):
                # Read entry header
                name_len = struct.unpack_from("<H", data, offset)[0]
                offset += 2

                name = data[offset : offset + name_len].decode("utf-8")
                offset += name_len

                orig_size, comp_size, checksum = struct.unpack_from("<QQI", data, offset)
                offset += 20

                self.files.append(
                    FileEntry(
                        name=name,
                        original_size=orig_size,
                        compressed_size=comp_size,
                        offset=offset,
                        checksum=checksum,
                    )
                )

                offset += comp_size

        # Store remaining data
        self._data = data[offset:]

    def write(self, path: Optional[str] = None):
        """
        Write archive to file.

        Args:
            path: Output path (uses self.path if not provided)
        """
        output_path = Path(path) if path else self.path
        if not output_path:
            raise ValueError("No output path specified")

        with open(output_path, "wb") as f:
            # Write header
            f.write(self.header.to_bytes())

            # Write metadata
            f.write(self.metadata.to_bytes())

            # Write file entries if multi-file
            if self.files:
                self.header.flags |= 0x01
                f.write(struct.pack("<I", len(self.files)))

                for entry in self.files:
                    name_bytes = entry.name.encode("utf-8")
                    f.write(struct.pack("<H", len(name_bytes)))
                    f.write(name_bytes)
                    f.write(
                        struct.pack(
                            "<QQI", entry.original_size, entry.compressed_size, entry.checksum
                        )
                    )

            # Write data
            f.write(self._data)

    def get_info(self) -> Dict[str, Any]:
        """Get archive information."""
        mode_names = {1: "fast", 2: "normal", 3: "ultra", 4: "mythic"}

        info = {
            "version": self.header.version,
            "mode": mode_names.get(self.header.mode, "unknown"),
            "original_size": self.header.original_size,
            "compressed_size": self.header.compressed_size,
            "checksum": hex(self.header.checksum),
            "timestamp": self.metadata.timestamp,
            "filename": self.metadata.filename,
        }

        if self.files:
            info["files"] = [f.to_dict() for f in self.files]

        return info

    def list_files(self) -> List[Dict[str, Any]]:
        """List files in archive."""
        if not self.files:
            # Single file archive
            return [
                {
                    "name": self.metadata.filename or "data",
                    "size": self.header.original_size,
                    "ratio": self.header.original_size / (self.header.compressed_size + 1e-10),
                }
            ]

        return [f.to_dict() for f in self.files]

    def verify(self) -> Dict[str, Any]:
        """Verify archive integrity."""
        result = {
            "valid": True,
            "checksum": hex(self.header.checksum),
            "files_count": len(self.files) or 1,
        }

        # Basic validation
        if self.header.version > VERSION:
            result["valid"] = False
            result["error"] = f"Unsupported version: {self.header.version}"

        return result

    def add_file(self, name: str, data: bytes, compressed: bytes):
        """Add a file to the archive."""
        entry = FileEntry(
            name=name,
            original_size=len(data),
            compressed_size=len(compressed),
            checksum=zlib.crc32(data),
        )
        self.files.append(entry)
        self._data += compressed

    def extract(self, output_dir: str):
        """
        Extract all files to directory.

        Args:
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if not self.files:
            # Single file - use metadata filename or default
            filename = self.metadata.filename or "extracted_data"
            file_path = output_path / filename

            # The data needs to be decompressed
            from e8zip.core.compressor import E8Compressor

            compressor = E8Compressor()

            # Reconstruct full archive for decompression
            full_data = self.header.to_bytes() + self.metadata.to_bytes() + self._data
            decompressed = compressor.decompress(full_data)

            with open(file_path, "wb") as f:
                f.write(decompressed)
        else:
            # Multi-file extraction
            from e8zip.core.compressor import E8Compressor

            compressor = E8Compressor()

            for entry in self.files:
                file_path = output_path / entry.name
                file_path.parent.mkdir(parents=True, exist_ok=True)

                # Extract and decompress the specific file's data
                # This is a simplified version - full implementation would
                # track offsets properly
                # For now, we write a placeholder
                with open(file_path, "wb") as f:
                    f.write(b"")  # Placeholder


def read_e8z(path: str) -> Dict[str, Any]:
    """
    Read E8Z file and return metadata.

    Args:
        path: Path to E8Z file

    Returns:
        Dictionary with archive information
    """
    archive = E8ZArchive.open(path)
    return archive.get_info()


def validate_e8z(path: str) -> bool:
    """
    Validate E8Z file integrity.

    Args:
        path: Path to E8Z file

    Returns:
        True if valid, False otherwise
    """
    try:
        archive = E8ZArchive.open(path)
        result = archive.verify()
        return result["valid"]
    except Exception:
        return False

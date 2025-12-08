"""
Archive Utilities

Functions for creating and extracting multi-file archives.
"""

import struct
import zlib
import json
from typing import List, Dict, Any, Optional
from pathlib import Path


def create_archive(files: List[Dict[str, Any]]) -> bytes:
    """
    Create a multi-file archive.

    Args:
        files: List of dicts with 'name', 'original_size', 'compressed' keys

    Returns:
        Archive bytes
    """
    # Archive header
    header = b"E8ZARC01"  # Archive sub-format

    # File count
    header += struct.pack("<I", len(files))

    # File index
    index_data = []
    data_offset = 0

    for f in files:
        name_bytes = f["name"].encode("utf-8")
        compressed = f["compressed"]

        index_entry = {
            "name": f["name"],
            "original_size": f["original_size"],
            "compressed_size": len(compressed),
            "offset": data_offset,
        }
        index_data.append(index_entry)
        data_offset += len(compressed)

    # Serialize index
    index_json = json.dumps(index_data).encode("utf-8")
    index_compressed = zlib.compress(index_json, level=9)

    header += struct.pack("<I", len(index_compressed))
    header += index_compressed

    # Concatenate all file data
    all_data = b"".join(f["compressed"] for f in files)

    return header + all_data


def extract_archive(data: bytes, output_dir: str) -> List[str]:
    """
    Extract a multi-file archive.

    Args:
        data: Archive bytes
        output_dir: Output directory

    Returns:
        List of extracted file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    offset = 0

    # Check header
    magic = data[:8]
    if magic != b"E8ZARC01":
        raise ValueError("Not a valid E8Z archive")
    offset += 8

    # File count
    file_count = struct.unpack_from("<I", data, offset)[0]
    offset += 4

    # Index
    index_len = struct.unpack_from("<I", data, offset)[0]
    offset += 4

    index_compressed = data[offset : offset + index_len]
    offset += index_len

    index_json = zlib.decompress(index_compressed)
    index = json.loads(index_json)

    # Data section start
    data_start = offset

    # Extract files
    extracted = []

    for entry in index:
        file_path = output_path / entry["name"]
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Get compressed data for this file
        file_offset = data_start + entry["offset"]
        file_size = entry["compressed_size"]
        compressed_data = data[file_offset : file_offset + file_size]

        # Write compressed data (caller should decompress)
        with open(file_path, "wb") as f:
            f.write(compressed_data)

        extracted.append(str(file_path))

    return extracted


def list_archive(data: bytes) -> List[Dict[str, Any]]:
    """
    List contents of an archive.

    Args:
        data: Archive bytes

    Returns:
        List of file entries
    """
    offset = 0

    magic = data[:8]
    if magic != b"E8ZARC01":
        raise ValueError("Not a valid E8Z archive")
    offset += 8

    file_count = struct.unpack_from("<I", data, offset)[0]
    offset += 4

    index_len = struct.unpack_from("<I", data, offset)[0]
    offset += 4

    index_compressed = data[offset : offset + index_len]
    index_json = zlib.decompress(index_compressed)
    index = json.loads(index_json)

    return index

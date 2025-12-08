"""
E8ZIP - Command Line Interface

Provides a WinRAR-like command line experience for E8 compression.
"""

import argparse
import sys
import os
import time
from pathlib import Path
from typing import Optional

from e8zip.core.compressor import E8Compressor, CompressionMode
from e8zip.formats.e8z import E8ZArchive


def format_size(size: int) -> str:
    """Format byte size to human readable."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"


def format_ratio(original: int, compressed: int) -> str:
    """Format compression ratio."""
    if compressed == 0:
        return "∞:1"
    ratio = original / compressed
    return f"{ratio:.2f}:1"


def print_banner():
    """Print E8ZIP banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║     ███████╗ █████╗ ███████╗██╗██████╗                                        ║
║     ██╔════╝██╔══██╗╚══███╔╝██║██╔══██╗                                       ║
║     █████╗  ╚█████╔╝  ███╔╝ ██║██████╔╝                                       ║
║     ██╔══╝  ██╔══██╗ ███╔╝  ██║██╔═══╝                                        ║
║     ███████╗╚█████╔╝███████╗██║██║                                            ║
║     ╚══════╝ ╚════╝ ╚══════╝╚═╝╚═╝                                            ║
║                                                                               ║
║     Geometric Lattice Compression - Based on E8 Kaleidoscope Mind             ║
║     "Compress paths through the lattice, not just snapshots."                 ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def cmd_compress(args):
    """Handle compress command."""
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: Input file/directory not found: {input_path}")
        return 1

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix(".e8z")

    # Parse compression mode
    try:
        mode = CompressionMode[args.mode.upper()]
    except KeyError:
        print(f"Error: Invalid mode '{args.mode}'. Valid modes: fast, normal, ultra, mythic")
        return 1

    print(f"🌀 E8ZIP Compression")
    print(f"   Input:  {input_path}")
    print(f"   Output: {output_path}")
    print(f"   Mode:   {mode.name.lower()}")
    print()

    # Initialize compressor
    compressor = E8Compressor(mode=mode, verbose=args.verbose)

    start_time = time.time()

    try:
        # Check file size for streaming
        STREAMING_THRESHOLD = 100 * 1024 * 1024  # 100 MB
        is_large_file = input_path.is_file() and input_path.stat().st_size > STREAMING_THRESHOLD

        if getattr(args, "stream", False) or is_large_file:
            if args.verbose:
                print(f"   Method: Streaming (Chunked)")

            from e8zip.core.streaming import StreamingCompressor

            streamer = StreamingCompressor(compressor)

            def progress_callback(stats):
                if args.verbose:
                    sys.stdout.write(
                        f"\r   Progress: {stats.progress:.1f}% | {format_size(stats.processed_size)} | {stats.speed_mb_s:.1f} MB/s"
                    )
                    sys.stdout.flush()

            stats = streamer.compress_file(
                str(input_path), str(output_path), progress_callback=progress_callback
            )
            print()  # Newline after progress

            original_size = stats.total_size
            compressed_size = stats.compressed_size

        elif input_path.is_file():
            original_size, compressed_size = compressor.compress_file(
                str(input_path), str(output_path)
            )
        else:
            original_size, compressed_size = compressor.compress_directory(
                str(input_path), str(output_path)
            )

        elapsed = time.time() - start_time

        print()
        print(f"✓ Compression complete!")
        print(f"   Original size:   {format_size(original_size)}")
        print(f"   Compressed size: {format_size(compressed_size)}")
        print(f"   Ratio:           {format_ratio(original_size, compressed_size)}")
        print(f"   Time:            {elapsed:.2f}s")
        print(f"   Speed:           {format_size(int(original_size / elapsed))}/s")

        return 0

    except Exception as e:
        print(f"Error during compression: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def cmd_decompress(args):
    """Handle decompress command."""
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: Archive not found: {input_path}")
        return 1

    if not input_path.suffix == ".e8z":
        print(f"Warning: File does not have .e8z extension")

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Remove .e8z extension
        output_path = input_path.with_suffix("")

    print(f"🌀 E8ZIP Decompression")
    print(f"   Input:  {input_path}")
    print(f"   Output: {output_path}")
    print()

    compressor = E8Compressor(verbose=args.verbose)

    start_time = time.time()

    try:
        # Check for streaming header
        is_streaming = False
        with open(input_path, "rb") as f:
            magic = f.read(8)
            if magic == b"E8ZSTRM1":
                is_streaming = True

        if is_streaming:
            if args.verbose:
                print(f"   Method: Streaming (Chunked)")

            from e8zip.core.streaming import StreamingCompressor

            streamer = StreamingCompressor(compressor)

            def progress_callback(stats):
                if args.verbose:
                    sys.stdout.write(
                        f"\r   Progress: {stats.progress:.1f}% | {format_size(stats.processed_size)} | {stats.speed_mb_s:.1f} MB/s"
                    )
                    sys.stdout.flush()

            stats = streamer.decompress_file(
                str(input_path), str(output_path), progress_callback=progress_callback
            )
            print()  # Newline after progress

            compressed_size = input_path.stat().st_size
            original_size = (
                stats.processed_size
            )  # In decompression, processed is output size? No, processed is input.
            # StreamingStats for decompression: processed_size is bytes read from input?
            # Let's check streaming.py. decompress_file updates processed_size with len(chunk_data) which is compressed data.
            # But we want original size.
            # StreamingStats has total_size which is original_size read from header.
            original_size = stats.total_size

        else:
            compressed_size, original_size = compressor.decompress_file(
                str(input_path), str(output_path)
            )

        elapsed = time.time() - start_time

        print()
        print(f"✓ Decompression complete!")
        print(f"   Compressed size: {format_size(compressed_size)}")
        print(f"   Restored size:   {format_size(original_size)}")
        print(f"   Time:            {elapsed:.2f}s")

        return 0

    except Exception as e:
        print(f"Error during decompression: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def cmd_info(args):
    """Handle info command."""
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: Archive not found: {input_path}")
        return 1

    try:
        with open(input_path, "rb") as f:
            data = f.read()

        # Parse basic header directly
        if not data.startswith(b"E8ZIP001"):
            print(f"Error: Not a valid E8Z file")
            return 1

        import struct

        offset = 8
        version = struct.unpack_from("<H", data, offset)[0]
        offset += 2
        mode = struct.unpack_from("<B", data, offset)[0]
        offset += 1
        flags = struct.unpack_from("<B", data, offset)[0]
        offset += 1
        original_size = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        compressed_size = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        checksum = struct.unpack_from("<I", data, offset)[0]

        mode_names = {1: "fast", 2: "normal", 3: "ultra", 4: "mythic"}

        print(f"🌀 E8ZIP Archive Info")
        print(f"═" * 60)
        print(f"   File:            {input_path.name}")
        print(f"   File Size:       {format_size(len(data))}")
        print(f"   Version:         {version}")
        print(f"   Mode:            {mode_names.get(mode, 'unknown')}")
        print(f"   Original Size:   {format_size(original_size)}")
        print(f"   Compressed Size: {format_size(compressed_size)}")
        print(f"   Ratio:           {format_ratio(original_size, compressed_size)}")
        print(f"   Checksum:        {hex(checksum)}")
        print()

        return 0

    except Exception as e:
        print(f"Error reading archive: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def cmd_list(args):
    """Handle list command."""
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: Archive not found: {input_path}")
        return 1

    try:
        archive = E8ZArchive.open(str(input_path))
        files = archive.list_files()

        print(f"🌀 Contents of {input_path.name}")
        print(f"═" * 60)
        print(f"{'Name':<40} {'Size':>10} {'Ratio':>8}")
        print(f"─" * 60)

        for f in files:
            ratio = f.get("ratio", 1.0)
            print(f"{f['name']:<40} {format_size(f['size']):>10} {ratio:.2f}x")

        print(f"─" * 60)
        print(f"Total: {len(files)} file(s)")

        return 0

    except Exception as e:
        print(f"Error reading archive: {e}")
        return 1


def cmd_test(args):
    """Handle test command - verify archive integrity."""
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: Archive not found: {input_path}")
        return 1

    print(f"🌀 Testing archive: {input_path.name}")

    try:
        archive = E8ZArchive.open(str(input_path))
        result = archive.verify()

        if result["valid"]:
            print(f"✓ Archive is valid!")
            print(f"   Checksum: {result['checksum']}")
            print(f"   Files verified: {result['files_count']}")
        else:
            print(f"✗ Archive is corrupted!")
            print(f"   Error: {result.get('error', 'Unknown')}")
            return 1

        return 0

    except Exception as e:
        print(f"Error testing archive: {e}")
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="E8ZIP - Geometric Lattice Compression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  e8zip compress myfile.txt                    Compress with normal mode
  e8zip compress myfile.txt -m ultra           Compress with ultra mode
  e8zip compress mydir/ -o archive.e8z         Compress directory
  e8zip decompress archive.e8z                 Decompress archive
  e8zip info archive.e8z                       Show archive information
  e8zip list archive.e8z                       List archive contents
  e8zip test archive.e8z                       Test archive integrity

Compression Modes:
  fast    - E8 quantization only (~1.5x ratio, fastest)
  normal  - Geodesic trajectory compression (~3-5x ratio)
  ultra   - Black hole holographic encoding (~10-20x ratio)
  mythic  - Lossy semantic preservation (~50-100x ratio)
""",
    )

    parser.add_argument("--version", action="version", version="E8ZIP 1.0.0")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--no-banner", action="store_true", help="Suppress banner")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Compress command
    compress_parser = subparsers.add_parser(
        "compress", aliases=["c"], help="Compress file or directory"
    )
    compress_parser.add_argument("input", help="Input file or directory")
    compress_parser.add_argument("-o", "--output", help="Output archive path")
    compress_parser.add_argument(
        "-m",
        "--mode",
        default="normal",
        choices=["fast", "normal", "ultra", "mythic", "leech", "quip"],
        help="Compression mode (default: normal)",
    )
    compress_parser.add_argument(
        "--stream", action="store_true", help="Force streaming mode (chunked)"
    )
    compress_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # Decompress command
    decompress_parser = subparsers.add_parser(
        "decompress", aliases=["d", "x"], help="Decompress archive"
    )
    decompress_parser.add_argument("input", help="Input archive (.e8z)")
    decompress_parser.add_argument("-o", "--output", help="Output path")
    decompress_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # Info command
    info_parser = subparsers.add_parser("info", aliases=["i"], help="Show archive information")
    info_parser.add_argument("input", help="Input archive (.e8z)")
    info_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # List command
    list_parser = subparsers.add_parser("list", aliases=["l"], help="List archive contents")
    list_parser.add_argument("input", help="Input archive (.e8z)")
    list_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # Test command
    test_parser = subparsers.add_parser("test", aliases=["t"], help="Test archive integrity")
    test_parser.add_argument("input", help="Input archive (.e8z)")
    test_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if not args.no_banner:
        print_banner()

    if args.command is None:
        parser.print_help()
        return 0

    # Route to command handler
    if args.command in ["compress", "c"]:
        return cmd_compress(args)
    elif args.command in ["decompress", "d", "x"]:
        return cmd_decompress(args)
    elif args.command in ["info", "i"]:
        return cmd_info(args)
    elif args.command in ["list", "l"]:
        return cmd_list(args)
    elif args.command in ["test", "t"]:
        return cmd_test(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())

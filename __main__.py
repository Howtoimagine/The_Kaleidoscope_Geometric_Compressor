"""
E8ZIP - Command Line Entry Point

Usage:
    python -m e8zip compress <file> [--mode MODE] [--output OUTPUT]
    python -m e8zip decompress <file> [--output OUTPUT]
    python -m e8zip info <file>
"""

from e8zip.cli import main

if __name__ == "__main__":
    main()

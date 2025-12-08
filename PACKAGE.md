# E8ZIP Package Structure

```
e8zip/
├── __init__.py              # Main package init
├── e8zip_gui.py             # TUI (Terminal User Interface)
├── cli.py                   # Command-line interface
├── setup.py                 # Package installation
├── e8zip.spec               # PyInstaller spec for exe
├── start_e8zip.bat          # Windows launcher
├── start_e8zip.sh           # Linux/Mac launcher
├── README.md                # Documentation
│
├── core/                    # Core compression algorithms
│   ├── __init__.py
│   ├── compressor.py        # Main E8Compressor class
│   ├── e8_lattice.py        # E8 lattice mathematics
│   ├── gpu_backend.py       # GPU acceleration (PyTorch)
│   ├── streaming.py         # Streaming compression for large files
│   ├── hyperbolic.py        # Poincaré ball model
│   ├── trajectory.py        # Geodesic trajectory compression
│   ├── codec.py             # E8 vector quantization
│   ├── black_hole.py        # Holographic encoding
│   ├── hadamard.py          # Walsh-Hadamard transforms
│   ├── leech_lattice.py     # 24D Leech lattice
│   ├── golden_ratio.py      # Phi-based operations
│   └── repair.py            # Lattice-based error correction
│
├── utils/                   # Utility functions
│   ├── __init__.py
│   └── math_utils.py        # Safe math operations
│
└── benchmarks/              # Benchmark tools
    ├── benchmark.py         # Main benchmark suite
    ├── benchmark_winrar.py  # WinRAR comparison
    ├── benchmark_winzip.py  # Windows ZIP comparison
    ├── benchmark_7zip.py    # 7-Zip comparison
    └── benchmark_files/     # Test files
```

## Quick Start

```bash
# Install
pip install -e .

# Run TUI
python e8zip_gui.py
# OR
start_e8zip.bat  # Windows
./start_e8zip.sh # Linux/Mac

# Command line
e8zip compress myfile.txt --mode mythic
e8zip decompress myfile.e8z
```

## Compression Modes

| Mode   | Ratio     | Description                           |
|--------|-----------|---------------------------------------|
| FAST   | ~1-2x     | E8 quantization only                  |
| NORMAL | ~2-5x     | Geodesic trajectory compression       |
| ULTRA  | ~5-20x    | Black hole holographic encoding       |
| MYTHIC | ~25-100x  | Lossy semantic preservation           |
| LEECH  | ~1-2x     | 24D Leech lattice (highest fidelity)  |
| QUIP   | ~1-2x     | Hadamard + E8 (QuIP# inspired)        |

## Building Standalone Executable

```bash
pip install pyinstaller
pyinstaller e8zip.spec
# Output in dist/E8ZIP.exe
```

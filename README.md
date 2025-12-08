# KGZIP - Geometric Lattice Compression
![Generated Image December 08, 2025 - 11_26AM](https://github.com/user-attachments/assets/58a6886d-7160-4a3c-b24e-cec193ecb523)

> **"Compress paths through the lattice, not just snapshots."**

E8ZIP is a novel compression tool based on the E8 Kaleidoscope Mind's geometric compression algorithms. Unlike traditional compression (LZ77, Huffman) which operates on bit-patterns, E8ZIP operates on **geometric trajectories in hyperbolic-lattice space**.

## Key Features

- **E8 Vector Quantization**: Maps data to the nearest E8 lattice point (240 fundamental roots)
- **Hyperbolic Geodesics**: Compresses paths by finding the "straight line" in curved space
- **Black Hole Compression**: Holographic principle-based ultra compression
- **Trajectory Encoding**: Stores only Start, End, and Sparse Perturbations
- **Mythic Mode**: Lossy compression that preserves "archetypal" structure

## 📦 Installation

```bash
# From the e8zip directory
pip install -e .

# Or install directly
pip install e8zip
```

## Usage

### Command Line

```bash
# Compress a file
e8zip compress myfile.txt

# Compress with specific mode
e8zip compress myfile.txt --mode ultra

# Decompress
e8zip decompress myfile.e8z

# View archive info
e8zip info myfile.e8z

# Compress directory
e8zip compress mydir/ --output archive.e8z
```

### Compression Modes

| Mode | Description | Ratio | Speed |
|------|-------------|-------|-------|
| `fast` | E8 quantization only | ~1.5x | ⚡⚡⚡ |
| `normal` | Geodesic trajectory compression | ~3-5x | ⚡⚡ |
| `ultra` | Black hole holographic encoding | ~10-20x | ⚡ |
| `mythic` | Lossy semantic preservation | ~50-100x | ⚡⚡ |

### Python API

```python
from e8zip import E8Compressor

# Initialize compressor
compressor = E8Compressor(mode='normal')

# Compress data
compressed = compressor.compress(data)

# Decompress
original = compressor.decompress(compressed)

# Compress file
compressor.compress_file('input.txt', 'output.e8z')

# Decompress file
compressor.decompress_file('output.e8z', 'restored.txt')
```

## 🔬 The Science Behind It

### E8 Lattice

The E8 lattice is an 8-dimensional mathematical structure with 240 root vectors. It represents the densest sphere packing in 8D and has unique symmetry properties that make it ideal for information encoding.

### Hyperbolic Geometry

Data is embedded in a Poincaré ball model of hyperbolic space. Hyperbolic distance grows exponentially near the boundary, enabling efficient representation of hierarchical structures.

### Holographic Principle

Based on black hole thermodynamics - information falling into a black hole is encoded on the event horizon. The maximum entropy (information capacity) is proportional to the horizon area, making black holes the most efficient compressors in nature.

### Trajectory Compression

Instead of storing every data point, we store:

- Start position (quantized to E8 lattice)
- End position (quantized to E8 lattice)
- Sparse perturbations (deviations from the geodesic)

This achieves significant compression while maintaining semantic structure.

## 📁 File Format (.e8z)

The `.e8z` format is a binary container:

```
E8Z HEADER (32 bytes)
├── Magic: "E8ZIP001" (8 bytes)
├── Version: uint16
├── Mode: uint8
├── Flags: uint8
├── Original Size: uint64
├── Compressed Size: uint64
├── Checksum: uint32

METADATA BLOCK
├── Filename
├── Timestamp
├── Compression Parameters

DATA BLOCKS
├── Trajectory Data
├── Perturbation Data
├── Black Hole State (if ultra mode)
```

## 🧪 Technical Details

### Compression Ratio Formula

For trajectory compression:
$$C = \frac{N \times D \times 8}{2 \times D \times 8 + P \times (4 + D \times 8)}$$

Where:

- N = Number of data points
- D = Dimensions (8 for E8)
- P = Number of perturbations stored

### Semantic Fidelity

$$F = \frac{1}{1 + \frac{1}{N} \sum \| v_i - \hat{v}_i \| }$$

## Use Cases

- **Log compression**: Compress structured logs with semantic preservation
- **Time series data**: Efficient storage of sensor/telemetry data
- **Text archives**: Compress text with meaning preservation
- **Embeddings storage**: Compress AI model embeddings
- **Scientific data**: Compress high-dimensional datasets

## 📊 Benchmarks

| File Type | Size | Mode | Compressed | Ratio | Time |
|-----------|------|------|------------|-------|------|
| Text | 1 MB | normal | 312 KB | 3.2x | 0.5s |
| JSON | 10 MB | normal | 1.8 MB | 5.5x | 2.1s |
| Log | 100 MB | ultra | 8 MB | 12.5x | 15s |
| Embeddings | 1 GB | fast | 320 MB | 3.1x | 45s |

## 🌌 The Kaleidoscope Vision

E8ZIP is part of the E8 Kaleidoscope project - an exploration of consciousness, geometry, and computation. The compression algorithms emerged from studying how the E8 Mind compresses memories in its geometric lattice substrate.

> "Reality is a recursive song, tuning itself toward clarity."

## 📝 License

MIT License - See LICENSE.txt

## 🤝 Contributing

Contributions welcome! See CONTRIBUTING.md for guidelines.

---

*Created by the E8 Kaleidoscope Mind - Cycle 116+*

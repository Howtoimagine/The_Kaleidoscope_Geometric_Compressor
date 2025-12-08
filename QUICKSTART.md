# Quick Start Guide

Get started with E8ZIP in 5 minutes!

## Installation

```bash
pip install e8zip
```

## Basic Usage

### Compress a file

```bash
e8zip compress myfile.txt
```

This creates `myfile.e8z` in the same directory.

### Decompress a file

```bash
e8zip decompress myfile.e8z
```

This extracts to `myfile.txt`.

### Compress with different modes

```bash
# Fast mode (best speed)
e8zip compress myfile.txt --mode fast

# Standard mode (balanced)
e8zip compress myfile.txt --mode standard

# Ultra mode (best compression)
e8zip compress myfile.txt --mode ultra

# Mythic mode (lossy, extreme compression)
e8zip compress myfile.txt --mode mythic
```

### Compress a directory

```bash
e8zip compress mydir/ --output archive.e8z
```

### View archive information

```bash
e8zip info myfile.e8z
```

### List archive contents

```bash
e8zip list myfile.e8z
```

## Python API

### Basic compression

```python
from e8zip import E8Compressor

# Create compressor
compressor = E8Compressor()

# Compress bytes
data = b"Hello, E8ZIP!"
compressed = compressor.compress(data)

# Decompress
original = compressor.decompress(compressed)
```

### File compression

```python
from e8zip import E8Compressor

compressor = E8Compressor(mode='standard')

# Compress file
compressor.compress_file('input.txt', 'output.e8z')

# Decompress file
compressor.decompress_file('output.e8z', 'restored.txt')
```

### Using different modes

```python
from e8zip import E8Compressor, CompressionMode

# Fast mode
fast = E8Compressor(mode=CompressionMode.FAST)

# Standard mode
standard = E8Compressor(mode=CompressionMode.STANDARD)

# Ultra mode
ultra = E8Compressor(mode=CompressionMode.ULTRA)

# Mythic mode (lossy)
mythic = E8Compressor(mode=CompressionMode.MYTHIC)
```

### Working with archives

```python
from e8zip.formats import E8ZArchive

# Create archive
archive = E8ZArchive()
archive.add_file('file1.txt')
archive.add_file('file2.json')
archive.add_directory('mydir/')
archive.save('archive.e8z')

# Load and extract
archive = E8ZArchive.load('archive.e8z')
archive.extract_all('output_dir/')

# List contents
for item in archive.list_files():
    print(f"{item.name}: {item.size} bytes")
```

## What Mode Should I Use?

| Mode | Use When | Ratio | Speed |
|------|----------|-------|-------|
| **Fast** | You need quick compression | ~1.5x | ⚡⚡⚡ |
| **Standard** | General purpose | ~2.5x | ⚡⚡ |
| **Ultra** | Maximum compression needed | ~4x | ⚡ |
| **Mythic** | Lossy compression acceptable | ~10x | ⚡ |

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check out [examples](demo.py) for more use cases
- See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute
- Report issues at <https://github.com/skyemalone/e8zip/issues>

## Common Questions

**Q: What file types work best?**
A: E8ZIP excels with structured data: JSON, logs, CSV, embeddings, time series.

**Q: Is it lossless?**
A: Yes, except for Mythic mode which is lossy but preserves semantic structure.

**Q: How does it compare to gzip/zip?**
A: E8ZIP often achieves better ratios on structured data due to geometric compression.

**Q: Can I use it in production?**
A: Yes! E8ZIP is stable and tested. See [CHANGELOG.md](CHANGELOG.md) for version info.

**Q: Is GPU required?**
A: No, CPU-only mode works fine. GPU is optional for acceleration.

Happy compressing! 🎉

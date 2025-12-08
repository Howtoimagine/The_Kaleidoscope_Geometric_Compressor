# Development Setup

This guide will help you set up E8ZIP for development.

## Prerequisites

- Python 3.9 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

## Setup Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/skyemalone/e8zip.git
   cd e8zip
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Unix/macOS
   source venv/bin/activate
   ```

3. **Install in development mode**

   ```bash
   pip install -e ".[dev]"
   ```

4. **Run tests**

   ```bash
   pytest tests/ -v
   ```

5. **Check code style**

   ```bash
   black --check .
   ```

6. **Run benchmarks**

   ```bash
   python -m benchmarks.benchmark
   ```

## Project Structure

```
e8zip/
├── core/              # Core compression algorithms
│   ├── e8_lattice.py      # E8 lattice implementation
│   ├── leech_lattice.py   # Leech lattice (Lambda_24)
│   ├── hyperbolic.py      # Hyperbolic geometry
│   ├── trajectory.py      # Trajectory compression
│   ├── golden_ratio.py    # Golden ratio geometry
│   ├── hadamard.py        # Hadamard transforms
│   ├── black_hole.py      # Holographic encoding
│   ├── compressor.py      # Main compressor
│   ├── codec.py           # Codec utilities
│   ├── file_analyzer.py   # File analysis
│   ├── repair.py          # Repair algorithms
│   ├── streaming.py       # Streaming support
│   └── gpu_backend.py     # GPU acceleration
├── formats/           # Archive formats
│   ├── e8z.py            # E8Z format handler
│   └── archive.py        # Archive utilities
├── utils/             # Utilities
│   └── math_utils.py     # Math helpers
├── tests/             # Test suite
│   ├── test_compressor.py
│   └── test_e8_lattice.py
├── benchmarks/        # Performance benchmarks
│   ├── benchmark.py
│   ├── benchmark_7zip.py
│   ├── benchmark_winrar.py
│   └── benchmark_winzip.py
├── cli.py             # CLI interface
├── e8zip_gui.py       # GUI interface
├── demo.py            # Demo script
└── __main__.py        # Package entry point
```

## Development Workflow

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Make your changes
3. Add tests for new functionality
4. Run tests: `pytest`
5. Format code: `black .`
6. Commit: `git commit -m "feat: add my feature"`
7. Push: `git push origin feature/my-feature`
8. Open a Pull Request

## Testing

### Run all tests

```bash
pytest
```

### Run specific test file

```bash
pytest tests/test_compressor.py
```

### Run with coverage

```bash
pytest --cov=e8zip --cov-report=html
```

### Run verbose

```bash
pytest -v
```

## Code Style

We use `black` for code formatting:

```bash
# Check formatting
black --check .

# Auto-format
black .
```

## GPU Development

For GPU-accelerated features:

```bash
pip install -e ".[gpu]"
```

This installs PyTorch for CUDA acceleration.

## Debugging

### Enable debug output

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Use the demo script

```bash
python demo.py
```

### Debug specific module

```bash
python -m e8zip.core.e8_lattice
```

## Building Documentation

(Coming soon)

## Creating a Release

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Commit changes
4. Create tag: `git tag v1.x.x`
5. Push: `git push origin v1.x.x`
6. GitHub Actions will publish to PyPI

## Troubleshooting

### Import errors

Make sure you installed in development mode: `pip install -e .`

### Test failures

Check Python version compatibility (3.9+)

### GPU not working

Verify CUDA installation and PyTorch compatibility

## Getting Help

- Check existing issues: <https://github.com/skyemalone/e8zip/issues>
- Open a new issue with the `question` label
- Contact: <skye@example.com>

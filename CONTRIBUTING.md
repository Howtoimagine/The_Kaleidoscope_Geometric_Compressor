# Contributing to E8ZIP

Thank you for your interest in contributing to E8ZIP! This document provides guidelines and instructions for contributing.

## Code of Conduct

Be respectful, inclusive, and professional in all interactions.

## How to Contribute

### Reporting Bugs

1. Check the [issue tracker](https://github.com/skyemalone/e8zip/issues) to avoid duplicates
2. Use the bug report template
3. Include:
   - Clear description of the bug
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version)
   - Relevant logs or error messages

### Suggesting Features

1. Check existing feature requests
2. Use the feature request template
3. Clearly describe:
   - The problem you're trying to solve
   - Your proposed solution
   - Alternative solutions considered
   - Impact on existing functionality

### Pull Requests

1. **Fork the repository**

   ```bash
   git clone https://github.com/skyemalone/e8zip.git
   cd e8zip
   ```

2. **Create a feature branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Set up development environment**

   ```bash
   pip install -e ".[dev]"
   ```

4. **Make your changes**
   - Write clear, readable code
   - Follow existing code style
   - Add tests for new functionality
   - Update documentation as needed

5. **Run tests**

   ```bash
   pytest tests/
   ```

6. **Format code**

   ```bash
   black .
   ```

7. **Commit your changes**

   ```bash
   git add .
   git commit -m "Add feature: description"
   ```

8. **Push to your fork**

   ```bash
   git push origin feature/your-feature-name
   ```

9. **Open a Pull Request**
   - Use the PR template
   - Link related issues
   - Describe your changes clearly
   - Request reviews

## Development Guidelines

### Code Style

- Follow PEP 8 style guide
- Use type hints where appropriate
- Write docstrings for public functions and classes
- Keep functions focused and small
- Use meaningful variable names

### Testing

- Write unit tests for new features
- Maintain or improve code coverage
- Test edge cases and error conditions
- Use pytest fixtures for common setup

### Documentation

- Update README.md if adding user-facing features
- Add docstrings to new functions/classes
- Update CHANGELOG.md with your changes
- Include code examples for new APIs

### Commit Messages

Follow conventional commit format:

```
type(scope): subject

body (optional)

footer (optional)
```

Types:

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Test additions or changes
- `chore`: Build process or auxiliary tool changes

Examples:

```
feat(compression): add Leech lattice compression mode
fix(cli): handle empty input files gracefully
docs(readme): add installation instructions for Windows
```

## Project Structure

```
e8zip/
├── core/           # Core compression algorithms
│   ├── e8_lattice.py
│   ├── leech_lattice.py
│   ├── hyperbolic.py
│   ├── trajectory.py
│   └── ...
├── formats/        # Archive format handlers
│   ├── e8z.py
│   └── archive.py
├── utils/          # Utility functions
├── cli.py          # Command-line interface
├── tests/          # Test suite
└── benchmarks/     # Performance benchmarks
```

## Areas Needing Help

- **Performance optimization** - GPU acceleration, parallel processing
- **New compression modes** - Experimental lattice structures
- **Format support** - Import/export from other formats
- **Documentation** - Examples, tutorials, API docs
- **Testing** - Edge cases, large file testing
- **Benchmarks** - Comparison with other tools

## Building from Source

```bash
# Clone repository
git clone https://github.com/skyemalone/e8zip.git
cd e8zip

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run benchmarks
python -m benchmarks.benchmark
```

## Release Process

(For maintainers)

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create git tag: `git tag v1.x.x`
4. Push tag: `git push origin v1.x.x`
5. GitHub Actions will build and publish to PyPI

## Questions?

- Open an issue for questions
- Join discussions in GitHub Discussions
- Contact: <skyemalone@example.com>

Thank you for contributing to E8ZIP! 🎉

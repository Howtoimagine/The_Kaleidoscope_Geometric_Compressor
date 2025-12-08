"""
E8ZIP - Geometric Lattice Compression

Setup script for installation.
"""

from setuptools import setup, find_packages
import os

# Read README with proper encoding
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "E8ZIP - Geometric Lattice Compression"

setup(
    name="e8zip",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "rich>=13.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
        ],
        "full": [
            "scipy>=1.9.0",
            "tqdm>=4.65.0",
        ],
        "gui": [
            "rich>=13.0.0",
        ],
        "gpu": [
            # PyTorch for NVIDIA CUDA acceleration
            "torch>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "e8zip=e8zip.cli:main",
            "e8zip-gui=e8zip.e8zip_gui:main",
        ],
    },
    author="Skye Malone",
    author_email="skye@example.com",
    description="E8ZIP - Geometric Lattice Compression",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/skyemalone/e8zip",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: System :: Archiving :: Compression",
    ],
    python_requires=">=3.9",
)

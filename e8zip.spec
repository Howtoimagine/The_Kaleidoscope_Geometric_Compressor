# -*- mode: python ; coding: utf-8 -*-
"""
E8ZIP PyInstaller Spec File

Build standalone executable with:
    pyinstaller e8zip.spec

Requires:
    pip install pyinstaller
"""

import os
import sys
from pathlib import Path

# Get the e8zip directory
spec_dir = os.path.dirname(os.path.abspath(SPEC))
e8zip_dir = spec_dir

block_cipher = None

# Collect all e8zip modules
a = Analysis(
    [os.path.join(e8zip_dir, 'e8zip_gui.py')],
    pathex=[e8zip_dir],
    binaries=[],
    datas=[
        (os.path.join(e8zip_dir, 'README.md'), '.'),
    ],
    hiddenimports=[
        'e8zip',
        'e8zip.core',
        'e8zip.core.compressor',
        'e8zip.core.e8_lattice',
        'e8zip.core.hyperbolic',
        'e8zip.core.trajectory',
        'e8zip.core.codec',
        'e8zip.core.black_hole',
        'e8zip.core.hadamard',
        'e8zip.core.leech_lattice',
        'e8zip.core.golden_ratio',
        'e8zip.core.repair',
        'e8zip.utils',
        'e8zip.utils.math_utils',
        'rich',
        'rich.console',
        'rich.panel',
        'rich.table',
        'rich.progress',
        'numpy',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'scipy',
        'pandas',
        'PIL',
        'tkinter',
        'PyQt5',
        'PySide2',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='E8ZIP',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path here if available
)

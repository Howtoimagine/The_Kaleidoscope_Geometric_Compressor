#!/bin/bash
# E8ZIP - Geometric Lattice Compression Launcher

echo ""
echo "    ███████╗ █████╗     ███████╗██╗██████╗ "
echo "    ██╔════╝██╔══██╗    ╚══███╔╝██║██╔══██╗"
echo "    █████╗  ╚█████╔╝      ███╔╝ ██║██████╔╝"
echo "    ██╔══╝  ██╔══██╗     ███╔╝  ██║██╔═══╝ "
echo "    ███████╗╚█████╔╝    ███████╗██║██║     "
echo "    ╚══════╝ ╚════╝     ╚══════╝╚═╝╚═╝     "
echo ""
echo "           Geometric Lattice Compression"
echo '    "Compress paths through the lattice, not just snapshots"'
echo ""
echo "    ●───●───●───●"
echo "   ╱│╲ ╱│╲ ╱│╲ ╱│╲"
echo "  ● │ ● │ ● │ ● │ ●"
echo "  │╲│╱│╲│╱│╲│╱│╲│╱│"
echo "  ●─●─●─●─●─●─●─●─●"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 not found. Please install Python 3.9+"
    exit 1
fi

# Check rich
python3 -c "import rich" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[INFO] Installing required packages..."
    pip3 install rich
fi

# Run GUI
echo "[INFO] Starting E8ZIP..."
echo ""
python3 e8zip_gui.py

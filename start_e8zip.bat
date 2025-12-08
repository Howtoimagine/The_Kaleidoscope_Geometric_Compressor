@echo off
title E8ZIP - Geometric Lattice Compression
color 0B

echo.
echo     ███████╗ █████╗     ███████╗██╗██████╗ 
echo     ██╔════╝██╔══██╗    ╚══███╔╝██║██╔══██╗
echo     █████╗  ╚█████╔╝      ███╔╝ ██║██████╔╝
echo     ██╔══╝  ██╔══██╗     ███╔╝  ██║██╔═══╝ 
echo     ███████╗╚█████╔╝    ███████╗██║██║     
echo     ╚══════╝ ╚════╝     ╚══════╝╚═╝╚═╝     
echo.
echo            Geometric Lattice Compression
echo     "Compress paths through the lattice, not just snapshots"
echo.
echo     ●───●───●───●
echo    ╱│╲ ╱│╲ ╱│╲ ╱│╲
echo   ● │ ● │ ● │ ● │ ●
echo   │╲│╱│╲│╱│╲│╱│╲│╱│
echo   ●─●─●─●─●─●─●─●─●
echo.

cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.9+
    pause
    exit /b 1
)

REM Check if rich is installed
python -c "import rich" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing required packages...
    pip install rich
)

REM Run the GUI
echo [INFO] Starting E8ZIP...
echo.
python e8zip_gui.py

pause

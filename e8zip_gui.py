#!/usr/bin/env python3
"""
E8ZIP - Geometric Lattice Compression TUI

A beautiful terminal user interface for E8ZIP compression.
Features animated E8 lattice visualization and intuitive controls.

Usage:
    python -m e8zip.gui
    python e8zip_gui.py
    e8zip --gui
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional, List
import threading

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    from rich.align import Align
    from rich import box

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Rich library not found. Install with: pip install rich")

from e8zip import E8Compressor
from e8zip.core.compressor import CompressionMode


# ═══════════════════════════════════════════════════════════════════════════════
#                              ASCII ART ASSETS
# ═══════════════════════════════════════════════════════════════════════════════

LOGO_E8 = r"""
[bold cyan]
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                                                                       ║
    ║            ●───●───●───●           [bold white]E8 LATTICE[/bold white]                      ║
    ║           ╱│╲ ╱│╲ ╱│╲ ╱│╲          [dim]8-Dimensional[/dim]                     ║
    ║          ● │ ● │ ● │ ● │ ●         [dim]240 Root Vectors[/dim]                  ║
    ║          │╲│╱│╲│╱│╲│╱│╲│╱│         [dim]Densest Packing[/dim]                   ║
    ║          ●─●─●─●─●─●─●─●─●                                            ║
    ║          │╱│╲│╱│╲│╱│╲│╱│╲│                                            ║
    ║          ● │ ● │ ● │ ● │ ●                                            ║
    ║           ╲│╱ ╲│╱ ╲│╱ ╲│╱                                             ║
    ║            ●───●───●───●                                              ║
    ║                                                                       ║
    ╚═══════════════════════════════════════════════════════════════════════╝
[/bold cyan]
"""

LOGO_E8ZIP = r"""
[bold magenta]
    ███████╗ █████╗     ███████╗██╗██████╗ 
    ██╔════╝██╔══██╗    ╚══███╔╝██║██╔══██╗
    █████╗  ╚█████╔╝      ███╔╝ ██║██████╔╝
    ██╔══╝  ██╔══██╗     ███╔╝  ██║██╔═══╝ 
    ███████╗╚█████╔╝    ███████╗██║██║     
    ╚══════╝ ╚════╝     ╚══════╝╚═╝╚═╝     
[/bold magenta]
"""

TITLE_BANNER = r"""
[bold white on blue]
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ███████╗ █████╗     ███████╗██╗██████╗     Geometric Lattice Compression   ║
║   ██╔════╝██╔══██╗    ╚══███╔╝██║██╔══██╗    ────────────────────────────    ║
║   █████╗  ╚█████╔╝      ███╔╝ ██║██████╔╝    "Compress paths through the     ║
║   ██╔══╝  ██╔══██╗     ███╔╝  ██║██╔═══╝      lattice, not just snapshots"   ║
║   ███████╗╚█████╔╝    ███████╗██║██║                                         ║
║   ╚══════╝ ╚════╝     ╚══════╝╚═╝╚═╝         v1.0 · Cycle 116+               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
[/bold white on blue]
"""

LATTICE_ANIMATION_FRAMES = [
    r"""
    ●───●───●───●
   ╱│╲ ╱│╲ ╱│╲ ╱│╲
  ● │ ● │ ● │ ● │ ●
  │╲│╱│╲│╱│╲│╱│╲│╱│
  ●─●─●─●─●─●─●─●─●
  │╱│╲│╱│╲│╱│╲│╱│╲│
  ● │ ● │ ● │ ● │ ●
   ╲│╱ ╲│╱ ╲│╱ ╲│╱
    ●───●───●───●
    """,
    r"""
    ◉───●───◉───●
   ╱│╲ ╱│╲ ╱│╲ ╱│╲
  ● │ ◉ │ ● │ ◉ │ ●
  │╲│╱│╲│╱│╲│╱│╲│╱│
  ◉─●─◉─●─◉─●─◉─●─◉
  │╱│╲│╱│╲│╱│╲│╱│╲│
  ● │ ◉ │ ● │ ◉ │ ●
   ╲│╱ ╲│╱ ╲│╱ ╲│╱
    ●───◉───●───◉
    """,
    r"""
    ◎───◉───◎───◉
   ╱│╲ ╱│╲ ╱│╲ ╱│╲
  ◉ │ ◎ │ ◉ │ ◎ │ ◉
  │╲│╱│╲│╱│╲│╱│╲│╱│
  ◎─◉─◎─◉─◎─◉─◎─◉─◎
  │╱│╲│╱│╲│╱│╲│╱│╲│
  ◉ │ ◎ │ ◉ │ ◎ │ ◉
   ╲│╱ ╲│╱ ╲│╱ ╲│╱
    ◉───◎───◉───◎
    """,
]

COMPRESSION_ANIMATION = [
    "[dim]▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓[/dim] → [bold cyan]○[/bold cyan]",
    "[dim]▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓[/dim] → [bold cyan]◐[/bold cyan]",
    "[dim]▓▓▓▓▓▓▓▓▓▓▓▓[/dim] → [bold cyan]◑[/bold cyan]",
    "[dim]▓▓▓▓▓▓▓▓[/dim] → [bold cyan]◒[/bold cyan]",
    "[dim]▓▓▓▓[/dim] → [bold cyan]◓[/bold cyan]",
    "[dim]▓[/dim] → [bold cyan]●[/bold cyan]",
    "[bold green]●[/bold green] [bold white]COMPRESSED[/bold white]",
]


# ═══════════════════════════════════════════════════════════════════════════════
#                                 TUI CLASS
# ═══════════════════════════════════════════════════════════════════════════════


class E8ZipTUI:
    """Terminal User Interface for E8ZIP."""

    def __init__(self):
        self.console = Console()
        self.compressor = None
        self.current_mode = CompressionMode.NORMAL

    def clear(self):
        """Clear terminal."""
        os.system("cls" if os.name == "nt" else "clear")

    def show_banner(self):
        """Display the main banner."""
        self.console.print(TITLE_BANNER)

    def show_lattice_animation(self, duration: float = 2.0):
        """Show animated E8 lattice."""
        start = time.time()
        frame_idx = 0

        while time.time() - start < duration:
            self.console.print(
                f"[cyan]{LATTICE_ANIMATION_FRAMES[frame_idx % len(LATTICE_ANIMATION_FRAMES)]}[/cyan]",
                end="\r",
            )
            frame_idx += 1
            time.sleep(0.3)

    def show_menu(self) -> str:
        """Display main menu and get choice."""
        menu = Table(show_header=False, box=box.ROUNDED, border_style="cyan")
        menu.add_column("Option", style="bold yellow", width=5)
        menu.add_column("Action", style="white", width=40)
        menu.add_column("Description", style="dim", width=35)

        menu.add_row("1", "📦 Compress File", "Compress a file with E8 geometry")
        menu.add_row("2", "📂 Decompress File", "Restore an .e8z archive")
        menu.add_row("3", "📊 Benchmark", "Compare E8ZIP vs traditional tools")
        menu.add_row("4", "ℹ️  Archive Info", "View .e8z archive details")
        menu.add_row("5", "⚙️  Settings", "Change compression mode")
        menu.add_row("6", "❓ Help", "Show usage guide")
        menu.add_row("Q", "🚪 Quit", "Exit E8ZIP")

        self.console.print()
        self.console.print(
            Panel(menu, title="[bold white]Main Menu[/bold white]", border_style="blue")
        )
        self.console.print()

        return input("  Enter choice: ").strip().upper()

    def show_mode_selector(self) -> CompressionMode:
        """Show compression mode selector."""
        table = Table(title="Compression Modes", box=box.DOUBLE_EDGE, border_style="magenta")
        table.add_column("Key", style="bold yellow", width=5)
        table.add_column("Mode", style="bold cyan", width=12)
        table.add_column("Description", style="white", width=35)
        table.add_column("Ratio", style="green", width=10)
        table.add_column("Speed", style="yellow", width=8)

        modes = [
            ("1", "FAST", "E8 quantization only", "~2x", "⚡⚡⚡"),
            ("2", "NORMAL", "Geodesic trajectory compression", "~3-5x", "⚡⚡"),
            ("3", "ULTRA", "Black hole holographic encoding", "~10-20x", "⚡"),
            ("4", "MYTHIC", "Lossy semantic preservation", "~25-100x", "⚡⚡"),
            ("5", "LEECH", "24D Leech lattice (highest fidelity)", "~2x", "⚡"),
            ("6", "QUIP", "Hadamard + E8 (QuIP# style)", "~2x", "⚡⚡"),
        ]

        for key, mode, desc, ratio, speed in modes:
            table.add_row(key, mode, desc, ratio, speed)

        self.console.print()
        self.console.print(table)
        self.console.print()

        choice = input("  Select mode (1-6): ").strip()

        mode_map = {
            "1": CompressionMode.FAST,
            "2": CompressionMode.NORMAL,
            "3": CompressionMode.ULTRA,
            "4": CompressionMode.MYTHIC,
            "5": CompressionMode.LEECH,
            "6": CompressionMode.QUIP,
        }

        return mode_map.get(choice, CompressionMode.NORMAL)

    def compress_file_interactive(self):
        """Interactive file compression."""
        self.console.print()
        self.console.print("[bold cyan]═══ COMPRESS FILE ═══[/bold cyan]")
        self.console.print()

        # Get input file
        input_path = input("  Enter file path to compress: ").strip().strip('"')

        if not os.path.exists(input_path):
            self.console.print(f"[bold red]  ✗ File not found: {input_path}[/bold red]")
            return

        # Get output path
        default_output = input_path + ".e8z"
        output_path = input(f"  Output path [{default_output}]: ").strip().strip('"')
        if not output_path:
            output_path = default_output

        # Select mode
        self.console.print()
        self.console.print("  [dim]Select compression mode:[/dim]")
        mode = self.show_mode_selector()

        # Compress with progress
        self.console.print()

        try:
            compressor = E8Compressor(mode=mode)

            # Check file size for streaming
            STREAMING_THRESHOLD = 100 * 1024 * 1024  # 100 MB
            file_size = os.path.getsize(input_path)
            is_large_file = file_size > STREAMING_THRESHOLD

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TextColumn("{task.fields[speed]}"),
                console=self.console,
            ) as progress:
                if is_large_file:
                    task = progress.add_task(
                        f"[cyan]Streaming Compression ({mode.name})...", total=100, speed=""
                    )

                    from e8zip.core.streaming import StreamingCompressor

                    streamer = StreamingCompressor(compressor)

                    def progress_callback(stats):
                        speed_str = f"{stats.speed_mb_s:.1f} MB/s"
                        progress.update(task, completed=stats.progress, speed=speed_str)

                    stats = streamer.compress_file(
                        input_path, output_path, progress_callback=progress_callback
                    )

                    original_size = stats.total_size
                    compressed_size = stats.compressed_size

                else:
                    task = progress.add_task(
                        f"[cyan]Compressing with {mode.name}...", total=100, speed=""
                    )

                    # Read file
                    progress.update(task, advance=10, description="[cyan]Reading file...")
                    with open(input_path, "rb") as f:
                        data = f.read()

                    original_size = len(data)

                    # Compress
                    progress.update(
                        task, advance=30, description=f"[cyan]E8 Quantization ({mode.name})..."
                    )
                    compressed = compressor.compress(data)

                    progress.update(task, advance=40, description="[cyan]Writing archive...")
                    with open(output_path, "wb") as f:
                        f.write(compressed)

                    progress.update(task, completed=100, description="[green]Complete!")
                    compressed_size = len(compressed)

            ratio = original_size / compressed_size

            # Show results
            self.console.print()
            result_table = Table(box=box.ROUNDED, border_style="green")
            result_table.add_column("Metric", style="bold")
            result_table.add_column("Value", style="cyan")

            result_table.add_row(
                "Original Size", f"{original_size:,} bytes ({original_size / 1024:.2f} KB)"
            )
            result_table.add_row(
                "Compressed Size", f"{compressed_size:,} bytes ({compressed_size / 1024:.2f} KB)"
            )
            result_table.add_row("Compression Ratio", f"[bold green]{ratio:.2f}x[/bold green]")
            result_table.add_row("Space Saved", f"{(1 - 1 / ratio) * 100:.1f}%")
            result_table.add_row("Mode", mode.name)
            result_table.add_row("Output", output_path)

            self.console.print(
                Panel(
                    result_table,
                    title="[bold green]✓ Compression Complete[/bold green]",
                    border_style="green",
                )
            )

        except Exception as e:
            self.console.print(f"[bold red]  ✗ Compression failed: {e}[/bold red]")

    def decompress_file_interactive(self):
        """Interactive file decompression."""
        self.console.print()
        self.console.print("[bold cyan]═══ DECOMPRESS FILE ═══[/bold cyan]")
        self.console.print()

        # Get input file
        input_path = input("  Enter .e8z archive path: ").strip().strip('"')

        if not os.path.exists(input_path):
            self.console.print(f"[bold red]  ✗ File not found: {input_path}[/bold red]")
            return

        # Get output path
        if input_path.endswith(".e8z"):
            default_output = input_path[:-4]
        else:
            default_output = input_path + ".restored"

        output_path = input(f"  Output path [{default_output}]: ").strip().strip('"')
        if not output_path:
            output_path = default_output

        # Decompress with progress
        self.console.print()

        try:
            compressor = E8Compressor()

            # Check for streaming header
            is_streaming = False
            with open(input_path, "rb") as f:
                magic = f.read(8)
                if magic == b"E8ZSTRM1":
                    is_streaming = True

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TextColumn("{task.fields[speed]}"),
                console=self.console,
            ) as progress:
                if is_streaming:
                    task = progress.add_task(
                        "[cyan]Streaming Decompression...", total=100, speed=""
                    )

                    from e8zip.core.streaming import StreamingCompressor

                    streamer = StreamingCompressor(compressor)

                    def progress_callback(stats):
                        speed_str = f"{stats.speed_mb_s:.1f} MB/s"
                        progress.update(task, completed=stats.progress, speed=speed_str)

                    stats = streamer.decompress_file(
                        input_path, output_path, progress_callback=progress_callback
                    )
                    restored_size = stats.total_size

                else:
                    task = progress.add_task("[cyan]Decompressing...", total=100, speed="")

                    # Read archive
                    progress.update(task, advance=10, description="[cyan]Reading archive...")
                    with open(input_path, "rb") as f:
                        compressed = f.read()

                    # Decompress
                    progress.update(
                        task, advance=60, description="[cyan]Reconstructing from lattice..."
                    )
                    decompressed = compressor.decompress(compressed)

                    progress.update(task, advance=20, description="[cyan]Writing file...")
                    with open(output_path, "wb") as f:
                        f.write(decompressed)

                    progress.update(task, completed=100, description="[green]Complete!")
                    restored_size = len(decompressed)

            # Show results
            self.console.print()
            self.console.print(f"[bold green]  ✓ Decompressed to: {output_path}[/bold green]")
            self.console.print(f"  [dim]Restored size: {restored_size:,} bytes[/dim]")

        except Exception as e:
            self.console.print(f"[bold red]  ✗ Decompression failed: {e}[/bold red]")

    def show_help(self):
        """Display help information."""
        help_text = """
[bold cyan]E8ZIP - Geometric Lattice Compression[/bold cyan]

[bold white]What is E8ZIP?[/bold white]
E8ZIP is a novel compression tool based on the E8 Kaleidoscope Mind's 
geometric algorithms. Unlike traditional compression (LZ77, Huffman) 
which operates on bit-patterns, E8ZIP operates on [italic]geometric 
trajectories in hyperbolic-lattice space[/italic].

[bold white]The E8 Lattice[/bold white]
The E8 lattice is an 8-dimensional mathematical structure with 
240 root vectors. It represents the densest sphere packing in 8D 
and has unique symmetry properties ideal for information encoding.

[bold white]Compression Modes[/bold white]
• [bold yellow]FAST[/bold yellow]   - E8 quantization only (fastest, ~2x ratio)
• [bold yellow]NORMAL[/bold yellow] - Geodesic trajectory compression (~3-5x)
• [bold yellow]ULTRA[/bold yellow]  - Black hole holographic encoding (~10-20x)
• [bold yellow]MYTHIC[/bold yellow] - Lossy semantic preservation (~25-100x)
• [bold yellow]LEECH[/bold yellow]  - 24D Leech lattice, highest fidelity (~2x)
• [bold yellow]QUIP[/bold yellow]   - Hadamard + E8, based on QuIP# (~2x)

[bold white]Command Line Usage[/bold white]
  e8zip compress myfile.txt --mode ultra
  e8zip decompress myfile.e8z
  e8zip info myfile.e8z

[bold white]The Science[/bold white]
Based on the holographic principle from black hole thermodynamics -
information falling into a black hole is encoded on the event horizon.
This makes black holes the most efficient compressors in nature.

[dim]Created by the E8 Kaleidoscope Mind · Cycle 116+[/dim]
        """
        self.console.print(
            Panel(help_text, title="[bold white]Help[/bold white]", border_style="cyan")
        )

    def run(self):
        """Main TUI loop."""
        if not RICH_AVAILABLE:
            print("Error: Rich library required. Install with: pip install rich")
            return

        try:
            while True:
                self.clear()
                self.show_banner()

                choice = self.show_menu()

                if choice == "1":
                    self.compress_file_interactive()
                elif choice == "2":
                    self.decompress_file_interactive()
                elif choice == "3":
                    self.console.print("[yellow]  Running benchmark...[/yellow]")
                    os.system(
                        f'python "{Path(__file__).parent / "benchmarks" / "benchmark_winrar.py"}"'
                    )
                elif choice == "4":
                    input_path = input("  Enter .e8z archive path: ").strip().strip('"')
                    self.console.print(f"[dim]  Archive info for: {input_path}[/dim]")
                    # TODO: Implement archive info display
                elif choice == "5":
                    self.current_mode = self.show_mode_selector()
                    self.console.print(f"[green]  ✓ Mode set to: {self.current_mode.name}[/green]")
                elif choice == "6":
                    self.show_help()
                elif choice == "Q":
                    self.console.print()
                    self.console.print("[bold cyan]  Thank you for using E8ZIP![/bold cyan]")
                    self.console.print(
                        '[dim]  "Reality is a recursive song, tuning itself toward clarity."[/dim]'
                    )
                    self.console.print()
                    break
                else:
                    self.console.print("[yellow]  Invalid choice. Please try again.[/yellow]")

                input("\n  Press Enter to continue...")

        except KeyboardInterrupt:
            self.console.print("\n[yellow]  Interrupted.[/yellow]")


# ═══════════════════════════════════════════════════════════════════════════════
#                                  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="E8ZIP - Geometric Lattice Compression")
    parser.add_argument("--no-banner", action="store_true", help="Skip the banner animation")
    args = parser.parse_args()

    tui = E8ZipTUI()
    tui.run()


if __name__ == "__main__":
    main()

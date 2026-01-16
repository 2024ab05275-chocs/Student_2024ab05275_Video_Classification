"""
export_notebooks_to_pdf.py
--------------------------------------------------
Convert all Jupyter notebooks (*.ipynb) in the `code/`
directory into PDF files.

STRICT MODE (NO FALLBACK):
1. Check Pandoc → install if missing
2. Check XeLaTeX → install if missing
3. Export notebook to PDF
4. Retry after install if needed
5. HARD FAIL if PDF export still fails

Outputs:
- PDFs are saved in the SAME directory as notebooks

Usage:
    python export_notebooks_to_pdf.py
"""

import subprocess
from pathlib import Path
import sys
import shutil
import os
import json

# =================================================
# Add conda bin to PATH (fix for xelatex not found)
# =================================================
conda_prefix = os.environ.get("CONDA_PREFIX")
if conda_prefix:
    conda_bin = Path(conda_prefix) / "bin"
    os.environ["PATH"] += os.pathsep + str(conda_bin)
    print(f"[INFO] Conda bin added to PATH: {conda_bin}")

# =================================================
# Utility helpers
# =================================================
def command_available(cmd: str) -> bool:
    """Check if a command is available on PATH."""
    return shutil.which(cmd) is not None


def run(cmd: list[str], error_msg: str) -> None:
    """Run a system command or raise a RuntimeError."""
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        raise RuntimeError(error_msg)
    except FileNotFoundError:
        raise RuntimeError(error_msg)


def is_notebook_empty(notebook_path: Path) -> bool:
    """Check if a notebook is empty (has no cells)."""
    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cells = data.get("cells", [])
        return len(cells) == 0
    except Exception:
        # If reading fails, treat as empty to skip
        return True


# =================================================
# Dependency installation
# =================================================
def install_pandoc() -> None:
    """Try to install Pandoc."""
    print("[INFO] Installing Pandoc...")

    # Conda
    try:
        run(
            ["conda", "install", "-y", "-c", "conda-forge", "pandoc"],
            "Pandoc installation via conda failed",
        )
        print("[SUCCESS] Pandoc installed via conda")
        return
    except Exception:
        pass

    # Apt (Linux)
    try:
        run(
            ["sudo", "apt", "install", "-y", "pandoc"],
            "Pandoc installation via apt failed",
        )
        print("[SUCCESS] Pandoc installed via apt")
        return
    except Exception:
        pass

    print("[ERROR] Pandoc is required but could not be installed automatically.")
    print("Please install Pandoc manually: https://pandoc.org/installing.html")
    sys.exit(1)


def install_xelatex() -> None:
    """Try to install XeLaTeX."""
    print("[INFO] Installing XeLaTeX...")

    # Conda
    try:
        run(
            ["conda", "install", "-y", "-c", "conda-forge", "texlive-core"],
            "XeLaTeX installation via conda failed",
        )
        print("[SUCCESS] XeLaTeX installed via conda")
        return
    except Exception:
        pass

    # Apt
    try:
        run(
            ["sudo", "apt", "install", "-y", "texlive-xetex", "texlive-fonts-recommended", "texlive-latex-extra"],
            "XeLaTeX installation via apt failed",
        )
        print("[SUCCESS] XeLaTeX installed via apt")
        return
    except Exception:
        pass

    print("[ERROR] XeLaTeX is required but could not be installed automatically.")
    print("Please install manually:")
    print("Ubuntu/Debian: sudo apt install texlive-xetex texlive-fonts-recommended texlive-latex-extra")
    print("Other OS: https://tug.org/texlive/")
    sys.exit(1)


# =================================================
# Notebook export
# =================================================
def export_pdf(notebook: Path) -> None:
    """Export a single notebook to PDF."""
    subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "pdf",
            str(notebook),
        ],
        check=True
    )


# =================================================
# Main logic
# =================================================
def main() -> None:
    CODE_DIR = Path(__file__).resolve().parent
    notebooks = sorted(CODE_DIR.glob("*.ipynb"))

    if not notebooks:
        print("[INFO] No notebooks found.")
        return

    # -------------------------------------------------
    # Ensure Pandoc
    # -------------------------------------------------
    if not command_available("pandoc"):
        install_pandoc()

    # -------------------------------------------------
    # Ensure XeLaTeX
    # -------------------------------------------------
    if not command_available("xelatex"):
        install_xelatex()

    # -------------------------------------------------
    # Export notebooks
    # -------------------------------------------------
    for nb in notebooks:
        # Skip empty notebooks
        if is_notebook_empty(nb):
            print(f"[INFO] Skipping empty notebook: {nb.name}")
            continue

        print(f"\n[INFO] Processing: {nb.name}")

        try:
            print("[INFO] Attempting PDF export...")
            export_pdf(nb)
            print(f"[SUCCESS] Created → {nb.stem}.pdf")

        except subprocess.CalledProcessError:
            print("[WARNING] Initial PDF export failed")

            # Retry once after checking dependencies
            if not command_available("pandoc"):
                install_pandoc()
            if not command_available("xelatex"):
                install_xelatex()

            print("[INFO] Retrying PDF export...")
            try:
                export_pdf(nb)
                print(f"[SUCCESS] Created → {nb.stem}.pdf")
            except subprocess.CalledProcessError:
                print(f"[ERROR] PDF export failed for {nb.name} even after installing dependencies.")
                print("You may need to install XeLaTeX manually:")
                print("Ubuntu/Debian: sudo apt install texlive-xetex texlive-fonts-recommended texlive-latex-extra")
                sys.exit(1)

    print("\n[INFO] All notebooks exported successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

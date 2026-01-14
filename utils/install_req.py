# utils/install_req.py

"""
Virtual environment dependency installer.

Called by start.py after venv creation.

Responsibilities:
- activate correct venv python
- upgrade pip safely
- install requirements.txt
- fail fast on errors
"""

import os
import sys
import subprocess
from pathlib import Path


# ==========================================================
# Helpers
# ==========================================================
def get_venv_python(venv_dir: str) -> str:
    """
    Return the path to the venv's python executable.
    """
    if os.name == "nt":  # Windows
        python_path = Path(venv_dir) / "Scripts" / "python.exe"
    else:  # Linux / macOS
        python_path = Path(venv_dir) / "bin" / "python"

    if not python_path.exists():
        raise FileNotFoundError(
            f"Python executable not found in venv: {python_path}"
        )

    return str(python_path)


def run(cmd: list[str]):
    """
    Run a subprocess command with strict error handling.
    """
    subprocess.check_call(cmd)


# ==========================================================
# Main installer
# ==========================================================
def install_requirements(venv_dir: str):
    print(">--Installing dependencies--<")

    venv_python = get_venv_python(venv_dir)

    project_root = Path(__file__).resolve().parents[1]
    requirements_file = project_root / "requirements.txt"

    if not requirements_file.exists():
        raise FileNotFoundError(
            f"requirements.txt not found at {requirements_file}"
        )

    # Upgrade pip first (critical for wheels)
    print("🔧 Upgrading pip...")
    run([venv_python, "-m", "pip", "install", "--upgrade", "pip"])

    # Install dependencies
    print("📦 Installing requirements...")
    run([
        venv_python,
        "-m",
        "pip",
        "install",
        "-r",
        str(requirements_file)
    ])

    print("✅ All dependencies installed successfully")


# ==========================================================
# CLI entry
# ==========================================================
def main():
    if len(sys.argv) != 2:
        print("Usage: python install_req.py <venv_dir>")
        sys.exit(1)

    venv_dir = sys.argv[1]

    try:
        install_requirements(venv_dir)
    except Exception as e:
        print(f"❌ Dependency installation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

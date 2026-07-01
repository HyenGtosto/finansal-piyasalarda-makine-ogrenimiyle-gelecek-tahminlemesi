"""Lightweight executable launcher for the Streamlit dashboard.

The Streamlit app itself stays in the project folder. This avoids packaging
TensorFlow and the full training stack into the launcher executable.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


if getattr(sys, "frozen", False):
    PROJECT_ROOT = Path(sys.executable).resolve().parent
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    app_path = PROJECT_ROOT / "app.py"
    subprocess.Popen(
        [
            "python",
            "-m",
            "streamlit",
            "run",
            str(app_path),
        ],
        cwd=PROJECT_ROOT,
    )


if __name__ == "__main__":
    main()

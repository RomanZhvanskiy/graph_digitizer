"""GUI launcher for Plot Digitizer.

This module is intentionally tiny. The actual GUI lives in frontend.py.
Run with:

    python -m plot_digitizer.app

or, after editable install:

    plot-digitizer-gui
"""
from __future__ import annotations

from plot_digitizer.frontend import run


def main() -> None:
    run()


if __name__ == "__main__":
    main()
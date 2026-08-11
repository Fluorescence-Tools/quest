"""Entry point for `python -m quest.gui`."""

from __future__ import annotations

import sys
from .app import start_gui

if __name__ == "__main__":
    sys.exit(start_gui())

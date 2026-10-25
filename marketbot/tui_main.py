#!/usr/bin/env python3
"""Launch the MarketBot terminal UI."""

import sys

from .tui import run_tui

if __name__ == "__main__":
    raise SystemExit(run_tui())

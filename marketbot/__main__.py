#!/usr/bin/env python3
"""Entry point: python -m marketbot"""

import sys

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())

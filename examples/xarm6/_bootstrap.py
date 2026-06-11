"""Ensure project root is on sys.path for example scripts."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
root = str(PROJECT_ROOT)
if root not in sys.path:
    sys.path.insert(0, root)

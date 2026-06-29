import sys
from pathlib import Path

for root in (Path(__file__).resolve().parents[1], Path(__file__).resolve().parents[2]):
    root = str(root)
    if root not in sys.path:
        sys.path.insert(0, root)

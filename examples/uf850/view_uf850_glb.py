import subprocess, sys
from pathlib import Path
import _bootstrap  # noqa: F401
if __name__ == "__main__":
  s = Path(__file__).resolve().parents[1] / "view_robot_glb.py"
  raise SystemExit(subprocess.call([sys.executable, str(s), "--robot", "uf850", *sys.argv[1:]]))

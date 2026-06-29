"""Verify Lite6 URDF in Genesis."""

import sys

import _bootstrap  # noqa: F401
from verify_robot import main

if __name__ == "__main__":
  sys.argv = [sys.argv[0], "--robot", "lite6", *sys.argv[1:]]
  main()

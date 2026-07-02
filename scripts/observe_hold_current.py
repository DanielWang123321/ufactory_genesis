#!/usr/bin/env python3
"""CLI wrapper for long-duration static hold current/torque observation."""

from ufactory.dynamics.observe import cli_observe_hold_current


if __name__ == "__main__":
    raise SystemExit(cli_observe_hold_current())

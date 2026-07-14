from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "gen_kinematics_params.py"


def test_malformed_robot_ip_is_rejected_before_sdk_connection():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "192168.1.65"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "invalid IPv4 address '192168.1.65'" in result.stderr
    assert "192.168.1.65" in result.stderr
    assert "connect serial failed" not in result.stderr


def test_ipv6_robot_address_is_rejected_before_sdk_connection():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "::1"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "IPv6 is not supported" in result.stderr
    assert "connect serial failed" not in result.stderr

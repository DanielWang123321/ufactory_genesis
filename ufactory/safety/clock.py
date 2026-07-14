"""Clock implementations for deterministic real-time scheduling tests."""

from __future__ import annotations

import time


class SystemClock:
    def monotonic_ns(self) -> int:
        return time.monotonic_ns()

    def wait_until_ns(self, deadline_ns: int) -> None:
        while True:
            remaining_ns = int(deadline_ns) - time.monotonic_ns()
            if remaining_ns <= 0:
                return
            time.sleep(remaining_ns / 1_000_000_000.0)

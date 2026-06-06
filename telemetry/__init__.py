"""Stream 3 territory — telemetry (`telemetry/`).

Weave instrumentation (§10) with a safe local/no-op fallback so the system runs
offline. Stream 3 (O8) adds the Weave Evaluation harness, leaderboard, and the
failure->fix->improve pitch trace.
"""

from __future__ import annotations

from .weave_ops import Telemetry, backend, init_telemetry, op

__all__ = ["op", "init_telemetry", "Telemetry", "backend"]

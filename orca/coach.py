"""Verbal coach + credit assignment (§6.4) — Stream 3 (O3, O4).

Reads the trace, reasons in natural language about credit (delegation vs
execution errors), produces ``performance_score`` / ``learning_signal`` (mostly
objective, §7.3) and the next behavior-card. Phase 0 is a no-op placeholder so
the interface exists for the fork.
"""

from __future__ import annotations

from contracts import EpisodeMetrics, EpisodeTrace

from .orca import Proposal


def coach(trace: EpisodeTrace, metrics: EpisodeMetrics) -> Proposal:
    """Phase 0 placeholder — no coaching (cards frozen, §6.6)."""
    return Proposal(notes="phase0-noop-coach")


__all__ = ["coach"]

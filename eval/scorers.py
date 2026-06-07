"""Custom eval scorers (§10) — Stream 3 (O8).

The five scorers the Weave Evaluation/leaderboard ranks conditions by, written as
**pure functions** of the objective episode data so they are testable and work
offline (the Weave wiring in ``weave_eval`` just adapts them):

  * ``frontier``            — DAG frontier value reached (the headline)
  * ``milestone_depth``     — index of the deepest milestone on the ladder
  * ``time_to_win``         — rounds to the dragon (``None`` if not won)
  * ``invalid_rate``        — failed actions / total
  * ``cooperation_events``  — handoffs + useful messages (co-op proxy)

None of these is Orca's own opinion — all are objective env stats (§6.4).
"""

from __future__ import annotations

from typing import Optional

from contracts import EpisodeMetrics, EpisodeTrace
from contracts.enums import Milestone

_LADDER = list(Milestone)


def frontier(metrics: EpisodeMetrics) -> float:
    return float(metrics.frontier_value)


def milestone_depth(metrics: EpisodeMetrics) -> int:
    """Index of the reached milestone on the shallow→deep ladder (0..len-1)."""
    return _LADDER.index(metrics.frontier_milestone)


def time_to_win(metrics: EpisodeMetrics) -> Optional[int]:
    """Rounds to the dragon, or ``None`` if the episode wasn't a win (§7.4)."""
    if not metrics.won:
        return None
    return metrics.milestone_timeline.get(Milestone.DRAGON_DEFEATED.value, metrics.n_rounds)


def invalid_rate(metrics: EpisodeMetrics) -> float:
    return float(metrics.invalid_rate)


def cooperation_events(metrics: EpisodeMetrics, trace: Optional[EpisodeTrace] = None) -> int:
    """Co-op proxy: handoffs + messages across the team.

    Messages are counted once: from the trace's bus when available (the real
    env), else from the per-agent ``messages_sent`` (the sim, whose trace carries
    no messages) — never both, to avoid double-counting the same messages.
    """
    handoffs = sum(st.handoffs_given + st.handoffs_received for st in metrics.agent_stats)
    if trace is not None and trace.messages:
        messages = len(trace.messages)
    else:
        messages = sum(st.messages_sent for st in metrics.agent_stats)
    return handoffs + messages


def score_episode(metrics: EpisodeMetrics, trace: Optional[EpisodeTrace] = None) -> dict[str, float]:
    """All scorers for one episode as a flat dict (the Weave row)."""
    ttw = time_to_win(metrics)
    return {
        "frontier": frontier(metrics),
        "milestone_depth": milestone_depth(metrics),
        "time_to_win": -1.0 if ttw is None else float(ttw),
        "invalid_rate": invalid_rate(metrics),
        "cooperation_events": float(cooperation_events(metrics, trace)),
    }


SCORER_NAMES = ["frontier", "milestone_depth", "time_to_win", "invalid_rate", "cooperation_events"]

__all__ = [
    "frontier",
    "milestone_depth",
    "time_to_win",
    "invalid_rate",
    "cooperation_events",
    "score_episode",
    "SCORER_NAMES",
]

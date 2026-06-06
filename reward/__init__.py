"""Stream 1 territory — reward (`reward/`).

Phase 0 ships the frontier ladder (§7.1) + penalties (§7.2) computed once per
episode. Stream 1 (E6) adds the speed phase (§7.4) post-win.
"""

from __future__ import annotations

from .dag import FRONTIER_LADDER, MILESTONE_VALUE, frontier_value, is_win
from .reward import DEFAULT_WEIGHTS, reward_computer

__all__ = [
    "reward_computer",
    "DEFAULT_WEIGHTS",
    "frontier_value",
    "is_win",
    "MILESTONE_VALUE",
    "FRONTIER_LADDER",
]

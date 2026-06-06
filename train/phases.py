"""Phasing controller (§6.6) — Stream 3 (O6).

  * Phase 0 (warmup): cards FROZEN; Orca learns delegation only (bandit).
  * Phase 1: Orca may edit cards / approve memory (accept-gated).
  * Phase 2 (post first win): enable the speed reward (§7.4).

Phase 0's foundation is fixed at PHASE_0; Stream 3 implements the transitions.
"""

from __future__ import annotations

from enum import IntEnum


class Phase(IntEnum):
    PHASE_0 = 0  # cards frozen, bandit-only
    PHASE_1 = 1  # coaching on, accept-gated
    PHASE_2 = 2  # speed reward (post first win)


def current_phase(episode_idx: int, phase0_length: int, first_win_seen: bool) -> Phase:
    """Phase from progress. Phase 0 always reports PHASE_0 (foundation)."""
    if first_win_seen:
        return Phase.PHASE_2
    if episode_idx >= phase0_length:
        return Phase.PHASE_1
    return Phase.PHASE_0


__all__ = ["Phase", "current_phase"]

"""Stream 3 territory — the orchestrator (`orca/`).

Phase 0 ships the no-op Orca (frozen cards, no learning). Stream 3 fills in the
delegation bandit (O2), scoring (O3), verbal coach (O4), and accept-gate (O5).
"""

from __future__ import annotations

from .bandit import EpsilonGreedyBandit
from .cards import DEFAULT_ROSTER, default_cards, make_default_card
from .gate import accept_gate
from .orca import NoOpOrca, OrcaConfig, Proposal

__all__ = [
    "NoOpOrca",
    "OrcaConfig",
    "Proposal",
    "EpsilonGreedyBandit",
    "accept_gate",
    "DEFAULT_ROSTER",
    "default_cards",
    "make_default_card",
]

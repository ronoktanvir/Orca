"""Stream 3 territory — the orchestrator (`orca/`).

The real Architecture-C2 manager: a delegation bandit (O2), objective scoring
(O3), a verbal coach with credit assignment (O4), and an accept/reject gate (O5),
phased by ``train/phases.py`` (O6). :class:`NoOpOrca` is kept as the offline /
Phase 0 fallback so ``main`` runs without an LLM.
"""

from __future__ import annotations

from .bandit import EpsilonGreedyBandit
from .cards import DEFAULT_ROSTER, default_cards, make_default_card
from .digest import AgentDigest, TraceDigest, build_digest
from .gate import AcceptGate, accept_gate
from .orca import NoOpOrca, Orca, OrcaConfig, Proposal
from .scoring import learning_signal, performance_score, score_agent, score_agents
from .situations import SITUATION_ARMS, roster_for_arm, strategic_directives

__all__ = [
    "NoOpOrca",
    "Orca",
    "OrcaConfig",
    "Proposal",
    "EpsilonGreedyBandit",
    "AcceptGate",
    "accept_gate",
    "DEFAULT_ROSTER",
    "default_cards",
    "make_default_card",
    "build_digest",
    "TraceDigest",
    "AgentDigest",
    "performance_score",
    "learning_signal",
    "score_agent",
    "score_agents",
    "SITUATION_ARMS",
    "roster_for_arm",
    "strategic_directives",
]

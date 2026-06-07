"""Contracts 6/7 and 7/7 — ``EpisodeTrace`` and ``EpisodeMetrics``.

``EpisodeTrace`` is the *raw* auditable event log of one episode (§6.1, §10):
every action + validity result, every message, the milestone timeline, and
(optionally) the coordinate-free observation snapshots. ``EpisodeMetrics`` is the
*computed* digest the bandit consumes (§7): the headline ``team_reward`` scalar,
penalties, and per-agent objective stats. Keeping them separate is what lets
Orca's scores stay advisory while the objective DAG frontier remains the
headline (§6.4 anti-circularity).
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from .action import Action
from .behavior_card import BehaviorCard
from .enums import Milestone, Role
from .message import Message


# --------------------------------------------------------------------------- #
# EpisodeTrace (contract 6/7) — raw event log
# --------------------------------------------------------------------------- #
class ActionRecord(BaseModel):
    """One resolved action and its outcome (§3.3 validity, §10 logging)."""

    round: int = Field(ge=0)
    agent_id: str
    action: Action
    valid: bool
    reason: Optional[str] = None  # invalid reason string, or a note
    result: dict[str, Any] = Field(default_factory=dict)  # e.g. {"gathered": {"wood": 3}}


class MilestoneEvent(BaseModel):
    """A DAG milestone hit at a particular round (the §6.1 timeline)."""

    milestone: Milestone
    round: int = Field(ge=0)


class ReasoningRecord(BaseModel):
    """One worker's natural-language reasoning for a single turn (§6.1, §6.4).

    This is the worker LLM's own "why I did this" — the same text that surfaces in
    Weave's ``worker_decision`` event. Recording it on the trace (already scrubbed
    through the coordinate-leak filter, §3.2) lets Orca's coach read *how* a worker
    reasoned, not just its action counts — without round-tripping the Weave API.
    """

    round: int = Field(ge=0)
    agent_id: str
    text: str = ""


class EpisodeTrace(BaseModel):
    """The full, replayable record of one episode."""

    episode_idx: int = Field(ge=0)
    seed: str
    n_rounds: int = Field(ge=0)
    agent_ids: list[str] = Field(default_factory=list)
    # Orca's config for this episode: chosen bandit arms + role assignment snapshot.
    config: dict[str, Any] = Field(default_factory=dict)
    behavior_cards: list[BehaviorCard] = Field(default_factory=list)
    action_records: list[ActionRecord] = Field(default_factory=list)
    messages: list[Message] = Field(default_factory=list)
    milestone_timeline: list[MilestoneEvent] = Field(default_factory=list)
    frontier_reached: Milestone = Milestone.START  # deepest milestone hit (pre-penalty)
    terminated_reason: str = "t_max"  # "win" | "frontier_target" | "t_max" | "all_failed"
    # Coordinate-free observation snapshots, kept for audit + the coord-leak test.
    observations: list[dict] = Field(default_factory=list)
    # Per-turn worker LLM reasoning (scrubbed) — lets Orca's coach read *how* each
    # worker reasoned, not just its action stats (§6.1/§6.4). Additive/optional.
    reasoning_log: list[ReasoningRecord] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# EpisodeMetrics (contract 7/7) — computed digest
# --------------------------------------------------------------------------- #
class AgentStats(BaseModel):
    """Per-agent objective stats (§6.1) + the two advisory dials (§7.3)."""

    agent_id: str
    role: Role
    actions_taken: int = 0
    invalid_actions: int = 0
    idle_rounds: int = 0
    deaths: int = 0
    items_gathered: dict[str, int] = Field(default_factory=dict)
    items_crafted: dict[str, int] = Field(default_factory=dict)
    handoffs_given: int = 0
    handoffs_received: int = 0
    messages_sent: int = 0
    # Advisory dials — filled by Orca's coach (no-op in Phase 0).
    performance_score: float = Field(default=0.0, ge=0.0, le=1.0)  # §7.3
    learning_signal: float = Field(default=0.0, ge=-1.0, le=1.0)  # §7.3


class EpisodeMetrics(BaseModel):
    """The computed episode digest. ``team_reward`` is the headline scalar (§7.1-7.2)."""

    episode_idx: int = Field(ge=0)
    seed: str
    frontier_milestone: Milestone
    frontier_value: float = Field(ge=0.0, le=1.0)  # raw ladder value (pre-penalty)
    team_reward: float = Field(ge=0.0)  # frontier_value - penalties, clipped >=0 (THE scalar)
    penalties: dict[str, float] = Field(default_factory=dict)  # {"deaths","invalid","idle"}
    invalid_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    idle_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    deaths: int = 0
    n_rounds: int = Field(ge=0)
    won: bool = False
    milestone_timeline: dict[str, int] = Field(default_factory=dict)  # milestone -> round
    agent_stats: list[AgentStats] = Field(default_factory=list)
    speed_bonus: float = Field(default=0.0, ge=0.0)  # post-win only (§7.4); 0 in Phase 0


__all__ = [
    "EpisodeTrace",
    "EpisodeMetrics",
    "ActionRecord",
    "MilestoneEvent",
    "ReasoningRecord",
    "AgentStats",
]

"""The orchestrator (§6). Phase 0 ships the **no-op Orca**.

Orca runs *between* episodes; it never acts inside one. In Phase 0 it returns the
frozen default cards/roster and learns nothing — exactly the F5 "no-op Orca". The
method surface (``choose_config`` → ``observe_outcome`` → ``coach`` → ``commit``)
matches the §8 run loop so Stream 3 can fill in the bandit, coach, scoring, and
accept-gate without changing the loop or any other stream's folder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from contracts import BehaviorCard, EpisodeMetrics, EpisodeTrace
from contracts.enums import Role

from .cards import default_cards


@dataclass
class OrcaConfig:
    """Orca's per-episode delegation config (read at episode start)."""

    roster: list[tuple[str, Role]]
    behavior_cards: dict[str, BehaviorCard]
    arms: dict[str, str] = field(default_factory=dict)  # chosen bandit arms (empty in Phase 0)

    def roles(self) -> dict[str, Role]:
        return {aid: role for aid, role in self.roster}


@dataclass
class Proposal:
    """An Orca coaching proposal (no-op in Phase 0)."""

    behavior_cards: dict[str, BehaviorCard] = field(default_factory=dict)
    memory_edits: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


class NoOpOrca:
    """A do-nothing orchestrator: frozen cards, no learning (Phase 0, §6.6)."""

    def __init__(self, roster: list[tuple[str, Role]]) -> None:
        self.roster = roster
        self._cards = default_cards(roster)

    def choose_config(self, history: list) -> OrcaConfig:
        """Return frozen defaults; no bandit choice in Phase 0."""
        return OrcaConfig(roster=self.roster, behavior_cards=dict(self._cards), arms={})

    def observe_outcome(self, config: OrcaConfig, metrics: EpisodeMetrics) -> None:
        """Where the bandit would update once per episode (§6.3). No-op in Phase 0."""
        return None

    def coach(self, trace: EpisodeTrace, metrics: EpisodeMetrics) -> Proposal:
        """Where verbal coaching + scoring would happen (§6.4). No-op in Phase 0."""
        return Proposal(notes="phase0-noop")

    def commit(self, proposal: Proposal) -> None:
        """Where accepted updates would persist (§6.5). No-op in Phase 0."""
        return None


__all__ = ["NoOpOrca", "OrcaConfig", "Proposal"]

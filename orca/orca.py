"""The orchestrator (§6).

Two implementations behind one method surface (``choose_config`` →
``observe_outcome`` → ``coach`` → ``commit``, matching the §8 run loop):

  * :class:`NoOpOrca` — the Phase 0 / offline fallback: frozen default cards, no
    learning. Kept so ``main`` runs fully offline (Green-main law).
  * :class:`Orca` — the real Architecture-C2 manager (Stream 3): a delegation
    **bandit** picks arms once per episode (§6.3), objective **scoring** fills the
    advisory dials (§7.3), a verbal **coach** rewrites cards between episodes
    (§6.4), and an accept-gate keeps only non-regressing edits (§6.5).

Orca runs *between* episodes; it never acts inside one. The headline reward is
always the objective ``EpisodeMetrics.team_reward`` — Orca's scores are advisory
and never summed into it (§6.4 anti-circularity).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from contracts import BehaviorCard, EpisodeMetrics, EpisodeTrace
from contracts.enums import Role

from .bandit import EpsilonGreedyBandit
from .cards import default_cards, make_default_card
from .scoring import score_agents
from .situations import (
    S1,
    SITUATION_ARMS,
    default_arms,
    roster_for_arm,
    strategic_directives,
)


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
    """An Orca coaching proposal between episodes (§6.4)."""

    behavior_cards: dict[str, BehaviorCard] = field(default_factory=dict)
    memory_edits: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    rationale: str = ""  # the coach's NL credit-assignment reasoning (logged, §6.4)
    scores: dict[str, dict[str, float]] = field(default_factory=dict)  # advisory dials, logged

    def is_empty(self) -> bool:
        return not self.behavior_cards and not self.memory_edits


# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
class Orca:
    """The real Architecture-C2 manager: bandit + scoring + verbal coach (§6).

    Parameters
    ----------
    roster:
        The base ``(agent_id, role)`` list — only the *ids* are fixed; the
        delegation bandit reassigns roles via the S1 arm each episode.
    llm:
        An ``LLMClient`` for the verbal coach (§6.4). ``None`` -> the coach falls
        back to a deterministic rule-based credit assignment (offline-safe).
    enable_bandit / enable_coach:
        Phase gates (§6.6). Phase 0 -> bandit only; Phase 1 -> coach on too.
    """

    def __init__(
        self,
        roster: list[tuple[str, Role]],
        *,
        llm: Any = None,
        epsilon: float = 0.2,
        seed: int = 0,
        optimistic: float = 1.0,
        enable_bandit: bool = True,
        enable_coach: bool = False,
        telemetry: Any = None,
    ) -> None:
        self.agent_ids = [aid for aid, _role in roster]
        self.bandit = EpsilonGreedyBandit(
            arms=SITUATION_ARMS, epsilon=epsilon, seed=seed, optimistic=optimistic
        )
        self.llm = llm
        self.enable_bandit = enable_bandit
        self.enable_coach = enable_coach
        self.telemetry = telemetry
        self._learning = True  # set False by freeze() for held-out eval
        # Committed coaching overrides, keyed by agent_id (empty until Phase 1).
        self._coached: dict[str, BehaviorCard] = {}
        self.last_arms: dict[str, str] = {}
        # Seeds the bandit has actually learned from — the held-out guard reads
        # this to prove B/C never entered training (Law 4 / §9).
        self.trained_seeds: set[str] = set()

    # -- delegation (bandit, §6.3) ------------------------------------------- #
    def _build_cards(
        self, roster: list[tuple[str, Role]], arms: dict[str, str]
    ) -> dict[str, BehaviorCard]:
        """Cards = (committed coaching | role default) + per-episode arm directives."""
        extra = strategic_directives(arms)
        cards: dict[str, BehaviorCard] = {}
        for aid, role in roster:
            override = self._coached.get(aid)
            base = (
                override.model_copy(update={"role": role})
                if override is not None
                else make_default_card(aid, role)
            )
            cards[aid] = base.model_copy(
                update={"directives": list(base.directives) + extra}
            )
        return cards

    def choose_config(self, history: Optional[list] = None, *, greedy: bool = False) -> OrcaConfig:
        """Pick one arm per situation (ε-greedy) -> roster + cards (§6.3).

        ``greedy=True`` uses the best learned arm with no exploration — for gate
        batches and held-out eval, which must measure the policy, not perturb it.
        """
        if not self.enable_bandit:
            arms = default_arms()
        elif greedy:
            arms = {sit: self.bandit.greedy(sit) for sit in SITUATION_ARMS}
        else:
            arms = {sit: self.bandit.choose(sit) for sit in SITUATION_ARMS}
        self.last_arms = arms
        roster = roster_for_arm(self.agent_ids, arms[S1])
        cards = self._build_cards(roster, arms)
        return OrcaConfig(roster=roster, behavior_cards=cards, arms=arms)

    def observe_outcome(self, config: OrcaConfig, metrics: EpisodeMetrics) -> None:
        """Credit every chosen arm with this episode's objective team_reward (§6.3)."""
        if not (self.enable_bandit and self._learning):
            return
        self.trained_seeds.add(metrics.seed)
        for sit, arm in config.arms.items():
            if sit in SITUATION_ARMS and arm in SITUATION_ARMS[sit]:
                self.bandit.update(sit, arm, metrics.team_reward)

    def bandit_values(self) -> dict[str, dict[str, float]]:
        """Current arm-value table — the data behind the learning curve (§6.3)."""
        return self.bandit.values()

    # -- verbal coaching (§6.4) ---------------------------------------------- #
    def coach(self, trace: EpisodeTrace, metrics: EpisodeMetrics) -> Proposal:
        """Read the digest, assign credit in NL, and propose the next cards (§6.4)."""
        if not self.enable_coach:
            return Proposal(notes="coach-disabled")
        from .coach import run_coach  # lazy: avoid import cycle

        return run_coach(
            trace,
            metrics,
            cards={aid: c for aid, c in self._coached.items()},
            llm=self.llm,
            telemetry=self.telemetry,
        )

    def commit(self, proposal: Proposal) -> None:
        """Persist accepted coaching overrides (§6.5)."""
        for aid, card in proposal.behavior_cards.items():
            self._coached[aid] = card

    # -- gate support / freezing (§6.5, §9) ---------------------------------- #
    def snapshot(self) -> dict[str, BehaviorCard]:
        """Deep copy of committed coaching — the rollback point for the gate."""
        return {aid: c.model_copy(deep=True) for aid, c in self._coached.items()}

    def restore(self, snap: dict[str, BehaviorCard]) -> None:
        self._coached = {aid: c.model_copy(deep=True) for aid, c in snap.items()}

    def freeze(self) -> None:
        """Stop learning & explore: greedy arms, frozen cards (held-out eval, §9)."""
        self._learning = False
        self.enable_coach = False
        self.bandit.epsilon = 0.0

    @staticmethod
    def objective_scores(metrics: EpisodeMetrics) -> EpisodeMetrics:
        """Fill the advisory dials from objective env stats only (§7.3).

        Called every episode (any phase) so the dials are always logged. Returns a
        copy of ``metrics`` with each ``AgentStats`` scored — ``team_reward`` is
        untouched (anti-circularity, §6.4)."""
        scored = score_agents(metrics.agent_stats)
        return metrics.model_copy(update={"agent_stats": scored})


__all__ = ["NoOpOrca", "Orca", "OrcaConfig", "Proposal"]

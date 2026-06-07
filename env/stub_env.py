"""The Phase 0 stub environment (F3).

A shallow but *real* end-to-end env: ~5 regions, the wood->stone->iron DAG slice,
move/scout/gather/craft/wait/report with validity rejection, coordinate-free
observations, and a max-frontier milestone tracker. Turn-based synchronous rounds
(§3.6): every free agent is queried for one macro-action, actions resolve, the
bus delivers messages at t+1 (§5.2), the clock advances.

This whole class is a STUB behind the frozen contracts (workflow §1). Stream 1
swaps it for the deep graph-on-plane world without other streams noticing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from contracts import (
    ActionRecord,
    BehaviorCard,
    Message,
    MilestoneEvent,
    Observation,
)
from contracts.enums import ActionName, Milestone, Role

from . import techtree
from .actions import resolve_action
from .observation import serialize_observation
from .rng import make_rng
from .seeds import make_world
from .world import AgentState

# Milestones are declared shallow->deep in the enum; declaration order == depth.
_MILESTONE_ORDER = list(Milestone)


def _depth(m: Milestone) -> int:
    return _MILESTONE_ORDER.index(m)


# E4 survival (§3.5). Hunger drains each round (faster when moving/fighting); at 0
# hunger, health drains; at 0 health the agent dies, drops non-equipped inventory,
# and respawns at the start region after a cost. Rates are deliberately gentle so a
# competent agent (the scripted oracle, ~hundreds of rounds) survives without eating
# — death is reachable for an agent that neglects food, and is forced in tests.
# Tunable upward later. All draws go through the seeded RNG (deterministic).
_HUNGER_DRAIN_IDLE = 0.0004
_HUNGER_DRAIN_ACTIVE = 0.0008  # moving / fighting costs more (§3.5)
_HEALTH_DRAIN_STARVING = 0.05  # per round while hunger is 0
_RESPAWN_COST = 5  # rounds dead before respawning at the start region
_ACTIVE_ACTIONS = {ActionName.MOVE, ActionName.FIGHT}
# Equipped gear survives death; raw resources + consumables are dropped (§3.5).
_EQUIPPED_KEEP = {
    "wooden_pickaxe", "stone_pickaxe", "iron_pickaxe", "diamond_pickaxe",
    "stone_sword", "iron_sword", "diamond_sword", "diamond_armor", "shield",
    "bucket", "flint_and_steel",
}


@dataclass
class StepResult:
    """What one synchronous round produced."""

    records: list[ActionRecord] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)  # newly posted this round
    milestone_events: list[MilestoneEvent] = field(default_factory=list)


class StubEnv:
    """Shallow, deterministic, coordinate-free Minecraft-lite env."""

    def __init__(
        self,
        seed: str = "A",
        episode_idx: int = 0,
        agents: Optional[list[tuple[str, Role]]] = None,
        *,
        t_max: int = 600,
        day_length: int = 100,
        message_window: int = 8,
        stop_at_milestone: Milestone | None = Milestone.IRON,
        behavior_cards: Optional[dict[str, BehaviorCard]] = None,
    ) -> None:
        self.seed = seed
        self.episode_idx = episode_idx
        self.t_max = t_max
        self.day_length = day_length
        self.message_window = message_window
        self.stop_at_milestone = stop_at_milestone
        self._agent_spec = agents or [("agent_1", Role.MINER)]
        self.behavior_cards = behavior_cards or {}

        self.round_idx = 0
        self.frontier = Milestone.START
        self.milestone_timeline: list[MilestoneEvent] = []
        self.all_messages: list[Message] = []
        self.all_records: list[ActionRecord] = []
        self._inbox: list[Message] = []  # delivered this round
        self._posted: list[Message] = []  # posted this round -> delivered next round
        self._terminated_reason: Optional[str] = None
        self.world = None  # set in reset()

    # ------------------------------------------------------------------ #
    @property
    def agent_ids(self) -> list[str]:
        return [aid for aid, _role in self._agent_spec]

    def reset(self) -> dict[str, Observation]:
        """(Re)build the world and place agents; return initial observations."""
        self.world = make_world(self.seed)
        for aid, role in self._agent_spec:
            self.world.add_agent(
                AgentState(agent_id=aid, role=role, region_id=self.world.start_region_id)
            )
        self.round_idx = 0
        self.frontier = Milestone.START
        self.milestone_timeline = [MilestoneEvent(milestone=Milestone.START, round=0)]
        self.all_messages.clear()
        self.all_records.clear()
        self._inbox.clear()
        self._posted.clear()
        self._terminated_reason = None
        return {aid: self.observe(aid) for aid in self.agent_ids}

    # ------------------------------------------------------------------ #
    def observe(self, agent_id: str) -> Observation:
        """The only observation path — coordinate-free (§3.2)."""
        card = self.behavior_cards.get(agent_id)
        assignment = card.assignment if card else ""
        recent = self._recent_for(agent_id)
        return serialize_observation(
            self.world,
            agent_id,
            round_idx=self.round_idx,
            day_length=self.day_length,
            assignment=assignment,
            frontier=self.frontier,
            recent_messages=recent,
        )

    def _recent_for(self, agent_id: str) -> list[Message]:
        relevant = [
            m for m in self._inbox if m.to in ("team", agent_id) and m.from_agent != agent_id
        ]
        return relevant[-self.message_window :]

    def drain_posted(self) -> list[Message]:
        """Hand this round's freshly posted messages to an external bus and clear
        the internal queue (§5.2).

        The full-team loop runs a single authoritative :class:`~bus.CommBus`; when
        it owns delivery, it drains the env's messages each round so the env never
        *also* delivers them (which would double-deliver and add a round of delay).
        The offline single-agent path never calls this, so its inline delivery is
        unchanged. ``all_messages`` keeps the verbatim record either way."""
        posted = self._posted
        self._posted = []
        return posted

    # ------------------------------------------------------------------ #
    def _survival_tick(self, agent: AgentState, action, rng) -> None:
        """Advance one round of hunger/health/death/respawn for one agent (§3.5)."""
        if not agent.alive:
            agent.busy_rounds -= 1  # respawn countdown
            if agent.busy_rounds <= 0:
                agent.alive = True
                agent.health = 1.0
                agent.hunger = 1.0
                agent.busy_rounds = 0
                agent.region_id = self.world.start_region_id
            return
        active = action is not None and getattr(action, "name", None) in _ACTIVE_ACTIONS
        base = _HUNGER_DRAIN_ACTIVE if active else _HUNGER_DRAIN_IDLE
        agent.hunger = max(0.0, agent.hunger - base * (0.8 + 0.4 * rng.random()))
        if agent.hunger <= 0.0:
            agent.health = max(0.0, agent.health - _HEALTH_DRAIN_STARVING)
            if agent.health <= 0.0:
                agent.alive = False
                agent.deaths += 1
                agent.busy_rounds = _RESPAWN_COST
                # Drop non-equipped inventory; keep equipped gear (§3.5).
                agent.inventory = {
                    k: v for k, v in agent.inventory.items() if k in _EQUIPPED_KEEP
                }

    # ------------------------------------------------------------------ #
    @property
    def done(self) -> bool:
        if self._terminated_reason is not None:
            return True
        if self.round_idx >= self.t_max:
            return True
        if self.stop_at_milestone is not None and _depth(self.frontier) >= _depth(
            self.stop_at_milestone
        ):
            return True
        return False

    @property
    def terminated_reason(self) -> str:
        if self._terminated_reason:
            return self._terminated_reason
        if self.frontier == Milestone.DRAGON_DEFEATED:
            return "win"
        if self.stop_at_milestone is not None and _depth(self.frontier) >= _depth(
            self.stop_at_milestone
        ):
            return "frontier_target"
        if self.round_idx >= self.t_max:
            return "t_max"
        return "running"

    # ------------------------------------------------------------------ #
    def step(self, actions: dict[str, "object"]) -> StepResult:
        """Resolve one synchronous round of actions for all free agents (§3.6)."""
        result = StepResult()
        # Deliver messages posted last round (t+1 delivery, §5.2).
        self._inbox = list(self._posted)
        self._posted = []

        for aid in self.agent_ids:
            agent = self.world.agents[aid]
            if not agent.alive or agent.status != "free":
                continue
            action = actions.get(aid)
            if action is None:
                continue
            rng = make_rng(self.seed, self.episode_idx, self.round_idx, aid)
            res = resolve_action(self.world, agent, action, rng, self.round_idx)
            result.records.append(res.record)
            self.all_records.append(res.record)
            if res.message is not None:
                self._posted.append(res.message)
                self.all_messages.append(res.message)
                result.messages.append(res.message)

        # Survival tick: hunger/health/death/respawn for every agent (§3.5, E4).
        for aid in self.agent_ids:
            agent = self.world.agents[aid]
            srng = make_rng(self.seed, self.episode_idx, self.round_idx, aid + ":survival")
            self._survival_tick(agent, actions.get(aid), srng)

        # Update the (monotonic) max team frontier and timeline.
        detected = techtree.detect_frontier(
            self.world.pooled_inventory(), self.world.world_milestones
        )
        if _depth(detected) > _depth(self.frontier):
            self.frontier = detected
            event = MilestoneEvent(milestone=detected, round=self.round_idx)
            self.milestone_timeline.append(event)
            result.milestone_events.append(event)

        self.round_idx += 1
        if self.done and self._terminated_reason is None:
            self._terminated_reason = self.terminated_reason
        return result


__all__ = ["StubEnv", "StepResult"]

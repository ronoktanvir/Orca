"""Trace digest (§6.1) — Stream 3 (O1).

Orca never reads the raw token stream. It reads a *compact, objective digest* of
one episode: the DAG frontier + milestone timeline, per-agent objective stats
(subtask progress / invalids / idle / deaths / handoffs / useful messages), and
the bottlenecks that explain a stall (longest dead-runs, repeated invalids,
starvation). This is the input to the verbal coach (§6.4) and to scoring (§7.3),
and it is what gets logged to Weave (§10).

The digest is derived **only** from the objective ``EpisodeTrace`` +
``EpisodeMetrics`` — never from Orca's own opinion — which is what keeps the
anti-circularity wall intact (§6.4): the coach reasons over facts, not vibes.
"""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, Field

from contracts import EpisodeMetrics, EpisodeTrace
from contracts.enums import ActionName

# Actions that *advance* the world (vs. idle/comms): used for stall detection.
_PRODUCTIVE = {
    ActionName.GATHER,
    ActionName.CRAFT,
    ActionName.SMELT,
    ActionName.PLACE,
    ActionName.FIGHT,
    ActionName.MOVE,
    ActionName.SCOUT,
    ActionName.EAT,
    ActionName.GIVE_ITEM,
}


class AgentDigest(BaseModel):
    """One worker's objective episode summary + its bottleneck flags."""

    agent_id: str
    role: str
    assignment: str = ""
    actions_taken: int = 0
    invalid_actions: int = 0
    invalid_rate: float = 0.0
    idle_rounds: int = 0
    deaths: int = 0
    items_gathered: dict[str, int] = Field(default_factory=dict)
    items_crafted: dict[str, int] = Field(default_factory=dict)
    handoffs_given: int = 0
    handoffs_received: int = 0
    useful_messages: int = 0
    # Bottlenecks (§6.1).
    longest_stall: int = 0  # longest run of consecutive non-productive rounds
    top_invalid: tuple[str, str, int] | None = None  # (action, reason, count)
    flags: list[str] = Field(default_factory=list)  # short human-readable notes

    # Advisory dials (filled by scoring, §7.3) — copied through for logging.
    performance_score: float = 0.0
    learning_signal: float = 0.0


class TraceDigest(BaseModel):
    """The compact, objective episode digest Orca consumes (§6.1)."""

    episode_idx: int
    seed: str
    n_rounds: int
    frontier_milestone: str
    frontier_value: float
    team_reward: float
    won: bool
    terminated_reason: str = ""
    invalid_rate: float = 0.0
    idle_fraction: float = 0.0
    deaths: int = 0
    milestone_timeline: list[tuple[str, int]] = Field(default_factory=list)
    arms: dict[str, str] = Field(default_factory=dict)
    agents: list[AgentDigest] = Field(default_factory=list)
    bottlenecks: list[str] = Field(default_factory=list)  # team-level explanations

    # ------------------------------------------------------------------ #
    def render(self) -> str:
        """A compact, readable text block for the coach prompt + logging (§6.4)."""
        lines: list[str] = []
        lines.append(
            f"EPISODE {self.episode_idx} · seed={self.seed} · "
            f"frontier={self.frontier_milestone} (value={self.frontier_value:.2f}) · "
            f"team_reward={self.team_reward:.3f} · rounds={self.n_rounds} · "
            f"won={self.won} · end={self.terminated_reason}"
        )
        if self.arms:
            arms = ", ".join(f"{s}={a}" for s, a in self.arms.items())
            lines.append(f"delegation arms: {arms}")
        if self.milestone_timeline:
            tl = ", ".join(f"{m}@{r}" for m, r in self.milestone_timeline)
            lines.append(f"milestones: {tl}")
        lines.append(
            f"team: invalid_rate={self.invalid_rate:.2f} "
            f"idle_fraction={self.idle_fraction:.2f} deaths={self.deaths}"
        )
        lines.append("agents:")
        for a in self.agents:
            note = f" [{'; '.join(a.flags)}]" if a.flags else ""
            gathered = ",".join(f"{k}:{v}" for k, v in sorted(a.items_gathered.items())) or "-"
            crafted = ",".join(f"{k}:{v}" for k, v in sorted(a.items_crafted.items())) or "-"
            lines.append(
                f"  - {a.agent_id} ({a.role}): acts={a.actions_taken} "
                f"invalid={a.invalid_actions} idle={a.idle_rounds} deaths={a.deaths} "
                f"stall={a.longest_stall} msgs={a.useful_messages} "
                f"hand={a.handoffs_given}/{a.handoffs_received} "
                f"gathered=[{gathered}] crafted=[{crafted}]{note}"
            )
            if a.assignment:
                lines.append(f"      assigned: {a.assignment}")
        if self.bottlenecks:
            lines.append("bottlenecks:")
            for b in self.bottlenecks:
                lines.append(f"  * {b}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
def _agent_bottlenecks(trace: EpisodeTrace, agent_id: str) -> tuple[int, tuple[str, str, int] | None]:
    """Longest non-productive run + most-repeated invalid for one agent."""
    longest = run = 0
    invalid_counter: Counter[tuple[str, str]] = Counter()
    for rec in trace.action_records:
        if rec.agent_id != agent_id:
            continue
        productive = rec.valid and rec.action.name in _PRODUCTIVE
        if productive:
            run = 0
        else:
            run += 1
            longest = max(longest, run)
        if not rec.valid:
            invalid_counter[(rec.action.name.value, (rec.reason or "")[:60])] += 1
    top = None
    if invalid_counter:
        (act, reason), cnt = invalid_counter.most_common(1)[0]
        top = (act, reason, cnt)
    return longest, top


def _handoffs_received(trace: EpisodeTrace) -> Counter[str]:
    """Count HANDOFF/give-item messages by recipient (best-effort from the bus)."""
    received: Counter[str] = Counter()
    for m in trace.messages:
        to = getattr(m, "to", None)
        if to and to != "team":
            received[to] += 1
    return received


def build_digest(trace: EpisodeTrace, metrics: EpisodeMetrics) -> TraceDigest:
    """Compress an episode into the objective digest Orca reads (§6.1)."""
    assignments = {c.agent_id: c.assignment for c in trace.behavior_cards}
    handoffs_in = _handoffs_received(trace)
    msgs_by_agent: Counter[str] = Counter(
        getattr(m, "from_agent", "") for m in trace.messages
    )

    agents: list[AgentDigest] = []
    for st in metrics.agent_stats:
        longest_stall, top_invalid = _agent_bottlenecks(trace, st.agent_id)
        inv_rate = (st.invalid_actions / st.actions_taken) if st.actions_taken else 0.0
        flags: list[str] = []
        if st.deaths:
            flags.append(f"died x{st.deaths}")
        if longest_stall >= max(3, metrics.n_rounds // 3) and metrics.n_rounds:
            flags.append(f"stalled {longest_stall} rounds")
        if inv_rate >= 0.25 and st.actions_taken >= 4:
            flags.append(f"high invalid {inv_rate:.0%}")
        if st.actions_taken and st.idle_rounds / st.actions_taken >= 0.4:
            flags.append("mostly idle")
        if top_invalid and top_invalid[2] >= 3:
            flags.append(f"repeated invalid '{top_invalid[0]}' x{top_invalid[2]}")
        agents.append(
            AgentDigest(
                agent_id=st.agent_id,
                role=st.role.value,
                assignment=assignments.get(st.agent_id, ""),
                actions_taken=st.actions_taken,
                invalid_actions=st.invalid_actions,
                invalid_rate=inv_rate,
                idle_rounds=st.idle_rounds,
                deaths=st.deaths,
                items_gathered=dict(st.items_gathered),
                items_crafted=dict(st.items_crafted),
                handoffs_given=st.handoffs_given,
                handoffs_received=int(handoffs_in.get(st.agent_id, 0)),
                useful_messages=max(st.messages_sent, int(msgs_by_agent.get(st.agent_id, 0))),
                longest_stall=longest_stall,
                top_invalid=top_invalid,
                flags=flags,
                performance_score=st.performance_score,
                learning_signal=st.learning_signal,
            )
        )

    # Team-level bottleneck explanations (the "why did we stall here" line).
    bottlenecks: list[str] = []
    timeline = sorted(metrics.milestone_timeline.items(), key=lambda kv: kv[1])
    if not metrics.won and timeline:
        last_m, last_r = timeline[-1]
        gap = metrics.n_rounds - last_r
        if gap >= max(3, metrics.n_rounds // 3):
            bottlenecks.append(
                f"frontier stuck at '{last_m}' for the last {gap} rounds "
                f"(no milestone after round {last_r})"
            )
    if metrics.invalid_rate >= 0.2:
        bottlenecks.append(f"team invalid-rate high at {metrics.invalid_rate:.0%}")
    if metrics.idle_fraction >= 0.3:
        bottlenecks.append(f"team idle fraction high at {metrics.idle_fraction:.0%}")
    for a in agents:
        if a.top_invalid and a.top_invalid[2] >= 3:
            bottlenecks.append(
                f"{a.agent_id} kept failing '{a.top_invalid[0]}'"
                + (f" ({a.top_invalid[1]})" if a.top_invalid[1] else "")
                + f" x{a.top_invalid[2]}"
            )

    return TraceDigest(
        episode_idx=metrics.episode_idx,
        seed=metrics.seed,
        n_rounds=metrics.n_rounds,
        frontier_milestone=metrics.frontier_milestone.value,
        frontier_value=metrics.frontier_value,
        team_reward=metrics.team_reward,
        won=metrics.won,
        terminated_reason=getattr(trace, "terminated_reason", ""),
        invalid_rate=metrics.invalid_rate,
        idle_fraction=metrics.idle_fraction,
        deaths=metrics.deaths,
        milestone_timeline=timeline,
        arms=dict(trace.config.get("arms", {})) if isinstance(trace.config, dict) else {},
        agents=agents,
        bottlenecks=bottlenecks,
    )


__all__ = ["TraceDigest", "AgentDigest", "build_digest"]

"""Reward computer — turns an ``EpisodeTrace`` into ``EpisodeMetrics`` (§7).

The headline scalar is ``team_reward = frontier_value - penalties`` (clipped to
>= 0), computed **once per episode** (§7.1-7.2). Penalties are small quality
nudges (death/invalid/idle) that never overwhelm the progression signal. The two
advisory dials (performance_score / learning_signal) are left at 0 here — they
are Orca's job (§7.3), and are deliberately *not* summed into team_reward
(§6.4 anti-circularity). The speed bonus stays 0 until post-win Phase 2 (§7.4).
"""

from __future__ import annotations

from contracts import AgentStats, EpisodeMetrics, EpisodeTrace
from contracts.enums import ActionName, Role

from .dag import frontier_value, is_win

# Default penalty weights (§7.2 / §15): kept << a frontier step.
DEFAULT_WEIGHTS = {"deaths": 0.02, "invalid": 0.05, "idle": 0.05}


def _per_agent_stats(
    trace: EpisodeTrace, agent_roles: dict[str, Role]
) -> tuple[list[AgentStats], int, int, int, int]:
    """Build per-agent stats; return (stats, total_actions, total_invalid, total_idle, total_deaths)."""
    by_agent: dict[str, AgentStats] = {
        aid: AgentStats(agent_id=aid, role=agent_roles.get(aid, Role.MINER))
        for aid in trace.agent_ids
    }
    total_actions = total_invalid = total_idle = total_deaths = 0

    for rec in trace.action_records:
        st = by_agent.setdefault(
            rec.agent_id, AgentStats(agent_id=rec.agent_id, role=Role.MINER)
        )
        st.actions_taken += 1
        total_actions += 1
        if not rec.valid:
            st.invalid_actions += 1
            total_invalid += 1
        if rec.action.name == ActionName.WAIT:
            st.idle_rounds += 1
            total_idle += 1
        if rec.action.name in (ActionName.REPORT, ActionName.REQUEST_HELP) and rec.valid:
            st.messages_sent += 1
        if rec.action.name == ActionName.GIVE_ITEM and rec.valid:
            st.handoffs_given += 1
        for item, qty in rec.result.get("gathered", {}).items():
            st.items_gathered[item] = st.items_gathered.get(item, 0) + qty
        for item, qty in rec.result.get("crafted", {}).items():
            st.items_crafted[item] = st.items_crafted.get(item, 0) + qty

    return list(by_agent.values()), total_actions, total_invalid, total_idle, total_deaths


def reward_computer(
    trace: EpisodeTrace,
    *,
    agent_roles: dict[str, Role] | None = None,
    weights: dict[str, float] | None = None,
    baseline_steps: int | None = None,  # Phase 2 only (§7.4); unused in Phase 0
) -> EpisodeMetrics:
    """Compute the episode digest from a trace."""
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    agent_roles = agent_roles or {}

    stats, n_actions, n_invalid, n_idle, n_deaths = _per_agent_stats(trace, agent_roles)

    milestone = trace.frontier_reached
    base = frontier_value(milestone)

    invalid_rate = n_invalid / n_actions if n_actions else 0.0
    idle_fraction = n_idle / n_actions if n_actions else 0.0

    penalties = {
        "deaths": weights["deaths"] * n_deaths,
        "invalid": weights["invalid"] * invalid_rate,
        "idle": weights["idle"] * idle_fraction,
    }
    team_reward = max(0.0, base - sum(penalties.values()))

    return EpisodeMetrics(
        episode_idx=trace.episode_idx,
        seed=trace.seed,
        frontier_milestone=milestone,
        frontier_value=base,
        team_reward=team_reward,
        penalties=penalties,
        invalid_rate=invalid_rate,
        idle_fraction=idle_fraction,
        deaths=n_deaths,
        n_rounds=trace.n_rounds,
        won=is_win(milestone),
        milestone_timeline={ev.milestone.value: ev.round for ev in trace.milestone_timeline},
        agent_stats=stats,
        speed_bonus=0.0,  # Phase 0/1: never (§7.4)
    )


__all__ = ["reward_computer", "DEFAULT_WEIGHTS"]

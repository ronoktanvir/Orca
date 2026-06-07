"""Per-agent scoring (§7.3) — Stream 3 (O3).

Two advisory dials, computed **mostly from objective env stats** so Orca cannot
inflate its own headline (§6.4 anti-circularity). Neither is ever summed into
``team_reward`` (that stays the objective DAG frontier, §7.1).

  * ``performance_score`` ∈ [0,1] — subtask completion + frontier contribution +
    low invalid/idle, *lightly* Orca's opinion. Used in feedback + logged.
  * ``learning_signal``  ∈ [−1,1] — Orca's "adopt this lesson?" dial that scales
    an agent's memory-edit magnitude (§4.5): +1 add/strengthen, ~0 leave alone,
    −1 weaken/remove. The objective default is a non-negative "how much
    correction is warranted" signal; the verbal coach (O4) may flip it negative
    to weaken a rule it judges harmful.

Everything here is a pure function of ``AgentStats`` — same stats in, same
scores out — which is exactly what the §7.3 test pins down.
"""

from __future__ import annotations

from dataclasses import dataclass

from contracts import AgentStats

# How many gathered+crafted items count as "fully productive" output (saturates
# the output term). Small so the shallow iron run can reach it.
_OUTPUT_TARGET = 8.0


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass(frozen=True)
class ScoreWeights:
    """Tunable weights for ``performance_score`` (§7.3 / §15)."""

    valid: float = 0.40  # fraction of actions that were valid
    active: float = 0.25  # fraction of actions that were not idle
    output: float = 0.35  # gathered + crafted volume (subtask completion proxy)
    death_penalty: float = 0.25  # per death
    opinion_blend: float = 0.15  # weight on Orca's [0,1] opinion when supplied


DEFAULT_WEIGHTS = ScoreWeights()


def performance_score(
    stats: AgentStats,
    *,
    opinion: float | None = None,
    weights: ScoreWeights = DEFAULT_WEIGHTS,
) -> float:
    """Objective [0,1] competence score for one agent (§7.3).

    ``opinion`` (Orca's [0,1] read) is blended in *lightly* when provided; with
    ``None`` the score is purely objective — the anti-circularity guarantee.
    """
    n = stats.actions_taken
    if n <= 0:
        return 0.0  # an agent that never acted contributed nothing
    valid_frac = (n - stats.invalid_actions) / n
    active_frac = (n - stats.idle_rounds) / n
    produced = sum(stats.items_gathered.values()) + sum(stats.items_crafted.values())
    output = min(1.0, produced / _OUTPUT_TARGET)

    objective = (
        weights.valid * valid_frac
        + weights.active * active_frac
        + weights.output * output
    )
    objective -= weights.death_penalty * stats.deaths
    objective = _clip(objective, 0.0, 1.0)

    if opinion is None:
        return round(objective, 4)
    blended = (1.0 - weights.opinion_blend) * objective + weights.opinion_blend * _clip(
        opinion, 0.0, 1.0
    )
    return round(_clip(blended, 0.0, 1.0), 4)


def learning_signal(stats: AgentStats, *, override: float | None = None) -> float:
    """Objective [−1,1] "how much memory correction is warranted" dial (§7.3).

    Default: non-negative, proportional to dysfunction (invalid/idle/deaths) — a
    struggling agent warrants a stronger corrective memory edit. The verbal coach
    (O4) supplies ``override`` to set the sign (e.g. −1 to weaken a bad rule).
    """
    if override is not None:
        return round(_clip(override, -1.0, 1.0), 4)
    n = stats.actions_taken
    if n <= 0:
        return 0.0
    invalid_rate = stats.invalid_actions / n
    idle_fraction = stats.idle_rounds / n
    dysfunction = invalid_rate + idle_fraction + 0.3 * stats.deaths
    return round(_clip(dysfunction, 0.0, 1.0), 4)


def score_agent(
    stats: AgentStats,
    *,
    opinion: float | None = None,
    signal_override: float | None = None,
    weights: ScoreWeights = DEFAULT_WEIGHTS,
) -> tuple[float, float]:
    """Return ``(performance_score, learning_signal)`` for one agent."""
    return (
        performance_score(stats, opinion=opinion, weights=weights),
        learning_signal(stats, override=signal_override),
    )


def score_agents(
    agent_stats: list[AgentStats],
    *,
    opinions: dict[str, float] | None = None,
    signal_overrides: dict[str, float] | None = None,
    weights: ScoreWeights = DEFAULT_WEIGHTS,
) -> list[AgentStats]:
    """Return copies of ``agent_stats`` with the two dials filled (§7.3).

    Pure: reads only ``AgentStats`` (+ optional Orca opinions); never the trace,
    LLM, or RNG. ``opinions``/``signal_overrides`` are keyed by ``agent_id``.
    """
    opinions = opinions or {}
    signal_overrides = signal_overrides or {}
    out: list[AgentStats] = []
    for st in agent_stats:
        perf, sig = score_agent(
            st,
            opinion=opinions.get(st.agent_id),
            signal_override=signal_overrides.get(st.agent_id),
            weights=weights,
        )
        out.append(st.model_copy(update={"performance_score": perf, "learning_signal": sig}))
    return out


__all__ = [
    "ScoreWeights",
    "DEFAULT_WEIGHTS",
    "performance_score",
    "learning_signal",
    "score_agent",
    "score_agents",
]

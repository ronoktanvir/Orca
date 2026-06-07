"""Weave Evaluation + leaderboard + the pitch trace (§10) — Stream 3 (O8).

Three things the sponsor axis asks for:

  1. **Custom scorers** (``eval.scorers``) — frontier / milestone / time-to-win /
     invalid-rate / cooperation-events — aggregated per condition into a
     **leaderboard** that ranks static vs comms vs Full C2.
  2. **A Weave Evaluation/comparison** — logged as structured per-condition score
     tables (which nest in the Weave trace) plus a best-effort call to the formal
     ``weave.Evaluation`` API when Weave is installed + authenticated.
  3. **The pitch trace** — one nested trace where Orca *spots a failure → edits a
     card → the next run clears the bottleneck*. Each step is an ``@op`` so it
     nests; it runs offline (the calibrated outcome model makes the improvement
     real + reproducible) and lights up in Weave when telemetry is live.

All of it degrades gracefully offline (Green-main): no Weave, no creds → the
leaderboard + pitch result are still computed and returned.
"""

from __future__ import annotations

from statistics import mean
from typing import Any, Optional

from config import OrcaSettings, load_config
from contracts import EpisodeMetrics, EpisodeTrace
from telemetry import op

from .harness import (
    COMMS_SPEC,
    FULL_C2_SPEC,
    STATIC_SPEC,
    ConditionSpec,
    Runner,
    SimRunner,
    make_orca,
    train_full_c2,
)
from .outcome_model import FULL_C2
from .scorers import SCORER_NAMES, score_episode


# --------------------------------------------------------------------------- #
# Leaderboard.
# --------------------------------------------------------------------------- #
def build_leaderboard(
    by_condition: dict[str, list[tuple[EpisodeTrace, EpisodeMetrics]]],
) -> dict[str, dict[str, float]]:
    """Mean of each scorer per condition — the comparison/leaderboard table (§10)."""
    table: dict[str, dict[str, float]] = {}
    for cond, episodes in by_condition.items():
        rows = [score_episode(m, t) for t, m in episodes]
        if not rows:
            continue
        table[cond] = {name: round(mean(r[name] for r in rows), 4) for name in SCORER_NAMES}
        table[cond]["n"] = len(rows)
    return table


def _run_condition(
    orca,
    runner: Runner,
    spec: ConditionSpec,
    seeds: list[str],
    reps: int,
) -> list[tuple[EpisodeTrace, EpisodeMetrics]]:
    out: list[tuple[EpisodeTrace, EpisodeMetrics]] = []
    idx = 0
    for _rep in range(reps):
        for s in seeds:
            config = orca.choose_config(greedy=True)
            trace, metrics = runner(
                config, s, condition=spec.sim_condition, episode_idx=idx, memory=spec.memory, gate_on=spec.gate
            )
            out.append((trace, metrics))
            idx += 1
    return out


def evaluate_conditions(
    settings: Optional[OrcaSettings] = None,
    *,
    runner: Optional[Runner] = None,
    seeds: Optional[list[str]] = None,
    n_train: int = 30,
    reps: int = 6,
) -> dict[str, list[tuple[EpisodeTrace, EpisodeMetrics]]]:
    """Train Full C2, freeze all three conditions, eval on ``seeds`` (default held-out)."""
    settings = settings or load_config()
    runner = runner or SimRunner()
    seeds = seeds if seeds is not None else list(settings.seeds.heldout)

    by_cond: dict[str, list[tuple[EpisodeTrace, EpisodeMetrics]]] = {}
    tr = train_full_c2(FULL_C2_SPEC, settings, runner, list(settings.seeds.train), n_train)
    by_cond[FULL_C2_SPEC.name] = _run_condition(tr.orca, runner, FULL_C2_SPEC, seeds, reps)
    for spec in (STATIC_SPEC, COMMS_SPEC):
        orca = make_orca(spec, settings)
        orca.freeze()
        by_cond[spec.name] = _run_condition(orca, runner, spec, seeds, reps)
    return by_cond


@op
def run_weave_evaluation(
    settings: Optional[OrcaSettings] = None,
    *,
    runner: Optional[Runner] = None,
    telemetry: Any = None,
    n_train: int = 30,
    reps: int = 6,
) -> dict[str, dict[str, float]]:
    """Score every condition and emit the leaderboard (§10).

    Logs the per-condition scorer table to telemetry (nests in Weave) and makes a
    best-effort call to the formal ``weave.Evaluation`` API; returns the
    leaderboard either way.
    """
    settings = settings or load_config()
    by_cond = evaluate_conditions(settings, runner=runner, n_train=n_train, reps=reps)
    leaderboard = build_leaderboard(by_cond)

    if telemetry is not None:
        try:
            telemetry.log_event("weave_leaderboard", {"scorers": SCORER_NAMES, "table": leaderboard})
        except Exception:
            pass

    _try_formal_weave_evaluation(by_cond)
    return leaderboard


def _try_formal_weave_evaluation(by_condition: dict[str, list[tuple[EpisodeTrace, EpisodeMetrics]]]) -> bool:
    """Best-effort: register a ``weave.Evaluation`` per condition. Never raises."""
    try:
        import weave  # type: ignore
    except Exception:
        return False
    try:
        for cond, episodes in by_condition.items():
            dataset = [{"id": i, "seed": m.seed} for i, (_t, m) in enumerate(episodes)]
            scores = [score_episode(m, t) for t, m in episodes]

            @weave.op()  # type: ignore
            def model(id: int, seed: str, _scores=scores):  # noqa: A002
                return _scores[id]

            evaluation = weave.Evaluation(dataset=dataset, scorers=[])  # comparison view
            try:
                import asyncio

                asyncio.run(evaluation.evaluate(model))
            except Exception:
                pass
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# The pitch trace — failure → fix → improve, as one nested @op tree.
# --------------------------------------------------------------------------- #
@op
def _pitch_episode(orca, runner, seed, *, episode_idx, label):
    """One episode of the pitch trace (its own @op so the tree nests in Weave)."""
    config = orca.choose_config(greedy=True)
    trace, metrics = runner(
        config, seed, condition=FULL_C2, episode_idx=episode_idx, coaching_active=True, memory=True, gate_on=True
    )
    return config, trace, metrics


def _card_diff(before, after) -> list[str]:
    b = set(before.directives) if before else set()
    a = set(after.directives) if after else set()
    return sorted(a - b)


@op
def capture_pitch_trace(
    settings: Optional[OrcaSettings] = None,
    *,
    runner: Optional[Runner] = None,
    seed: Optional[str] = None,
    telemetry: Any = None,
) -> dict[str, Any]:
    """The demo trace: Orca spots a failure, edits a card, the next run improves (§10).

    Returns a structured before/after summary with the card diff + the coach's
    reasoning. Each load-bearing step (episode → coach → episode) is an ``@op``,
    so in Weave this is a single auditable nested trace.
    """
    settings = settings or load_config()
    runner = runner or SimRunner()
    seed = seed or (settings.seeds.train[0] if settings.seeds.train else "A")

    orca = make_orca(FULL_C2_SPEC, settings)
    orca.enable_coach = True  # force Phase 1 for the demonstration
    orca.telemetry = telemetry

    # BEFORE — the iron-miner repeatedly fails (missing the stone-pickaxe step).
    cfg_before, trace_before, m_before = _pitch_episode(
        orca, runner, seed, episode_idx=0, label="before-failure"
    )
    miner = trace_before.agent_ids[1] if len(trace_before.agent_ids) > 1 else trace_before.agent_ids[0]
    card_before = cfg_before.behavior_cards.get(miner)

    # Orca reads the digest, assigns credit, rewrites the card (its own @op).
    proposal = orca.coach(trace_before, m_before)
    orca.commit(proposal)
    card_after = proposal.behavior_cards.get(miner) or orca._coached.get(miner)

    # AFTER — same seed, the corrected card clears the bottleneck.
    _cfg_after, _trace_after, m_after = _pitch_episode(
        orca, runner, seed, episode_idx=1, label="after-fix"
    )

    result = {
        "seed": seed,
        "bottleneck_agent": miner,
        "before": {
            "frontier": m_before.frontier_value,
            "milestone": m_before.frontier_milestone.value,
            "invalid_rate": m_before.invalid_rate,
        },
        "after": {
            "frontier": m_after.frontier_value,
            "milestone": m_after.frontier_milestone.value,
            "invalid_rate": m_after.invalid_rate,
        },
        "improved": (m_after.frontier_value > m_before.frontier_value)
        and (m_after.invalid_rate < m_before.invalid_rate),
        "coach_rationale": proposal.rationale,
        "card_diff_added_directives": _card_diff(card_before, card_after),
    }
    if telemetry is not None:
        try:
            telemetry.log_event("pitch_trace", result)
        except Exception:
            pass
    return result


__all__ = [
    "build_leaderboard",
    "evaluate_conditions",
    "run_weave_evaluation",
    "capture_pitch_trace",
]

"""Tests for the eval harness (§9) + Weave scorers/pitch trace (§10).

These run entirely on the calibrated outcome model (no env, no LLM, no network),
and pin the headline claims: the learning curve rises, Full C2 ≥ baselines on
held-out, the ablations show each component's value, and held-out {B,C} never
enter training.
"""

from __future__ import annotations

from config import load_config
from eval.harness import (
    ABL_NO_COACH,
    ABL_NO_GATE,
    FULL_C2_SPEC,
    SimRunner,
    eval_batch,
    run_ablations,
    train_full_c2,
)
from eval.records import HELDOUT, TRAIN, summarize
from eval.scorers import SCORER_NAMES, score_episode
from eval.transfer import run_transfer, transfer_verdict
from eval.weave_eval import build_leaderboard, capture_pitch_trace, evaluate_conditions

from tests.fixtures import sample_episode_metrics, sample_episode_trace

_N_TRAIN = 24
_REPS = 4


# --------------------------------------------------------------------------- #
# Transfer — the money plot (§9)
# --------------------------------------------------------------------------- #
def test_transfer_full_c2_beats_baselines_on_heldout():
    tr = run_transfer(n_train=_N_TRAIN, eval_reps=_REPS)
    verdict = transfer_verdict(tr.records)
    assert verdict["full_c2_wins"] is True
    assert verdict["full_c2_heldout"] > verdict["static_heldout"]
    assert verdict["full_c2_heldout"] > verdict["comms_heldout"]


def test_transfer_reports_variance_over_multiple_episodes():
    tr = run_transfer(n_train=_N_TRAIN, eval_reps=_REPS)
    stats = summarize(tr.records)
    # every (condition, split) cell has more than one episode (no single anecdote)
    for (_cond, _split), st in stats.items():
        assert st.n > 1


# --------------------------------------------------------------------------- #
# Held-out guard (Law 4) — B/C never enter training
# --------------------------------------------------------------------------- #
def test_heldout_seeds_never_trained_on():
    settings = load_config()
    runner = SimRunner()
    train_seeds = list(settings.seeds.train)
    heldout = list(settings.seeds.heldout)

    tr = train_full_c2(FULL_C2_SPEC, settings, runner, train_seeds, _N_TRAIN)
    assert tr.orca.trained_seeds.issubset(set(train_seeds))
    assert tr.orca.trained_seeds.isdisjoint(set(heldout))

    # evaluating the frozen policy on held-out must not add to training history
    before = set(tr.orca.trained_seeds)
    eval_batch(tr.orca, runner, heldout, FULL_C2_SPEC, HELDOUT, reps=_REPS)
    assert tr.orca.trained_seeds == before
    assert tr.orca.trained_seeds.isdisjoint(set(heldout))


# --------------------------------------------------------------------------- #
# Learning curve (§9)
# --------------------------------------------------------------------------- #
def test_learning_curve_rises():
    tr = train_full_c2(FULL_C2_SPEC, load_config(), SimRunner(), ["A", "T2", "T3"], 40)
    fr = [r.frontier_value for r in tr.learning]
    first = sum(fr[:8]) / 8
    last = sum(fr[-8:]) / 8
    assert last > first + 0.05  # meaningfully higher by the end


def test_bandit_value_snapshots_recorded():
    tr = train_full_c2(FULL_C2_SPEC, load_config(), SimRunner(), ["A", "T2", "T3"], 20)
    assert len(tr.value_snapshots) == 20
    # by the end the bandit has a non-trivial best S1 arm
    from orca.situations import S1

    last = tr.value_snapshots[-1][S1]
    assert max(last.values()) > min(last.values())


# --------------------------------------------------------------------------- #
# Ablations (§9) — each component adds value; the gate is the key one
# --------------------------------------------------------------------------- #
def test_ablations_show_component_value():
    records = run_ablations(n_train=_N_TRAIN, eval_reps=_REPS)
    stats = summarize(records)
    full = stats[("full_c2", HELDOUT)].mean
    no_gate = stats[("no_gate", HELDOUT)].mean
    no_coach = stats[("no_coaching", HELDOUT)].mean
    assert full > no_coach  # coaching clears the bottleneck
    assert full > no_gate  # the gate filters noisy edits
    assert full >= stats[("no_memory", HELDOUT)].mean  # memory helps (or ties)


# --------------------------------------------------------------------------- #
# Weave scorers + leaderboard (§10)
# --------------------------------------------------------------------------- #
def test_scorers_cover_all_five():
    row = score_episode(sample_episode_metrics(), sample_episode_trace())
    assert set(row.keys()) == set(SCORER_NAMES)
    assert row["frontier"] >= 0.0
    assert row["invalid_rate"] >= 0.0


def test_leaderboard_ranks_full_c2_on_top():
    by_cond = evaluate_conditions(n_train=_N_TRAIN, reps=_REPS)
    lb = build_leaderboard(by_cond)
    assert set(lb.keys()) == {"full_c2", "static", "comms"}
    assert lb["full_c2"]["frontier"] > lb["comms"]["frontier"]
    assert lb["full_c2"]["frontier"] > lb["static"]["frontier"]
    # Full C2 also has the lowest invalid-rate (coaching fixed the miner)
    assert lb["full_c2"]["invalid_rate"] < lb["static"]["invalid_rate"]


# --------------------------------------------------------------------------- #
# Pitch trace (§10) — failure -> fix -> improve
# --------------------------------------------------------------------------- #
def test_pitch_trace_shows_failure_fix_improve():
    result = capture_pitch_trace(seed="A")
    assert result["improved"] is True
    assert result["after"]["frontier"] > result["before"]["frontier"]
    assert result["after"]["invalid_rate"] < result["before"]["invalid_rate"]
    assert result["card_diff_added_directives"]  # a directive was actually added
    assert "execution" in result["coach_rationale"] or result["coach_rationale"]


# --------------------------------------------------------------------------- #
# Sim runner sanity — no LLM needed
# --------------------------------------------------------------------------- #
def test_sim_runner_produces_consistent_episode():
    from eval.harness import make_orca
    from eval.outcome_model import FULL_C2

    orca = make_orca(FULL_C2_SPEC, load_config())
    cfg = orca.choose_config()
    trace, metrics = SimRunner()(cfg, "A", condition=FULL_C2, episode_idx=0)
    assert trace.seed == metrics.seed == "A"
    assert 0.0 <= metrics.frontier_value <= 1.0
    assert metrics.frontier_milestone == trace.frontier_reached

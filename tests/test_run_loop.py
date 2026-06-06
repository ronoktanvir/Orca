"""Smoke test: one full episode runs end-to-end and emits the contracts (F5 / §8)."""

from __future__ import annotations

from config import load_config
from contracts import EpisodeMetrics, EpisodeTrace
from telemetry import init_telemetry
from train.loop import run


def test_run_one_episode_offline():
    settings = load_config()
    settings.telemetry.mode = "off"
    settings.run.n_episodes = 1
    telemetry = init_telemetry(mode="off")

    results = run(settings, telemetry=telemetry)

    assert len(results) == 1
    trace, metrics = results[0]
    assert isinstance(trace, EpisodeTrace)
    assert isinstance(metrics, EpisodeMetrics)
    # reached iron, the F4/F5 acceptance
    assert metrics.frontier_milestone.value == "iron"
    assert metrics.frontier_value >= 0.20
    assert trace.terminated_reason == "frontier_target"
    assert metrics.invalid_rate == 0.0


def test_trace_and_metrics_are_consistent():
    settings = load_config()
    settings.telemetry.mode = "off"
    telemetry = init_telemetry(mode="off")
    (trace, metrics), = run(settings, telemetry=telemetry)

    assert trace.episode_idx == metrics.episode_idx == 0
    assert trace.seed == metrics.seed
    assert trace.n_rounds == metrics.n_rounds
    assert trace.frontier_reached == metrics.frontier_milestone
    # every action in the trace is logged with a validity result
    assert len(trace.action_records) == trace.n_rounds  # single agent, one action/round
    assert all(rec.action is not None for rec in trace.action_records)
    # observation snapshots captured for audit
    assert len(trace.observations) == trace.n_rounds


def test_local_telemetry_writes_artifacts(tmp_path):
    settings = load_config()
    settings.telemetry.mode = "local"
    telemetry = init_telemetry(mode="local", run_dir=str(tmp_path), run_id="testrun")
    run(settings, telemetry=telemetry)

    run_dir = tmp_path / "testrun"
    assert (run_dir / "episode_0000.json").exists()
    assert (run_dir / "events.jsonl").exists()

#!/usr/bin/env python3
"""Orca entry point — run one (or more) full episode(s) end-to-end (F5 / §8).

Usage:
    python run.py                       # one episode on seed A, telemetry auto
    python run.py --seed A --episodes 1
    python run.py --telemetry off       # fully offline, no artifacts
    python run.py --config configs/default.yaml

Completing this script is the F1/F5 acceptance: env -> agent -> orca(no-op) -> log,
emitting EpisodeTrace + EpisodeMetrics and logging via Weave or the local/no-op
fallback.
"""

from __future__ import annotations

import argparse
import sys

from config import load_config
from telemetry import init_telemetry
from train.loop import run


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run an Orca episode (Phase 0 foundation).")
    p.add_argument("--config", default=None, help="path to a YAML config (default: configs/default.yaml)")
    p.add_argument("--seed", default=None, help="override run.seed")
    p.add_argument("--episodes", type=int, default=None, help="override run.n_episodes")
    p.add_argument(
        "--telemetry",
        default=None,
        choices=["auto", "weave", "local", "off"],
        help="override telemetry.mode",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    settings = load_config(args.config)
    if args.seed is not None:
        settings.run.seed = args.seed
    if args.episodes is not None:
        settings.run.n_episodes = args.episodes
    if args.telemetry is not None:
        settings.telemetry.mode = args.telemetry

    telemetry = init_telemetry(
        mode=settings.telemetry.mode,
        project=settings.telemetry.project,
        run_dir=settings.telemetry.run_dir,
    )

    print(f"[orca] starting run · seed={settings.run.seed} · "
          f"episodes={settings.run.n_episodes} · {telemetry.summary()}")

    results = run(settings, telemetry=telemetry)

    print("\n[orca] episode results")
    print("  ep | seed |        frontier | team_reward | rounds | reason")
    print("  ---+------+-----------------+-------------+--------+--------")
    for trace, metrics in results:
        print(
            f"  {metrics.episode_idx:>2} | {metrics.seed:<4} | "
            f"{metrics.frontier_milestone.value:>15} | "
            f"{metrics.team_reward:>11.3f} | {metrics.n_rounds:>6} | {trace.terminated_reason}"
        )

    reached_iron = any(
        m.frontier_milestone.value in ("iron",) or m.frontier_value >= 0.20
        for _t, m in results
    )
    print(f"\n[orca] reached iron: {reached_iron}")
    print(f"[orca] {telemetry.summary()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

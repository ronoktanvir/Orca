"""Eval records + aggregation (§9) — Stream 3 (O7).

A flat, runner-agnostic record per episode so plots never care whether the data
came from the calibrated outcome model or the real env/LLM loop. Always reported
with variance over multiple seeds/episodes — never a single anecdote (Law 4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, pstdev

from contracts import EpisodeMetrics

TRAIN = "train"
HELDOUT = "heldout"


@dataclass
class EpisodeRecord:
    """One episode's headline numbers, tagged by condition + train/held-out split."""

    condition: str
    seed: str
    split: str
    episode_idx: int
    frontier_value: float
    team_reward: float
    invalid_rate: float
    milestone: str
    won: bool
    arms: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_metrics(
        cls,
        metrics: EpisodeMetrics,
        *,
        condition: str,
        split: str,
        arms: dict[str, str] | None = None,
    ) -> "EpisodeRecord":
        return cls(
            condition=condition,
            seed=metrics.seed,
            split=split,
            episode_idx=metrics.episode_idx,
            frontier_value=metrics.frontier_value,
            team_reward=metrics.team_reward,
            invalid_rate=metrics.invalid_rate,
            milestone=metrics.frontier_milestone.value,
            won=metrics.won,
            arms=dict(arms or {}),
        )


@dataclass
class Stat:
    mean: float
    std: float
    n: int

    def as_tuple(self) -> tuple[float, float, int]:
        return (self.mean, self.std, self.n)


def stat_of(values: list[float]) -> Stat:
    if not values:
        return Stat(0.0, 0.0, 0)
    return Stat(mean(values), pstdev(values) if len(values) > 1 else 0.0, len(values))


def summarize(
    records: list[EpisodeRecord], *, field_name: str = "frontier_value"
) -> dict[tuple[str, str], Stat]:
    """Mean/std/n of ``field_name`` grouped by ``(condition, split)`` (§9)."""
    groups: dict[tuple[str, str], list[float]] = {}
    for r in records:
        groups.setdefault((r.condition, r.split), []).append(float(getattr(r, field_name)))
    return {k: stat_of(v) for k, v in groups.items()}


__all__ = ["EpisodeRecord", "Stat", "stat_of", "summarize", "TRAIN", "HELDOUT"]

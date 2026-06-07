"""Delegation bandit skeleton (§6.3) — the quantitative learner.

Phase 0's Orca is a no-op (it does not yet learn), but the bandit interface is
laid down here so Stream 3 (O2) can drop in the real ε-greedy / Thompson update
without touching the run loop. The bandit acts **once per episode** over a tiny
discrete space (situations × arms), so it is a contextual bandit by construction
— no PPO, no backprop (§6.3). Value = running mean of the episode team frontier
observed when an arm was chosen.

This is a working ε-greedy skeleton; Phase 0 simply doesn't call it yet.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class _ArmStats:
    count: int = 0
    mean: float = 0.0


@dataclass
class EpsilonGreedyBandit:
    """Per-situation ε-greedy bandit over a small discrete arm menu (§6.3).

    ``optimistic`` sets the initial value of every arm (Sutton & Barto optimistic
    initialisation): with it ≥ the max achievable frontier, untried arms always
    look best, so the bandit tries each arm before it starts exploiting — which
    is what makes "Orca finds the best delegation" reliable within dozens of
    episodes rather than luck-of-the-ε-draw. The first real update overwrites the
    optimistic seed (count 0→1), so it never biases the converged value.
    """

    arms: dict[str, list[str]]  # situation -> list of arm names
    epsilon: float = 0.2
    seed: int = 0
    optimistic: float = 0.0
    _stats: dict[str, dict[str, _ArmStats]] = field(default_factory=dict)
    _rng: random.Random = field(default_factory=lambda: random.Random(0))

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._stats = {
            sit: {arm: _ArmStats(mean=self.optimistic) for arm in arm_list}
            for sit, arm_list in self.arms.items()
        }

    def choose(self, situation: str) -> str:
        """Pick an arm for ``situation`` (ε-greedy)."""
        arm_list = self.arms[situation]
        if self._rng.random() < self.epsilon:
            return self._rng.choice(arm_list)
        return self.greedy(situation)

    def greedy(self, situation: str) -> str:
        """The current best-value arm (pure argmax, no exploration/RNG).

        Used for held-out eval and gate batches, where we want to measure the
        learned policy, not perturb it (§6.5, §9)."""
        stats = self._stats[situation]
        return max(self.arms[situation], key=lambda a: stats[a].mean)

    def update(self, situation: str, arm: str, frontier: float) -> None:
        """Update an arm's value with one episode's team frontier (§6.3)."""
        s = self._stats[situation][arm]
        s.count += 1
        s.mean += (frontier - s.mean) / s.count

    def values(self) -> dict[str, dict[str, float]]:
        """Current arm-value table — the data behind the learning curve (§6.3)."""
        return {
            sit: {arm: st.mean for arm, st in arms.items()} for sit, arms in self._stats.items()
        }


__all__ = ["EpsilonGreedyBandit"]

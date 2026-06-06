"""Stream 1 territory — the environment (`env/`).

Phase 0 ships a shallow stub (``StubEnv``) behind the frozen contracts. Stream 1
deepens it (graph-on-plane, full tech tree, stochasticity, cooperation) without
touching any other stream's folder.
"""

from __future__ import annotations

from .observation import serialize_observation, time_of_day
from .seeds import ALL_SEEDS, HELDOUT_SEEDS, TRAIN_SEEDS, make_world
from .stub_env import StepResult, StubEnv
from .world import AgentState, Region, World

__all__ = [
    "StubEnv",
    "StepResult",
    "World",
    "Region",
    "AgentState",
    "make_world",
    "serialize_observation",
    "time_of_day",
    "TRAIN_SEEDS",
    "HELDOUT_SEEDS",
    "ALL_SEEDS",
]

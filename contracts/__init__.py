"""Orca frozen interface contracts (ORCA_master_build_spec.md §11).

The seven pydantic models below are the locked interface that all three streams
(Env / Agents / Orca+Eval) build against. **Frozen after the fork** — any change
is additive-only (new optional fields) and must be broadcast to the whole team
(workflow plan §2, §8).

The seven contracts:
    1. Observation       — what a worker sees each turn (coord-free)   §3.2
    2. Action            — one macro-action per free turn               §3.3
    3. Message           — a structured bus message                     §5.1
    4. BehaviorCard      — Orca-owned WHO + coaching                     §6.2
    5. ExecutionMemory   — agent-owned HOW-TO heuristics                 §4.5
    6. EpisodeTrace      — raw replayable episode log                    §6.1
    7. EpisodeMetrics    — computed digest; carries the headline reward  §7
"""

from __future__ import annotations

from .action import Action
from .behavior_card import BehaviorCard
from .enums import (
    ActionName,
    Bearing,
    Biome,
    DistanceBand,
    Layer,
    MessageType,
    Milestone,
    Role,
    Structure,
    TimeOfDay,
)
from .episode import (
    ActionRecord,
    AgentStats,
    EpisodeMetrics,
    EpisodeTrace,
    MilestoneEvent,
    ReasoningRecord,
)
from .execution_memory import MEMORY_CAP, ExecutionMemory, Heuristic
from .message import Message
from .observation import (
    Exit,
    HereView,
    Landmark,
    Observation,
    SelfView,
    TeammateView,
)

# The seven frozen contracts.
CONTRACTS = (
    Observation,
    Action,
    Message,
    BehaviorCard,
    ExecutionMemory,
    EpisodeTrace,
    EpisodeMetrics,
)

__all__ = [
    # the seven
    "Observation",
    "Action",
    "Message",
    "BehaviorCard",
    "ExecutionMemory",
    "EpisodeTrace",
    "EpisodeMetrics",
    "CONTRACTS",
    # observation sub-models
    "Exit",
    "HereView",
    "Landmark",
    "SelfView",
    "TeammateView",
    # episode sub-models
    "ActionRecord",
    "AgentStats",
    "MilestoneEvent",
    "ReasoningRecord",
    # memory sub-models
    "Heuristic",
    "MEMORY_CAP",
    # enums
    "ActionName",
    "Bearing",
    "Biome",
    "DistanceBand",
    "Layer",
    "MessageType",
    "Milestone",
    "Role",
    "Structure",
    "TimeOfDay",
]

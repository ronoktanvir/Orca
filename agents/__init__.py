"""Stream 2 territory — workers (`agents/`).

Phase 0 ships the scripted ``ShallowOracle`` placeholder + the frozen agent
interface, role primers, and the execution-memory guard filter. Stream 2 swaps in
the real ``LLMWorker`` (turn loop, prompts, JSON parse/validate, memory write).
"""

from __future__ import annotations

from .base import Agent
from .memory import (
    guard_filter,
    looks_seed_specific,
    scrub_seed_specific,
    update_execution_memory,
)
from .prompts import (
    ROLE_PRIMERS,
    WorkerOutput,
    build_memory_prompt,
    build_repair_prompt,
    build_worker_prompt,
)
from .scripted import ShallowOracle
from .worker import LLMWorker

__all__ = [
    "Agent",
    "ShallowOracle",
    "LLMWorker",
    "ROLE_PRIMERS",
    "WorkerOutput",
    "build_worker_prompt",
    "build_repair_prompt",
    "build_memory_prompt",
    "guard_filter",
    "looks_seed_specific",
    "scrub_seed_specific",
    "update_execution_memory",
]

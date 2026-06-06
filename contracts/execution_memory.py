"""Contract 5/7 — ``ExecutionMemory``: agent-owned transferable HOW-TO (§4.5).

A *bounded* (cap 8, §15) list of schema'd heuristics — never free-form notes.
Written at episode end, guard-filtered (strip coord-like / seed-specific content
in ``agents/memory.py``), and accept-gated before it persists. The cap is part
of the contract: the list cannot exceed 8 entries.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

MEMORY_CAP = 8  # §15: memory cap = 8 heuristics/agent


class Heuristic(BaseModel):
    """One transferable HOW-TO rule (§4.5)."""

    condition: str  # e.g. "need iron but only wooden pickaxe"
    action: str  # e.g. "craft stone pickaxe first"
    confidence: float = Field(ge=0.0, le=1.0)


class ExecutionMemory(BaseModel):
    """The agent's own persistent heuristic store."""

    agent_id: str
    heuristics: list[Heuristic] = Field(default_factory=list, max_length=MEMORY_CAP)


__all__ = ["ExecutionMemory", "Heuristic", "MEMORY_CAP"]

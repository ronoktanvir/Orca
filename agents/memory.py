"""Execution-memory: schema + guard filter (§4.5) — Stream 2 (A5).

The schema (``ExecutionMemory`` / ``Heuristic``, cap 8) is frozen in contracts.
The guard filter — strip seed-specific / coordinate-like content before a
heuristic persists — is the concrete "harness sophistication" mechanism (§4.5)
and the third leg of the coordinate-leak invariant. A working filter ships here;
Stream 2 adds the LLM-written memory, ``learning_signal`` modulation, and the
accept-gated persistence.
"""

from __future__ import annotations

import re

from contracts import ExecutionMemory, Heuristic

# Patterns that look seed-specific / coordinate-like and must not enter memory.
_COORD_PATTERNS = [
    re.compile(r"-?\d+\.\d+\s*[,;]\s*-?\d+\.\d+"),  # float pairs e.g. "12.0, 3.5"
    re.compile(r"\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\)"),  # (x, y)
    re.compile(r"\br_\d+\b"),  # internal region ids e.g. r_07
    re.compile(r"\bpos\b", re.IGNORECASE),
    re.compile(r"\b\d+\s*(?:blocks?|meters?|tiles?|steps?)\b", re.IGNORECASE),  # distances
]


def looks_seed_specific(text: str) -> bool:
    """True if ``text`` contains coordinate-like / seed-specific content (§4.5)."""
    return any(p.search(text) for p in _COORD_PATTERNS)


def guard_filter(memory: ExecutionMemory) -> ExecutionMemory:
    """Drop heuristics whose condition/action leaks coords or seed specifics."""
    kept: list[Heuristic] = [
        h
        for h in memory.heuristics
        if not (looks_seed_specific(h.condition) or looks_seed_specific(h.action))
    ]
    return ExecutionMemory(agent_id=memory.agent_id, heuristics=kept)


__all__ = ["guard_filter", "looks_seed_specific"]

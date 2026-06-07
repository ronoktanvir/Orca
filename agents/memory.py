"""Execution-memory: schema + guard filter + learning-signal write (§4.5) — Stream 2 (A5).

The schema (``ExecutionMemory`` / ``Heuristic``, cap 8) is frozen in contracts.
The guard filter — strip seed-specific / coordinate-like content before a
heuristic persists — is the concrete "harness sophistication" mechanism (§4.5)
and the third leg of the coordinate-leak invariant. This module ships:

  * ``looks_seed_specific`` / ``scrub_seed_specific`` — the leak detector + redactor,
  * ``guard_filter`` — drop leaky heuristics before persistence,
  * ``update_execution_memory`` — the agent-owned episode-end write whose edit
    magnitude is modulated by Orca's ``learning_signal`` (+1 add/strengthen,
    ~0 no change, −1 weaken/remove the flagged rule).

The LLM that *proposes* the heuristics lives in ``agents/worker.py``; persistence
is accept-gated by Stream 3 (§6.5).
"""

from __future__ import annotations

import re

from contracts import ExecutionMemory, Heuristic
from contracts.execution_memory import MEMORY_CAP

# Patterns that look seed-specific / coordinate-like and must not enter memory.
_COORD_PATTERNS = [
    re.compile(r"-?\d+\.\d+\s*[,;]\s*-?\d+\.\d+"),  # float pairs e.g. "12.0, 3.5"
    re.compile(r"\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\)"),  # (x, y)
    re.compile(r"\br_\d+\b"),  # internal region ids e.g. r_07
    re.compile(r"\bpos\b", re.IGNORECASE),
    re.compile(r"\b\d+\s*(?:blocks?|meters?|tiles?|steps?)\b", re.IGNORECASE),  # distances
]

# Edits below this |learning_signal| are treated as "no change" (§4.5 ~0 dial).
NEUTRAL_BAND = 0.05
# Confidence floor below which a weakened heuristic is removed entirely.
_CONFIDENCE_FLOOR = 0.1


def looks_seed_specific(text: str) -> bool:
    """True if ``text`` contains coordinate-like / seed-specific content (§4.5)."""
    return any(p.search(text) for p in _COORD_PATTERNS)


def scrub_seed_specific(text: str) -> str:
    """Strip coordinate-like / seed-specific spans from ``text`` (§4.5).

    Used to *sanitize* model-emitted message/report content so coordination
    survives while the coord-leak invariant holds. Coordinate spans are removed
    (not just flagged) and whitespace is collapsed; a string that is *only*
    coordinates therefore scrubs down to empty (the caller then drops it).
    """
    cleaned = text or ""
    for p in _COORD_PATTERNS:
        cleaned = p.sub(" ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def guard_filter(memory: ExecutionMemory) -> ExecutionMemory:
    """Drop heuristics whose condition/action leaks coords or seed specifics."""
    kept: list[Heuristic] = [
        h
        for h in memory.heuristics
        if not (looks_seed_specific(h.condition) or looks_seed_specific(h.action))
    ]
    return ExecutionMemory(agent_id=memory.agent_id, heuristics=kept)


def _key(h: Heuristic) -> tuple[str, str]:
    return (h.condition.strip().lower(), h.action.strip().lower())


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def update_execution_memory(
    memory: ExecutionMemory,
    proposed: list[Heuristic],
    learning_signal: float,
    *,
    cap: int = MEMORY_CAP,
) -> ExecutionMemory:
    """Apply an episode-end memory write modulated by ``learning_signal`` (§4.5).

    - ``|learning_signal| <= NEUTRAL_BAND``: minimal change — keep existing
      (guard-filtered) heuristics, no add/remove.
    - ``learning_signal > 0``: ADD new transferable heuristics and STRENGTHEN
      matching existing ones (confidence bump scaled by the signal).
    - ``learning_signal < 0``: WEAKEN/REMOVE the heuristics Orca flagged (passed
      in ``proposed``); confidence is reduced by |signal| and rules that fall
      below the floor are dropped.

    The result is always guard-filtered (never persists coord-like / seed-specific
    text) and capped at ``cap`` (keeping the highest-confidence rules).
    """
    base = guard_filter(memory).heuristics
    # Guard-filter the proposal too — leaky proposals never get a foothold.
    clean_proposed = [
        h for h in proposed if not (looks_seed_specific(h.condition) or looks_seed_specific(h.action))
    ]

    if abs(learning_signal) <= NEUTRAL_BAND:
        kept = list(base)
    elif learning_signal > 0:
        by_key = {_key(h): h for h in base}
        bump = 0.2 * learning_signal
        for h in clean_proposed:
            k = _key(h)
            if not h.condition.strip() or not h.action.strip():
                continue
            if k in by_key:
                cur = by_key[k]
                by_key[k] = Heuristic(
                    condition=cur.condition,
                    action=cur.action,
                    confidence=_clamp(cur.confidence + bump),
                )
            else:
                by_key[k] = Heuristic(
                    condition=h.condition,
                    action=h.action,
                    confidence=_clamp(h.confidence),
                )
        kept = list(by_key.values())
    else:  # learning_signal < 0 -> weaken/remove flagged rules
        flagged = {_key(h) for h in clean_proposed}
        penalty = 0.5 * abs(learning_signal)
        kept = []
        for h in base:
            if _key(h) in flagged:
                new_conf = h.confidence - penalty
                if new_conf < _CONFIDENCE_FLOOR:
                    continue  # removed
                kept.append(
                    Heuristic(condition=h.condition, action=h.action, confidence=_clamp(new_conf))
                )
            else:
                kept.append(h)

    if len(kept) > cap:
        kept = sorted(kept, key=lambda h: h.confidence, reverse=True)[:cap]

    return guard_filter(ExecutionMemory(agent_id=memory.agent_id, heuristics=kept))


__all__ = [
    "guard_filter",
    "looks_seed_specific",
    "scrub_seed_specific",
    "update_execution_memory",
    "NEUTRAL_BAND",
]

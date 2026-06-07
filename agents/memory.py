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
from typing import Any

from bus.messages import normalize_recipient
from contracts import Action, ExecutionMemory, Heuristic
from contracts.enums import ActionName
from contracts.execution_memory import MEMORY_CAP

# Patterns that look seed-specific / coordinate-like and must not enter memory.
# Order matters for ``scrub_seed_specific`` (applied in sequence): the specific
# float / parenthesized / region / distance spans are removed BEFORE the generic
# integer-pair pattern, so a float pair like "12.0, 3.5" is excised whole instead
# of leaving a digit fragment behind.
_COORD_PATTERNS = [
    re.compile(r"-?\d+\.\d+\s*[,;]\s*-?\d+\.\d+"),  # float pairs e.g. "12.0, 3.5"
    re.compile(r"\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\)"),  # (x, y)
    re.compile(r"\br_\d+\b"),  # internal region ids e.g. r_07
    re.compile(r"\bpos\b", re.IGNORECASE),
    re.compile(r"\b\d+\s*(?:blocks?|meters?|tiles?|steps?)\b", re.IGNORECASE),  # distances
    # Integer coordinate-like pairs e.g. "12, 3" / "12;3". Catches bare integer
    # pairs the float pattern misses, while leaving legitimate ids like "agent_2"
    # (no digit,digit sequence) and lone counts (e.g. "6 ingots") untouched.
    re.compile(r"-?\d+\s*[,;]\s*-?\d+"),
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


# --------------------------------------------------------------------------- #
# Action-arg sanitization (§3.2) — shared so the run loop can enforce the leak
# invariant at the env boundary for ANY worker (not only LLMWorker), covering the
# advertised ``worker_factory`` seam.
# --------------------------------------------------------------------------- #
def _is_coord_number(x: Any) -> bool:
    """A real number (not bool) — matches obs_guard's coordinate-pair test."""
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def scrub_arg_value(value: Any) -> tuple[Any, bool]:
    """Recursively sanitize an action-arg value so it can never leak (§3.2).

    Returns ``(scrubbed_value, unusable)``. ``unusable`` is True when the value
    cannot be salvaged without leaking — a string that was *entirely* a
    coordinate/seed leak (scrubs to empty) or a 2-element numeric list/tuple (a
    coordinate pair). Leaky dict KEYS (region ids / "pos" / coordinate-like text)
    are dropped. Enum-ish values ("N", "iron_ore", "agent_2") and clean scalars
    pass through unchanged.
    """
    unusable = False
    if isinstance(value, str):
        if looks_seed_specific(value):
            cleaned = scrub_seed_specific(value)
            if value.strip() and not cleaned:
                return cleaned, True  # entirely a leak -> unusable
            return cleaned, False
        return value, False
    if isinstance(value, dict):
        out: dict = {}
        for k, v in value.items():
            if isinstance(k, str) and looks_seed_specific(k):
                continue  # drop keys that themselves leak (e.g. "r_07", "pos")
            nv, u = scrub_arg_value(v)
            out[k] = nv
            unusable = unusable or u
        return out, unusable
    if isinstance(value, (list, tuple)):
        if len(value) == 2 and all(_is_coord_number(x) for x in value):
            return value, True  # a 2-element all-numeric list/tuple is a coord pair
        items = []
        for v in value:
            nv, u = scrub_arg_value(v)
            items.append(nv)
            unusable = unusable or u
        return (tuple(items) if isinstance(value, tuple) else items), unusable
    return value, False


def sanitize_action_args(action: Action) -> Action:
    """Return ``action`` with every arg leak-free (§3.2), else a ``wait`` fallback.

    Walks all values AND keys (through nested dict/list/tuple). For comm actions
    the ``to`` arg is recipient-validated first (a leaky ``to`` downgrades to
    ``team`` rather than nuking the report). Leaky keys are dropped; a numeric
    coordinate pair or an arg that scrubs to empty makes the action fall back to
    ``wait``. Pure (no telemetry) so the run loop can apply it to every collected
    action — the trace's ``ActionRecord``s stay leak-free regardless of which
    worker produced them. Idempotent, so re-sanitizing an LLMWorker action is a
    no-op."""
    if not action.args:
        return action
    args = action.args
    if action.name in (ActionName.REPORT, ActionName.REQUEST_HELP) and "to" in args:
        args = {**args, "to": normalize_recipient(args.get("to"))}
    sanitized, unusable = scrub_arg_value(args)
    if unusable:
        return Action(name=ActionName.WAIT)
    if sanitized == action.args:
        return action
    return Action(name=action.name, args=sanitized)


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
    "scrub_arg_value",
    "sanitize_action_args",
    "update_execution_memory",
    "NEUTRAL_BAND",
]

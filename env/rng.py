"""Deterministic seeded RNG (§3.6).

Every stochastic draw uses ``rng = make_rng(seed, episode_idx, round, agent_id)``
so identical inputs give identical outcomes — and, crucially, *across processes*.
We derive the integer seed via SHA-256 rather than Python's builtin ``hash()``,
because ``hash()`` of strings is salted per-process (PYTHONHASHSEED) and would
break reproducibility between runs.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

_MASK32 = 0xFFFFFFFF


def derive_seed(*parts: Any) -> int:
    """Stable 32-bit integer seed from arbitrary parts (process-independent)."""
    key = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest, 16) & _MASK32


def make_rng(seed: str, episode_idx: int, round_idx: int, agent_id: str) -> random.Random:
    """A fresh, deterministic ``random.Random`` for one (seed, episode, round, agent)."""
    return random.Random(derive_seed(seed, episode_idx, round_idx, agent_id))


__all__ = ["derive_seed", "make_rng"]

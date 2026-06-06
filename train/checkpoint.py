"""Checkpointing (§8) — Stream 3.

Persist Orca state (cards, memories, bandit tables) every episode so a run can
resume / roll back / demo any point (§8). Phase 0 has nothing learned to persist;
this is the interface for the streams.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_checkpoint(state: dict[str, Any], path: str | Path) -> Path:
    """Write a JSON checkpoint (placeholder; deepened in the streams)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    return path


__all__ = ["save_checkpoint"]

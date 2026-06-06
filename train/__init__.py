"""Run loop + phasing + checkpointing (the integration glue, F5 / §8).

This is the foundation's thin end-to-end loop. The full §8 loop (bandit update,
coaching, accept-gate) emerges as the streams fill in their stubs.
"""

from __future__ import annotations

from .checkpoint import save_checkpoint
from .loop import run, run_episode
from .phases import Phase, current_phase

__all__ = ["run", "run_episode", "Phase", "current_phase", "save_checkpoint"]

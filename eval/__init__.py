"""Stream 3 territory — evaluation (`eval/`). §9.

The headline result: three conditions (static / comms-no-Orca / Full C2), a
transfer test (train on {A,T2,T3} → eval frozen on held-out {B,C}), ablations,
and five matplotlib figures. The outcome source is a pluggable runner — the
calibrated ``outcome_model`` for offline/CI plots, or the real env/LLM loop via
``RealRunner`` — behind one flat record format (Law 4: always report variance).
"""

from __future__ import annotations

from .harness import (
    COMMS_SPEC,
    FULL_C2_SPEC,
    STATIC_SPEC,
    RealRunner,
    SimRunner,
    run_ablations,
    run_learning_curve,
    run_transfer,
)
from .records import EpisodeRecord, summarize
from .transfer import transfer_verdict

__all__ = [
    "SimRunner",
    "RealRunner",
    "STATIC_SPEC",
    "COMMS_SPEC",
    "FULL_C2_SPEC",
    "run_transfer",
    "run_ablations",
    "run_learning_curve",
    "transfer_verdict",
    "EpisodeRecord",
    "summarize",
]

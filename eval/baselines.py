"""Baselines (§9) — Stream 3 (O7).

The three conditions compared throughout §9, defined once in ``harness`` and
re-exported here for discoverability:

  * ``STATIC_SPEC``  — fixed balanced roster, no Orca, no comms, no memory.
  * ``COMMS_SPEC``   — agents message, but no delegation/coaching/memory.
  * ``FULL_C2_SPEC`` — Orca bandit + (phased) coaching + memory + accept-gate.

``run_baselines`` evaluates all three (frozen) on the train pool + held-out and
returns flat records; the transfer/ablation experiments build on these.
"""

from __future__ import annotations

from typing import Optional

from config import OrcaSettings, load_config

from .harness import (
    COMMS_SPEC,
    FULL_C2_SPEC,
    STATIC_SPEC,
    Runner,
    SimRunner,
    eval_batch,
    make_orca,
    train_full_c2,
)
from .records import HELDOUT, TRAIN, EpisodeRecord


def run_baselines(
    settings: Optional[OrcaSettings] = None,
    *,
    runner: Optional[Runner] = None,
    n_train: int = 30,
    eval_reps: int = 6,
    gate_batch: Optional[int] = None,
) -> list[EpisodeRecord]:
    """Eval all three conditions on train + held-out; Full C2 is trained first."""
    settings = settings or load_config()
    runner = runner or SimRunner()
    gate_batch = settings.eval.gate_batch if gate_batch is None else gate_batch
    train_seeds = list(settings.seeds.train)
    heldout = list(settings.seeds.heldout)

    records: list[EpisodeRecord] = []
    tr = train_full_c2(FULL_C2_SPEC, settings, runner, train_seeds, n_train, gate_batch=gate_batch)
    records += eval_batch(tr.orca, runner, train_seeds, FULL_C2_SPEC, TRAIN, reps=eval_reps)
    records += eval_batch(tr.orca, runner, heldout, FULL_C2_SPEC, HELDOUT, reps=eval_reps)
    for spec in (STATIC_SPEC, COMMS_SPEC):
        orca = make_orca(spec, settings)
        orca.freeze()
        records += eval_batch(orca, runner, train_seeds, spec, TRAIN, reps=eval_reps)
        records += eval_batch(orca, runner, heldout, spec, HELDOUT, reps=eval_reps)
    return records


__all__ = ["STATIC_SPEC", "COMMS_SPEC", "FULL_C2_SPEC", "run_baselines"]

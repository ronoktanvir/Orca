"""Ablations (§9) — Stream 3 (O7).

Full C2 with one knob removed at a time — memory ON/OFF · coaching ON/OFF ·
accept-gate ON/OFF — each trained then evaluated on held-out {B,C}, one bar on
the same axes. The accept-gate ablation is the important one: it shows the gate's
anti-noise value (ungated coaching keeps every noisy edit and regresses).

The experiment lives in ``harness.run_ablations``; re-exported here.
"""

from __future__ import annotations

from .harness import ABL_NO_COACH, ABL_NO_GATE, ABL_NO_MEMORY, FULL_C2_SPEC, run_ablations

__all__ = ["run_ablations", "FULL_C2_SPEC", "ABL_NO_MEMORY", "ABL_NO_COACH", "ABL_NO_GATE"]

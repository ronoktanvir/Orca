"""Transfer eval — the money plot (§9) — Stream 3 (O7).

Train Full C2 on {A,T2,T3}; freeze the learned bandit + cards; evaluate all three
conditions on held-out {B,C}. Hypothesis tested: Full C2 ≥ baselines on held-out
⇒ a *transferable delegation strategy*, not memorised terrain. Held-out seeds are
never trained on (the frozen Orca's ``trained_seeds`` proves it).

**Caveat:** with the default ``SimRunner`` this is demonstrated on the *offline
calibrated outcome model* (``eval/outcome_model.py``) — a CI/demo scaffold, not
the real LLM-worker environment. It is not yet evidence that the real C2 system
transfers; that requires Stream 2's ``LLMWorker`` evaluated via
``harness.RealRunner``. The harness/plots/verdict are runner-agnostic, so the same
code produces the real result once that path is live.

The experiment lives in ``harness.run_transfer``; this module re-exports it plus a
one-line verdict helper used by the demo + tests.
"""

from __future__ import annotations

from .harness import TransferResult, run_transfer
from .outcome_model import COMMS, FULL_C2, STATIC
from .records import HELDOUT, EpisodeRecord, summarize


def transfer_verdict(records: list[EpisodeRecord]) -> dict[str, float | bool]:
    """Held-out means per condition + whether Full C2 ≥ both baselines (§9)."""
    stats = summarize(records, field_name="frontier_value")

    def held(cond: str) -> float:
        st = stats.get((cond, HELDOUT))
        return st.mean if st else 0.0

    full = held("full_c2")
    static = held("static")
    comms = held("comms")
    return {
        "full_c2_heldout": round(full, 4),
        "static_heldout": round(static, 4),
        "comms_heldout": round(comms, 4),
        "full_c2_wins": full >= static and full >= comms,
    }


__all__ = ["run_transfer", "TransferResult", "transfer_verdict", "STATIC", "COMMS", "FULL_C2"]

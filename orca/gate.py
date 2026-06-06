"""Accept/reject gate (§6.5) — Stream 3 (O5).

Hill-climb-with-rollback: keep an Orca update iff it doesn't regress mean team
frontier on the eval seed pool (within ε), else roll back. This turns noisy LLM
edits into monotone-ish improvement. Phase 0 always "keeps" because the no-op
Orca proposes nothing.
"""

from __future__ import annotations

from .orca import Proposal


def accept_gate(proposal: Proposal, *, epsilon: float = 0.0) -> bool:
    """Phase 0 placeholder — nothing to gate (no-op Orca). Returns True."""
    return True


__all__ = ["accept_gate"]

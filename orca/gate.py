"""Accept/reject gate (§6.5) — Stream 3 (O5).

Hill-climb-with-rollback: after Orca proposes new cards/memory, run a small eval
batch on the **train** seed pool and **keep the update iff mean team frontier ≥
current best − ε**; otherwise roll the cards back. This turns noisy LLM edits
into monotone-ish improvement and is the anti-noise safeguard the spec calls
critical. A static-baseline snapshot is kept for comparison throughout (§9).

The decision rule (:meth:`AcceptGate.consider`) is a pure function of numbers so
it is trivially testable; :meth:`AcceptGate.evaluate` wires it to an Orca +
``eval_fn`` (provided by the loop) and performs the snapshot/commit/rollback.

Held-out seeds are **never** passed to the gate — it scores only on the train
pool (anti-leakage, §9 / Law 4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Callable, Optional

from contracts import EpisodeMetrics

from .orca import Proposal


@dataclass
class GateDecision:
    """One accept/reject outcome (logged to Weave, §10)."""

    accepted: bool
    mean_frontier: float
    best: float
    epsilon: float
    rolled_back: bool = False
    n_eval: int = 0
    note: str = ""


@dataclass
class AcceptGate:
    """Hill-climb-with-rollback gate over the train seed pool (§6.5)."""

    epsilon: float = 0.02
    baseline: float = 0.0  # starting "current best"; also the static-baseline snapshot
    static_baseline: float = field(init=False)
    best: float = field(init=False)
    history: list[GateDecision] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.static_baseline = self.baseline
        self.best = self.baseline

    # -- pure decision rule -------------------------------------------------- #
    def consider(self, mean_frontier: float, *, n_eval: int = 0, note: str = "") -> GateDecision:
        """Keep iff ``mean_frontier ≥ best − ε``; ratchet ``best`` up on keep."""
        accepted = mean_frontier >= self.best - self.epsilon
        if accepted:
            self.best = max(self.best, mean_frontier)
        dec = GateDecision(
            accepted=accepted,
            mean_frontier=mean_frontier,
            best=self.best,
            epsilon=self.epsilon,
            rolled_back=not accepted,
            n_eval=n_eval,
            note=note,
        )
        self.history.append(dec)
        return dec

    # -- orchestration ------------------------------------------------------- #
    def evaluate(
        self,
        orca: Any,
        proposal: Proposal,
        eval_fn: Callable[[], list[EpisodeMetrics]],
        *,
        metric: str = "team_reward",
        telemetry: Any = None,
    ) -> GateDecision:
        """Tentatively commit ``proposal``, eval on the train pool, keep or roll back.

        ``eval_fn()`` runs the small batch with the *currently committed* cards and
        returns its ``EpisodeMetrics`` (the loop binds it to the train seeds with
        the bandit frozen so the gate measures card quality, not exploration).
        """
        if proposal is None or proposal.is_empty():
            dec = GateDecision(
                accepted=True,
                mean_frontier=self.best,
                best=self.best,
                epsilon=self.epsilon,
                note="empty-proposal",
            )
            self.history.append(dec)
            return dec

        snap = orca.snapshot()
        orca.commit(proposal)
        metrics_list = eval_fn() or []
        scores = [float(getattr(m, metric)) for m in metrics_list]
        mean_frontier = mean(scores) if scores else self.best
        dec = self.consider(mean_frontier, n_eval=len(scores), note=proposal.notes)
        if dec.rolled_back:
            orca.restore(snap)

        if telemetry is not None:
            try:
                telemetry.log_event(
                    "accept_gate",
                    {
                        "accepted": dec.accepted,
                        "rolled_back": dec.rolled_back,
                        "mean_frontier": round(dec.mean_frontier, 4),
                        "best": round(dec.best, 4),
                        "epsilon": dec.epsilon,
                        "static_baseline": round(self.static_baseline, 4),
                        "n_eval": dec.n_eval,
                        "edited_agents": list(proposal.behavior_cards.keys()),
                    },
                )
            except Exception:
                pass
        return dec


def accept_gate(proposal: Proposal, *, epsilon: float = 0.0) -> bool:
    """Back-compat thin gate (no eval batch available) — keeps everything.

    The real anti-noise gating uses :class:`AcceptGate`; this preserves the
    original Phase-0 loop call (``if accept_gate(proposal): commit``) so ``main``
    runs offline without a train-pool runner wired in.
    """
    return True


__all__ = ["AcceptGate", "GateDecision", "accept_gate"]

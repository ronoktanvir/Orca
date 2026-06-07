"""Verbal coach + credit assignment (§6.4) — Stream 3 (O4).

Orca reads the objective **digest** (§6.1) and reasons in natural language about
*credit*: was a stall a **delegation error** (wrong who/assignment) or an
**execution error** (right assignment, the agent lacked a step/tool)? That
reasoning is written into the next ``BehaviorCard`` (assignment + directives,
``version`` bumped) and logged to Weave (§10) — a readable, auditable layer that
is more compelling than a counterfactual plot.

Two paths, same output contract:
  * **LLM path** — ``orca_llm.complete(prompt, schema=CoachOutput)`` → validated
    JSON. One repair retry; any failure falls through to ↓.
  * **Heuristic path** — a deterministic, offline-safe credit assignment straight
    from the digest's bottleneck flags. This is what keeps ``pytest`` / ``main``
    runnable without an API key, and makes the failure→fix demo reproducible.

``performance_score`` stays objective (§7.3, computed by ``scoring``); the coach
only sets ``learning_signal`` (its "adopt this lesson?" dial) and the card text.
``team_reward`` is never touched here (anti-circularity, §6.4).
"""

from __future__ import annotations

import json
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

from contracts import BehaviorCard, EpisodeMetrics, EpisodeTrace
from contracts.enums import Role
from telemetry import op

from .cards import make_default_card
from .digest import TraceDigest, build_digest
from .orca import Proposal
from .scoring import performance_score

CreditKind = Literal["delegation", "execution", "none"]


# --------------------------------------------------------------------------- #
# LLM output schema (validated with pydantic; the client returns raw text).
# --------------------------------------------------------------------------- #
class AgentCoaching(BaseModel):
    agent_id: str
    credit: CreditKind = "none"
    reasoning: str = ""
    new_assignment: Optional[str] = None
    new_directives: list[str] = Field(default_factory=list)
    learning_signal: float = Field(default=0.0, ge=-1.0, le=1.0)


class CoachOutput(BaseModel):
    team_reasoning: str = ""
    agents: list[AgentCoaching] = Field(default_factory=list)


_SYSTEM = (
    "You are Orca, a manager that coaches a team of worker agents between episodes. "
    "You assign credit objectively and edit each worker's behavior-card. Distinguish "
    "DELEGATION errors (wrong assignment / wrong who) from EXECUTION errors (right "
    "assignment, the worker missed a prerequisite step or tool). Be concrete and brief."
)


def _prompt(digest: TraceDigest, cards: dict[str, BehaviorCard]) -> str:
    card_lines = []
    for aid, c in cards.items():
        card_lines.append(
            f"  {aid} ({c.role.value}) v{c.version}: assign='{c.assignment}' "
            f"directives={c.directives}"
        )
    cards_block = "\n".join(card_lines) if card_lines else "  (all agents on role defaults)"
    return (
        "Here is the objective digest of the last episode:\n\n"
        f"{digest.render()}\n\n"
        "Current behavior-cards:\n"
        f"{cards_block}\n\n"
        "For each agent that should change, decide credit ('delegation' or 'execution'), "
        "give one sentence of reasoning, and propose a new_assignment and/or new_directives "
        "that would clear the bottleneck next episode. Set learning_signal in [-1,1] "
        "(+ to add/strengthen a rule, - to weaken one). Only include agents that need a "
        "change. Respond as JSON: "
        '{"team_reasoning": str, "agents": [{"agent_id", "credit", "reasoning", '
        '"new_assignment", "new_directives", "learning_signal"}]}.'
    )


def _call_llm(llm: Any, digest: TraceDigest, cards: dict[str, BehaviorCard]) -> Optional[CoachOutput]:
    """One LLM call + one repair retry; returns ``None`` on any failure."""
    prompt = _prompt(digest, cards)
    for attempt in range(2):
        try:
            raw = llm.complete(prompt, schema=CoachOutput, system=_SYSTEM)
        except TypeError:
            raw = llm.complete(prompt, schema=CoachOutput)  # client without system kw
        except Exception:
            return None
        try:
            return CoachOutput.model_validate_json(raw)
        except (ValidationError, ValueError):
            try:
                return CoachOutput.model_validate(json.loads(raw))
            except Exception:
                if attempt == 0:
                    continue  # retry once
                return None
    return None


# --------------------------------------------------------------------------- #
# Heuristic (offline) credit assignment — deterministic from the digest.
# --------------------------------------------------------------------------- #
def _heuristic(digest: TraceDigest) -> CoachOutput:
    frontier_stalled = any("frontier stuck" in b for b in digest.bottlenecks)
    agents: list[AgentCoaching] = []
    for a in digest.agents:
        # Execution error: the agent kept failing the same action (missing step/tool).
        if a.top_invalid and a.top_invalid[2] >= 2:
            action, reason, cnt = a.top_invalid
            fix = (
                f"Before '{action}', ensure its prerequisites are met"
                + (f" (env said: {reason})" if reason else "")
                + "."
            )
            agents.append(
                AgentCoaching(
                    agent_id=a.agent_id,
                    credit="execution",
                    reasoning=(
                        f"{a.agent_id} failed '{action}' {cnt}x — right assignment, "
                        f"missing a prerequisite step; keep the role, add a directive."
                    ),
                    new_directives=[fix],
                    learning_signal=min(1.0, 0.4 + 0.1 * cnt),
                )
            )
        # Delegation error: idled/stalled while the team frontier was stuck.
        elif frontier_stalled and (
            "mostly idle" in a.flags or any("stalled" in f for f in a.flags)
        ):
            agents.append(
                AgentCoaching(
                    agent_id=a.agent_id,
                    credit="delegation",
                    reasoning=(
                        f"{a.agent_id} idled/stalled while the frontier was stuck — "
                        f"a delegation gap; give it productive work earlier."
                    ),
                    new_assignment=(
                        "Take initiative early: pursue the next blocking subtask "
                        "instead of waiting for a handoff."
                    ),
                    learning_signal=0.6,
                )
            )
    n_deleg = sum(1 for a in agents if a.credit == "delegation")
    n_exec = sum(1 for a in agents if a.credit == "execution")
    if not agents:
        team_reasoning = "No changes warranted — no clear bottleneck this episode."
    else:
        parts = []
        if n_deleg:
            parts.append(f"{n_deleg} delegation fix(es) (reassign idle/stalled workers)")
        if n_exec:
            parts.append(f"{n_exec} execution fix(es) (add missing-prerequisite directives)")
        prefix = "Frontier stalled; " if frontier_stalled else ""
        team_reasoning = prefix + " and ".join(parts) + "."
    return CoachOutput(team_reasoning=team_reasoning, agents=agents)


# --------------------------------------------------------------------------- #
def _roles_from_metrics(metrics: EpisodeMetrics) -> dict[str, Role]:
    return {st.agent_id: st.role for st in metrics.agent_stats}


def _apply(
    out: CoachOutput,
    cards: dict[str, BehaviorCard],
    roles: dict[str, Role],
) -> dict[str, BehaviorCard]:
    """Turn the coach's per-agent edits into new BehaviorCards (version bumped)."""
    new_cards: dict[str, BehaviorCard] = {}
    for ac in out.agents:
        if ac.new_assignment is None and not ac.new_directives:
            continue  # nothing actionable
        role = roles.get(ac.agent_id, Role.MINER)
        base = cards.get(ac.agent_id) or make_default_card(ac.agent_id, role)
        directives = list(base.directives)
        for d in ac.new_directives:
            if d and d not in directives:
                directives.append(d)
        new_cards[ac.agent_id] = base.model_copy(
            update={
                "role": role,
                "assignment": ac.new_assignment or base.assignment,
                "directives": directives,
                "version": base.version + 1,
            }
        )
    return new_cards


@op
def run_coach(
    trace: EpisodeTrace,
    metrics: EpisodeMetrics,
    *,
    cards: Optional[dict[str, BehaviorCard]] = None,
    llm: Any = None,
    telemetry: Any = None,
) -> Proposal:
    """Read the digest, assign credit, and propose the next cards (§6.4).

    ``cards`` are the currently-committed coaching overrides (by agent_id).
    Decorated ``@op`` so the reasoning nests in the Weave trace (§10).
    """
    cards = cards or {}
    digest = build_digest(trace, metrics)
    roles = _roles_from_metrics(metrics)

    out = _call_llm(llm, digest, cards) if llm is not None else None
    source = "llm"
    if out is None:
        out = _heuristic(digest)
        source = "heuristic"

    new_cards = _apply(out, cards, roles)

    # Advisory scores: performance objective (§7.3); learning_signal from the coach.
    signal_by_agent = {ac.agent_id: ac.learning_signal for ac in out.agents}
    scores: dict[str, dict[str, float]] = {}
    for st in metrics.agent_stats:
        scores[st.agent_id] = {
            "performance_score": performance_score(st),
            "learning_signal": signal_by_agent.get(st.agent_id, 0.0),
        }

    rationale = out.team_reasoning + (
        "\n" + "\n".join(f"- {ac.agent_id}: [{ac.credit}] {ac.reasoning}" for ac in out.agents)
        if out.agents
        else ""
    )

    proposal = Proposal(
        behavior_cards=new_cards,
        notes=f"coach({source})",
        rationale=rationale,
        scores=scores,
    )

    if telemetry is not None:
        try:
            telemetry.log_event(
                "orca_coach",
                {
                    "episode_idx": metrics.episode_idx,
                    "seed": metrics.seed,
                    "source": source,
                    "rationale": rationale,
                    "edited_agents": list(new_cards.keys()),
                    "card_versions": {aid: c.version for aid, c in new_cards.items()},
                    "scores": scores,
                },
            )
        except Exception:
            pass

    return proposal


# Back-compat alias for the original stub signature.
def coach(trace: EpisodeTrace, metrics: EpisodeMetrics) -> Proposal:
    """Phase-0-compatible entry point (no cards/LLM) — uses the heuristic path."""
    return run_coach(trace, metrics)


__all__ = ["run_coach", "coach", "CoachOutput", "AgentCoaching"]

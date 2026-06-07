"""Worker prompt construction + strict output schema (§4.3) — Stream 2 (A1, A6).

System prompt = ROLE_PRIMER[role] + behavior_card + execution_memory +
ACTION_SPEC + OUTPUT_SCHEMA + NO-LEAK rules. User prompt = the JSON observation +
a compacted history summary + the current assignment + the team DAG frontier.

The worker's strict output is a :class:`WorkerOutput`: exactly one
:class:`~contracts.Action` plus zero-or-more message *drafts*. The frozen
``Action`` / ``Message`` contracts remain the source of truth — message drafts
are intentionally lenient (``from``/``round`` are injected by the worker, never
trusted from the model) and are coerced into canonical ``Message`` objects in
``agents/worker.py`` (§4.4).
"""

from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, Field

from contracts import Action, BehaviorCard, ExecutionMemory, Observation
from contracts.enums import ActionName, MessageType, Role

ROLE_PRIMERS: dict[Role, str] = {
    Role.EXPLORER: "You are the Explorer: scout outward, reveal regions and structures, find biomes.",
    Role.MINER: "You are the Miner: gather cobblestone/coal/iron, manage lava/obsidian logistics.",
    Role.TINKERER: "You are the Tinkerer: craft, smelt, build gear and the nether portal.",
    Role.SUPPORT: "You are Support: food/hunger, combat, shelter, escorting and reviving.",
}

# Compact menu of macro-actions + their typical args (§3.3). The env — not the
# model — is the source of truth for *validity*; the worker only *chooses*.
ACTION_MENU: dict[ActionName, str] = {
    ActionName.MOVE: 'args: {"direction": one of N/NE/E/SE/S/SW/W/NW}',
    ActionName.SCOUT: "args: {} — reveal adjacent regions/biomes",
    ActionName.GATHER: 'args: {"resource": "<name>"} — needs the right tool/biome',
    ActionName.CRAFT: 'args: {"item": "<name>"} — needs inputs + table',
    ActionName.SMELT: 'args: {"item": "<name>"} — needs a furnace + fuel',
    ActionName.PLACE: 'args: {"item": "<item>"}',
    ActionName.FIGHT: 'args: {"target": "<mob>"}',
    ActionName.EAT: 'args: {"item": "<food>"}',
    ActionName.SLEEP: "args: {} — pass the night safely",
    ActionName.GIVE_ITEM: 'args: {"agent": "agent_k", "item": "<name>", "n": <int>}',
    ActionName.REQUEST_HELP: 'args: {"content": "<what you need>", "urgency": 0..1, "to": "team|agent_k"}',
    ActionName.REGROUP: 'args: {"agent": "agent_k"}',
    ActionName.REPORT: 'args: {"content": "<status>", "urgency": 0..1, "to": "team|agent_k"}',
    ActionName.WAIT: "args: {} — pass this turn (counts as idle)",
}

# Stated once to the model: act on the env's feedback instead of repeating a
# rejected action. This is what breaks the craft-loop deadlock — an uninformed
# worker re-issued the same invalid craft every round (§3.3, §4.4).
SELF_CORRECT_RULE = (
    "SELF-CORRECT: if LAST ACTION shows valid=false, DO NOT repeat that action. "
    "Read its reason and satisfy the missing prerequisite first (e.g. a furnace "
    "needs 8 cobblestone; a crafting_table needs 4 planks; planks come from wood). "
    "Use canonical item names (planks, sticks, crafting_table, wooden_pickaxe)."
)

# The one place the no-coordinate-leak rule is stated to the model (§3.2, §4.5).
NO_LEAK_RULES = (
    "NEVER write coordinates, numeric positions, exact distances (e.g. '40 blocks'), "
    "internal region ids (e.g. r_07), the word 'pos', or unique seed-specific landmark "
    "names in any action arg, message, or memory. Refer to places only by biome type, "
    "compass bearing (N/NE/.../NW), and coarse distance band (SAME_REGION/ADJACENT/NEAR/FAR)."
)


class DraftMessage(BaseModel):
    """A model-emitted message draft (§4.3).

    Lenient on purpose: ``from``/``round`` are injected by the worker (never
    trusted from the model) and ``type`` is a plain string coerced into
    :class:`~contracts.enums.MessageType` during sanitization. The canonical,
    validated object is a frozen :class:`~contracts.Message`.
    """

    model_config = ConfigDict(extra="ignore")

    to: str = "team"
    type: str = "report"
    content: str = ""
    urgency: float = 0.3


class WorkerOutput(BaseModel):
    """Strict worker output (§4.3): exactly one Action + zero-or-more messages.

    ``reasoning`` is logged to Weave but never stored to memory (§4.3). ``action``
    is the frozen :class:`~contracts.Action` contract (the source of truth);
    ``messages`` are lenient drafts coerced to ``Message`` by the worker.
    """

    model_config = ConfigDict(extra="ignore")

    reasoning: str = ""
    action: Action
    messages: list[DraftMessage] = Field(default_factory=list)


class DraftHeuristic(BaseModel):
    """A lenient HOW-TO heuristic draft for the episode-end memory write (§4.5)."""

    model_config = ConfigDict(extra="ignore")

    condition: str = ""
    action: str = ""
    confidence: float = 0.5


class MemoryProposal(BaseModel):
    """The model's episode-end memory proposal: transferable HOW-TO only (§4.5)."""

    model_config = ConfigDict(extra="ignore")

    heuristics: list[DraftHeuristic] = Field(default_factory=list)


def _action_spec() -> str:
    lines = [f"  - {name.value}: {hint}" for name, hint in ACTION_MENU.items()]
    return "ACTION MENU (the env decides validity; an illegal choice just loses the turn):\n" + "\n".join(lines)


def _output_schema_spec() -> str:
    example = {
        "reasoning": "one or two short sentences (logged, never stored to memory)",
        "action": {"name": "gather", "args": {"resource": "iron_ore"}},
        "messages": [
            {
                "to": "team",
                "type": "share_finding",
                "content": "iron-rich caves to the N",
                "urgency": 0.4,
            }
        ],
    }
    return (
        "OUTPUT — respond with a SINGLE strict JSON object and nothing else:\n"
        + json.dumps(example, indent=2)
        + "\nRules: exactly one action; messages may be []; message.type is one of "
        + "/".join(m.value for m in MessageType)
        + "; message.to is 'team', 'orca', or an agent id. Do not add other top-level keys."
    )


def _card_block(card: BehaviorCard | None) -> str:
    if card is None:
        return "BEHAVIOR CARD: (none)"
    lines = [
        "BEHAVIOR CARD (Orca-authored WHO + coaching):",
        f"  assignment: {card.assignment or '(none)'}",
    ]
    if card.directives:
        lines.append("  directives:")
        lines += [f"    - {d}" for d in card.directives]
    if card.priorities:
        lines.append("  priorities:")
        lines += [f"    - {p}" for p in card.priorities]
    if card.donts:
        lines.append("  donts:")
        lines += [f"    - {d}" for d in card.donts]
    return "\n".join(lines)


def _memory_block(memory: ExecutionMemory | None) -> str:
    if memory is None or not memory.heuristics:
        return "EXECUTION MEMORY (your own transferable HOW-TO heuristics): (empty)"
    lines = ["EXECUTION MEMORY (your own transferable HOW-TO heuristics):"]
    for h in memory.heuristics:
        lines.append(f"  - if {h.condition} -> {h.action} (confidence {h.confidence:.2f})")
    return "\n".join(lines)


def _last_action_line(obs: Observation) -> str:
    """A salient one-liner on the previous action's outcome (§3.3/§4.4).

    Rejected actions are called out loudly (with the env's reason) so the worker
    fixes the prerequisite instead of repeating the illegal action — the obs JSON
    already carries ``last_action``, but a plain-text callout is far more reliably
    acted on by the model."""
    la = getattr(obs, "last_action", None)
    if la is None:
        return "LAST ACTION: (none yet)"
    if la.valid:
        return f"LAST ACTION: {la.name} -> ok"
    return f"LAST ACTION: {la.name} -> REJECTED ({la.reason or 'invalid'}). Do NOT repeat it; fix the cause first."


def build_worker_prompt(
    obs: Observation,
    card: BehaviorCard,
    memory: ExecutionMemory,
    history_summary: str = "",
) -> str:
    """Build the full worker prompt (§4.3).

    Returns a single deterministic string with a SYSTEM section (role primer +
    behavior card + execution memory + action menu + output schema + no-leak
    rules) followed by a USER section (the coordinate-free JSON observation +
    compact history summary + assignment + team DAG frontier). Kept compact and
    deterministic so traces diff cleanly.
    """
    role = card.role if card is not None else (obs.self_view.role if obs else Role.MINER)
    primer = ROLE_PRIMERS.get(role, "You are a worker agent on a cooperative team.")
    name = card.agent_id if card is not None else ""
    if name:
        primer = f"You are {name}. " + primer

    system = "\n\n".join(
        [
            "== SYSTEM ==",
            primer,
            _card_block(card),
            _memory_block(memory),
            _action_spec(),
            _output_schema_spec(),
            "NO-LEAK RULES: " + NO_LEAK_RULES,
            SELF_CORRECT_RULE,
        ]
    )

    obs_json = json.dumps(obs.model_dump(mode="json", by_alias=True), indent=2)
    assignment = (card.assignment if card is not None else "") or obs.assignment or "(none)"
    user = "\n\n".join(
        [
            "== USER ==",
            "OBSERVATION (coordinate-free; the only thing you can perceive):",
            obs_json,
            _last_action_line(obs),
            f"CURRENT ASSIGNMENT: {assignment}",
            f"TEAM DAG FRONTIER REACHED: {obs.dag_frontier_reached}",
            "HISTORY SUMMARY (older turns compacted; message content is never truncated):\n"
            + (history_summary or "(none yet)"),
            "Choose exactly one action that advances the team DAG frontier, plus any messages. "
            "Respond with strict JSON only.",
        ]
    )
    return system + "\n\n" + user


def build_repair_prompt(original_prompt: str, bad_output: str, error: str) -> str:
    """The one-shot repair prompt (§4.4): re-ask with the bad output + error."""
    return (
        original_prompt
        + "\n\n== REPAIR ==\n"
        + "Your previous response could not be parsed/validated. Fix it.\n"
        + "PREVIOUS OUTPUT:\n"
        + (bad_output or "(empty)")
        + "\n\nVALIDATION ERROR:\n"
        + (error or "(unknown)")
        + "\n\nReturn ONLY corrected strict JSON matching the schema — no prose, no code fences."
    )


def build_memory_prompt(
    card: BehaviorCard,
    memory: ExecutionMemory,
    episode_digest: str,
) -> str:
    """Episode-end memory-write prompt (§4.5): ask ONLY for transferable HOW-TO."""
    role = card.role if card is not None else Role.MINER
    primer = ROLE_PRIMERS.get(role, "You are a worker agent on a cooperative team.")
    example = {
        "heuristics": [
            {
                "condition": "need iron but only have a wooden pickaxe",
                "action": "craft a stone pickaxe first",
                "confidence": 0.8,
            }
        ]
    }
    return "\n\n".join(
        [
            "== SYSTEM ==",
            primer,
            "Write ONLY transferable HOW-TO heuristics that would help on ANY seed: "
            "biome->resource rules, action ordering, tool prerequisites. "
            "No seed-specific facts, no narration.",
            "NO-LEAK RULES: " + NO_LEAK_RULES,
            _memory_block(memory),
            "== USER ==",
            "EPISODE DIGEST:\n" + (episode_digest or "(none)"),
            "Respond with a SINGLE strict JSON object and nothing else:\n"
            + json.dumps(example, indent=2)
            + "\nEach heuristic: {condition, action, confidence 0..1}. Keep at most 8.",
        ]
    )


def compact_history(lines: list[str], keep: int = 6) -> str:
    """Compact a list of per-round history lines into a bounded summary (§4.3/§5.2).

    Keeps the last ``keep`` detailed lines verbatim and collapses everything
    older into a single "(+N earlier rounds)" prefix so history stays bounded.
    """
    if not lines:
        return ""
    if len(lines) <= keep:
        return " | ".join(lines)
    older = len(lines) - keep
    recent = lines[-keep:]
    return f"(+{older} earlier rounds) " + " | ".join(recent)


__all__ = [
    "ROLE_PRIMERS",
    "ACTION_MENU",
    "NO_LEAK_RULES",
    "SELF_CORRECT_RULE",
    "DraftMessage",
    "WorkerOutput",
    "DraftHeuristic",
    "MemoryProposal",
    "build_worker_prompt",
    "build_repair_prompt",
    "build_memory_prompt",
    "compact_history",
]

"""Action resolution + validity enforcement (§3.3, §3.4, §4.4).

The env — not the LLM — is the source of truth for validity. Invalid actions
(craft without inputs, gather without the tool-gate, give_item with no one
present, move toward nothing) are **rejected, produce no effect, and are logged
as ``invalid_action`` with a reason** (§3.3). Invalid-action rate is a tracked
metric (§7.2, §10).

Phase 0 implements the shallow action set the oracle needs — move, scout,
gather, craft, wait, report/request_help — plus graceful rejection of the deeper
actions (smelt/place/fight/eat/sleep/give_item/regroup) that Stream 1 (E1, E4,
E5) will fill in.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Optional

from contracts import Action, ActionRecord, Message
from contracts.enums import ActionName, Bearing, MessageType

from . import techtree
from .world import AgentState, World

# Actions deferred to Stream 1; rejected (validly logged) in the stub.
_UNSUPPORTED = {
    ActionName.SMELT,
    ActionName.PLACE,
    ActionName.FIGHT,
    ActionName.EAT,
    ActionName.SLEEP,
    ActionName.GIVE_ITEM,
    ActionName.REGROUP,
}

# Macro-actions that count as "idle" for the idle-fraction penalty (§7.2).
IDLE_ACTIONS = {ActionName.WAIT}


@dataclass
class Resolution:
    """Outcome of resolving one action."""

    record: ActionRecord
    message: Optional[Message] = None


def _rec(
    round_idx: int,
    agent: AgentState,
    action: Action,
    valid: bool,
    reason: Optional[str] = None,
    result: Optional[dict] = None,
) -> ActionRecord:
    return ActionRecord(
        round=round_idx,
        agent_id=agent.agent_id,
        action=action,
        valid=valid,
        reason=reason,
        result=result or {},
    )


def _gather_yield(rng: Random, abundance: float) -> int:
    """Deterministic gather yield, always >= 1 when valid (§3.5, simplified)."""
    max_yield = max(1, round(3.0 * abundance))
    return rng.randint(1, max_yield)


def resolve_action(
    world: World,
    agent: AgentState,
    action: Action,
    rng: Random,
    round_idx: int,
) -> Resolution:
    """Resolve one action against world state; return record (+ optional message)."""
    name = action.name
    args = action.args
    region = world.regions[agent.region_id]

    # --- wait -------------------------------------------------------------- #
    if name == ActionName.WAIT:
        return Resolution(_rec(round_idx, agent, action, True, result={"idle": True}))

    # --- report / request_help (emit a structured bus message) ------------- #
    if name in (ActionName.REPORT, ActionName.REQUEST_HELP):
        content = str(args.get("content", args.get("status", "")))
        urgency = float(args.get("urgency", 0.3))
        msg_type = (
            MessageType.REQUEST_HELP
            if name == ActionName.REQUEST_HELP
            else MessageType.REPORT
        )
        msg = Message(
            **{"from": agent.agent_id},
            to=str(args.get("to", "team")),
            type=msg_type,
            content=content,
            urgency=max(0.0, min(1.0, urgency)),
            round=round_idx,
        )
        rec = _rec(round_idx, agent, action, True, result={"sent_message": True})
        return Resolution(rec, message=msg)

    # --- move -------------------------------------------------------------- #
    if name == ActionName.MOVE:
        raw = args.get("direction") or args.get("dir")
        if raw is None:
            return Resolution(_rec(round_idx, agent, action, False, "move needs a direction"))
        try:
            direction = Bearing(raw)
        except ValueError:
            return Resolution(_rec(round_idx, agent, action, False, f"bad direction '{raw}'"))
        dest = world.resolve_move(agent.region_id, direction)
        if dest is None:
            return Resolution(
                _rec(round_idx, agent, action, False, f"no region toward {direction.value}")
            )
        agent.region_id = dest
        world.regions[dest].discovered = True
        arrived = world.regions[dest]
        return Resolution(
            _rec(
                round_idx,
                agent,
                action,
                True,
                result={"moved_dir": direction.value, "arrived_biome": arrived.biome.value},
            )
        )

    # --- scout ------------------------------------------------------------- #
    if name == ActionName.SCOUT:
        region.discovered = True
        revealed = []
        for rid, _dist in world.neighbors(agent.region_id):
            world.regions[rid].discovered = True
            revealed.append(world.regions[rid].biome.value)
        return Resolution(
            _rec(round_idx, agent, action, True, result={"scouted_biomes": revealed})
        )

    # --- gather ------------------------------------------------------------ #
    if name == ActionName.GATHER:
        resource = str(args.get("resource", ""))
        abundance = region.resources.get(resource, 0.0)
        if abundance <= 0.0:
            return Resolution(
                _rec(round_idx, agent, action, False, f"no {resource or '?'} in {region.biome.value}")
            )
        if not techtree.gather_tool_ok(resource, agent.inventory):
            gate = techtree.GATHER_GATES.get(resource, "tool")
            return Resolution(
                _rec(round_idx, agent, action, False, f"need {gate} to gather {resource}")
            )
        n = _gather_yield(rng, abundance)
        agent.inventory[resource] = agent.inventory.get(resource, 0) + n
        return Resolution(
            _rec(round_idx, agent, action, True, result={"gathered": {resource: n}})
        )

    # --- craft ------------------------------------------------------------- #
    if name == ActionName.CRAFT:
        item = str(args.get("item", ""))
        ok, reason = techtree.craft_check(item, agent.inventory)
        if not ok:
            return Resolution(_rec(round_idx, agent, action, False, reason))
        recipe = techtree.RECIPES[item]
        for nm, qty in recipe.inputs.items():
            agent.inventory[nm] -= qty
            if agent.inventory[nm] <= 0:
                del agent.inventory[nm]
        for nm, qty in recipe.outputs.items():
            agent.inventory[nm] = agent.inventory.get(nm, 0) + qty
        return Resolution(
            _rec(round_idx, agent, action, True, result={"crafted": dict(recipe.outputs)})
        )

    # --- deferred to Stream 1 --------------------------------------------- #
    if name in _UNSUPPORTED:
        return Resolution(
            _rec(round_idx, agent, action, False, f"action '{name.value}' not supported in stub env")
        )

    return Resolution(_rec(round_idx, agent, action, False, f"unknown action '{name}'"))


__all__ = ["resolve_action", "Resolution", "IDLE_ACTIONS"]

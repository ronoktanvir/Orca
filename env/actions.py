"""Action resolution + validity enforcement (§3.3, §3.4, §4.4).

The env — not the LLM — is the source of truth for validity. Invalid actions
(craft without inputs, gather without the tool-gate, give_item with no one
present, move toward nothing) are **rejected, produce no effect, and are logged
as ``invalid_action`` with a reason** (§3.3). Invalid-action rate is a tracked
metric (§7.2, §10).

Phase 0 implemented the shallow action set the oracle needs — move, scout,
gather, craft, wait, report/request_help. Stream 1 E1 adds ``smelt`` and
``place`` (below); ``fight``/``eat``/``sleep``/``give_item``/``regroup`` stay
gracefully rejected until E4 (survival) / E5 (co-op).
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Optional

from contracts import Action, ActionRecord, Message
from contracts.enums import ActionName, Bearing, Biome, Layer, Milestone, MessageType, Structure

from . import techtree
from .world import AgentState, World

# Actions still deferred (rejected + validly logged). E1 added SMELT + PLACE; E3
# a deterministic FIGHT; E4 adds EAT (below). SLEEP/GIVE_ITEM/REGROUP land in E4
# (sleep) / E5 (co-op).
_UNSUPPORTED = {
    ActionName.SLEEP,
    ActionName.GIVE_ITEM,
    ActionName.REGROUP,
}

# Hunger restored per cooked_food eaten (§3.5, E4). EAT consumes one cooked_food.
_EAT_RESTORE = 0.5

# Mob -> (drop item, location predicate, human reason if location is wrong).
# E3-minimal: deterministic, single-agent, always-succeeds. E4 adds stochastic
# (~Binomial) yields + risk; E5 makes blaze/dragon superadditive (co-located gear).
def _fight_blaze_ok(region) -> bool:
    return region.structure == Structure.FORTRESS


def _fight_enderman_ok(region) -> bool:
    return region.layer == Layer.NETHER or region.biome == Biome.WARPED_FOREST


def _fight_dragon_ok(region) -> bool:
    return region.layer == Layer.END


# Superadditive combat (§3.5, E5): for blaze + the ender_dragon, success rises
# with the number of co-located teammates — solo is unreliable (~0.2), a co-located
# trio wins reliably (~0.85). This is THE cooperation incentive ("send a pair to
# the fortress / bring the team to the End"), so delegation matters. Endermen are
# a regular mob (deterministic). E4 will fold in gear/health terms; here it is a
# simple linear curve drawn from the existing seeded RNG (deterministic + logged).
_SUPERADDITIVE_TARGETS = {"blaze", "ender_dragon", "dragon"}
_FIGHT_SOLO_P = 0.2  # solo success
_FIGHT_ALLY_GAIN = 0.325  # per extra co-located ally: solo .20 -> pair .525 -> trio .85


def _n_colocated(world: World, agent: AgentState) -> int:
    """Alive agents sharing the agent's region (SAME_REGION), including itself."""
    rid = agent.region_id
    return sum(1 for a in world.agents.values() if a.alive and a.region_id == rid)


def _fight_success_p(n_colocated: int) -> float:
    return max(0.05, min(0.95, _FIGHT_SOLO_P + _FIGHT_ALLY_GAIN * (n_colocated - 1)))


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
        raw = args.get("direction") or args.get("dir") or args.get("to")
        # Cross-layer portal travel (§3.1) — explicit, since neighbors never cross
        # layers. Triggered by a portal/layer keyword instead of a compass bearing.
        if isinstance(raw, str) and raw.lower() in ("portal", "nether", "end", "overworld"):
            dest = world.portal_destination(agent.region_id)
            if dest is None:
                return Resolution(_rec(round_idx, agent, action, False, "no active portal here"))
            agent.region_id = dest
            world.enter(dest)  # discover + record NETHER_ENTERED / END_ENTERED
            arrived = world.regions[dest]
            return Resolution(
                _rec(
                    round_idx, agent, action, True,
                    result={"used_portal": True, "arrived_layer": arrived.layer.value,
                            "arrived_biome": arrived.biome.value},
                )
            )
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
        world.enter(dest)  # discover + record any arrival/structure milestone
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
        world.discover(region.id)
        revealed = []
        for rid, _dist in world.neighbors(agent.region_id):
            world.discover(rid)  # reveals biome + any structure milestone
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

    # --- smelt (furnace + fuel: ore -> ingot, raw food -> cooked; §3.4) ----- #
    if name == ActionName.SMELT:
        item = str(args.get("item", args.get("resource", "")))
        ok, reason = techtree.smelt_check(item, agent.inventory)
        if not ok:
            return Resolution(_rec(round_idx, agent, action, False, reason))
        output = techtree.SMELTS[item]
        # Consume one input + one unit of fuel; the furnace persists (prerequisite).
        agent.inventory[item] -= 1
        if agent.inventory[item] <= 0:
            del agent.inventory[item]
        agent.inventory[techtree.SMELT_FUEL] -= 1
        if agent.inventory[techtree.SMELT_FUEL] <= 0:
            del agent.inventory[techtree.SMELT_FUEL]
        agent.inventory[output] = agent.inventory.get(output, 0) + 1
        return Resolution(
            _rec(round_idx, agent, action, True, result={"smelted": {output: 1}})
        )

    # --- place (deploy a block / light a portal into the world; §3.1, §3.4) - #
    if name == ActionName.PLACE:
        item = str(args.get("item", args.get("block", "")))
        if not item:
            return Resolution(_rec(round_idx, agent, action, False, "place needs an item"))

        # Lighting a nether portal links this Overworld region to the Nether entry.
        if item == "nether_portal":
            if agent.inventory.get("nether_portal", 0) <= 0:
                return Resolution(_rec(round_idx, agent, action, False, "no nether_portal to place"))
            if region.layer != Layer.OVERWORLD:
                return Resolution(
                    _rec(round_idx, agent, action, False, "a nether portal must be lit in the Overworld")
                )
            ok, reason = world.light_nether_portal(region.id)
            if not ok:
                return Resolution(_rec(round_idx, agent, action, False, reason))
            agent.inventory["nether_portal"] -= 1
            if agent.inventory["nether_portal"] <= 0:
                del agent.inventory["nether_portal"]
            world.world_milestones.add(Milestone.PORTAL_BUILT)
            return Resolution(
                _rec(round_idx, agent, action, True, result={"lit_portal": "nether"})
            )

        # Activating the End portal (only in the stronghold) links to the End.
        if item == "end_portal":
            if agent.inventory.get("end_portal", 0) <= 0:
                return Resolution(_rec(round_idx, agent, action, False, "no end_portal to place"))
            ok, reason = world.activate_end_portal(region.id)
            if not ok:
                return Resolution(_rec(round_idx, agent, action, False, reason))
            agent.inventory["end_portal"] -= 1
            if agent.inventory["end_portal"] <= 0:
                del agent.inventory["end_portal"]
            world.world_milestones.add(Milestone.END_PORTAL_ACTIVE)
            return Resolution(
                _rec(round_idx, agent, action, True, result={"activated_end_portal": True})
            )

        # Otherwise: a plain building block (e.g. obsidian for a portal frame).
        if item not in techtree.PLACEABLE_BLOCKS:
            return Resolution(
                _rec(round_idx, agent, action, False, f"cannot place '{item}'")
            )
        if agent.inventory.get(item, 0) <= 0:
            return Resolution(
                _rec(round_idx, agent, action, False, f"no {item} to place")
            )
        agent.inventory[item] -= 1
        if agent.inventory[item] <= 0:
            del agent.inventory[item]
        # TODO E2+: track placed frame blocks in region world-state (count to 10).
        return Resolution(_rec(round_idx, agent, action, True, result={"placed": item}))

    # --- fight (E5: superadditive co-op for blaze/dragon; §3.5) ------------- #
    if name == ActionName.FIGHT:
        target = str(args.get("target", args.get("mob", ""))).lower()

        # 1) Location gating — reject (invalid + reason) if the target isn't here.
        if target == "blaze":
            if not _fight_blaze_ok(region):
                return Resolution(_rec(round_idx, agent, action, False, "blaze are only in a fortress"))
        elif target == "enderman":
            if not _fight_enderman_ok(region):
                return Resolution(_rec(round_idx, agent, action, False, "no enderman here"))
        elif target in ("ender_dragon", "dragon"):
            if not _fight_dragon_ok(region):
                return Resolution(_rec(round_idx, agent, action, False, "the dragon is only in the End"))
        else:
            return Resolution(_rec(round_idx, agent, action, False, f"cannot fight '{target or '?'}'"))

        # 2) Success model. Blaze + dragon are SUPERADDITIVE: a valid attempt may
        #    fail (solo ~0.2, trio ~0.85) — co-location decides it. A failed attempt
        #    is still a *valid* action (no drop), not an invalid one.
        n_colocated = _n_colocated(world, agent)
        if target in _SUPERADDITIVE_TARGETS:
            if rng.random() >= _fight_success_p(n_colocated):
                return Resolution(
                    _rec(round_idx, agent, action, True,
                         result={"target": target, "defeated": False, "n_colocated": n_colocated})
                )

        # 3) Success -> drop / win.
        if target == "blaze":
            agent.inventory["blaze_rod"] = agent.inventory.get("blaze_rod", 0) + 1
            return Resolution(
                _rec(round_idx, agent, action, True,
                     result={"defeated": "blaze", "drop": {"blaze_rod": 1}, "n_colocated": n_colocated})
            )
        if target == "enderman":
            agent.inventory["ender_pearl"] = agent.inventory.get("ender_pearl", 0) + 1
            return Resolution(
                _rec(round_idx, agent, action, True,
                     result={"defeated": "enderman", "drop": {"ender_pearl": 1}})
            )
        # ender_dragon
        world.world_milestones.add(Milestone.DRAGON_DEFEATED)
        return Resolution(
            _rec(round_idx, agent, action, True,
                 result={"defeated": "ender_dragon", "win": True, "n_colocated": n_colocated})
        )

    # --- eat (E4: restore hunger by consuming cooked_food; §3.5) ------------ #
    if name == ActionName.EAT:
        if agent.inventory.get("cooked_food", 0) <= 0:
            return Resolution(_rec(round_idx, agent, action, False, "no cooked_food to eat"))
        agent.inventory["cooked_food"] -= 1
        if agent.inventory["cooked_food"] <= 0:
            del agent.inventory["cooked_food"]
        agent.hunger = min(1.0, agent.hunger + _EAT_RESTORE)
        return Resolution(
            _rec(round_idx, agent, action, True, result={"ate": "cooked_food", "hunger": agent.hunger})
        )

    # --- deferred to E4 (sleep) / E5 (co-op) ------------------------------ #
    if name in _UNSUPPORTED:
        return Resolution(
            _rec(round_idx, agent, action, False, f"action '{name.value}' not supported in stub env")
        )

    return Resolution(_rec(round_idx, agent, action, False, f"unknown action '{name}'"))


__all__ = ["resolve_action", "Resolution", "IDLE_ACTIONS"]

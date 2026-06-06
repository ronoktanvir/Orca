"""Scripted shallow oracle (F4 / §12).

A deterministic, reactive policy that reaches the IRON milestone on the stub
seed. It is *not* an LLM — it reads the coordinate-free observation and returns
the single next action that advances the wood -> wooden tools -> stone tools ->
iron chain, building prerequisites on demand. This is the Phase 0 placeholder
agent; Stream 2 (A1) replaces it with the real LLM worker, and Stream 1 (E3)
ships the full-DAG oracle that reaches the dragon.

The policy is purely a function of the observation (no hidden state), which keeps
it deterministic and easy to reason about. Movement is by compass heading toward
biome *hints* — never coordinates.
"""

from __future__ import annotations

from contracts import Action, Observation
from contracts.enums import ActionName, Bearing, Biome

# Biomes where the oracle can mine cobblestone / iron.
_MINING_BIOMES = {Biome.MOUNTAINS.value, Biome.CAVES.value}
_WOOD_BIOMES = {Biome.FOREST.value, Biome.PLAINS.value}


def _wait() -> Action:
    return Action(name=ActionName.WAIT)


def _move(direction: Bearing) -> Action:
    return Action(name=ActionName.MOVE, args={"direction": direction.value})


def _move_toward(obs: Observation, target_biomes: set[str]) -> Action:
    """Head toward a target biome by hint; scout first if neighbors are unknown,
    otherwise explore along a frontier heading. Never uses coordinates."""
    here = obs.here
    # 1. A discovered exit already hints at the biome we want -> go there.
    for ex in here.exits:
        if ex.biome_hint.value in target_biomes:
            return _move(ex.dir)
    # 2. Unknown neighbors remain -> scout to reveal their biomes.
    if any(ex.biome_hint == Biome.UNKNOWN for ex in here.exits):
        return Action(name=ActionName.SCOUT)
    # 3. Explore along a frontier heading.
    if here.frontier_dirs:
        return _move(here.frontier_dirs[0])
    # 4. Nothing to do but scout again.
    return Action(name=ActionName.SCOUT)


class ShallowOracle:
    """Deterministically reaches IRON on the stub seed."""

    def __init__(self, agent_id: str = "agent_1") -> None:
        self.agent_id = agent_id

    def act(self, obs: Observation) -> Action:
        inv = obs.self_view.inventory
        biome = obs.self_view.current_biome.value
        here_resources = set(obs.here.resources_visible)

        def have(item: str, n: int = 1) -> bool:
            return inv.get(item, 0) >= n

        # Terminal: iron acquired -> the env ends this round; report it.
        if have("iron_ore"):
            return Action(
                name=ActionName.REPORT,
                args={"content": "iron acquired", "urgency": 0.5},
            )

        # Tooled for iron? Go mine it.
        if have("stone_pickaxe"):
            if "iron_ore" in here_resources:
                return Action(name=ActionName.GATHER, args={"resource": "iron_ore"})
            return _move_toward(obs, _MINING_BIOMES)

        # --- build the stone pickaxe (3 cobblestone + 2 sticks + table) ----- #
        if have("cobblestone", 3) and have("sticks", 2) and have("crafting_table"):
            return Action(name=ActionName.CRAFT, args={"item": "stone_pickaxe"})

        # Need cobblestone -> requires a wooden pickaxe + a mining biome.
        if not have("cobblestone", 3) and have("wooden_pickaxe"):
            if "cobblestone" in here_resources:
                return Action(name=ActionName.GATHER, args={"resource": "cobblestone"})
            return _move_toward(obs, _MINING_BIOMES)

        # Sticks for the stone pickaxe.
        if not have("sticks", 2) and have("planks", 2):
            return Action(name=ActionName.CRAFT, args={"item": "sticks"})

        # --- build the wooden tools / basics -------------------------------- #
        if not have("crafting_table") and have("planks", 4):
            return Action(name=ActionName.CRAFT, args={"item": "crafting_table"})

        if (
            not have("wooden_pickaxe")
            and have("planks", 3)
            and have("sticks", 2)
            and have("crafting_table")
        ):
            return Action(name=ActionName.CRAFT, args={"item": "wooden_pickaxe"})

        # Keep planks stocked while wood is available.
        if not have("planks", 4) and have("wood", 1):
            return Action(name=ActionName.CRAFT, args={"item": "planks"})

        # Gather wood where it grows.
        if "wood" in here_resources and not have("wood", 3):
            return Action(name=ActionName.GATHER, args={"resource": "wood"})

        # Need wood but none here -> head to a wood biome.
        if not have("wood", 1) and not have("planks", 1) and biome not in _WOOD_BIOMES:
            return _move_toward(obs, _WOOD_BIOMES)

        # Default: make progress toward the mining biome.
        return _move_toward(obs, _MINING_BIOMES)


__all__ = ["ShallowOracle"]

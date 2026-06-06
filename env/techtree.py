"""Shallow tech tree for the Phase 0 stub env (subset of §3.4).

Covers the ``wood -> stone tools -> iron`` slice only — exactly enough for the
scripted oracle to reach the IRON milestone (§12 / F4). Stream 1 (E1) deepens
this all the way to the dragon. Recipes, tool-gates and the biome->resource
*type* map are shared across seeds (§3.7); only abundances vary per seed.
"""

from __future__ import annotations

from contracts.enums import Biome, Milestone

# --------------------------------------------------------------------------- #
# Canonical identifier sets (item/resource names are plain strings in contracts)
# --------------------------------------------------------------------------- #
RESOURCES: set[str] = {
    "wood",
    "food",
    "cobblestone",
    "coal",
    "iron_ore",
    "diamond",
    "sand",
}

# --------------------------------------------------------------------------- #
# Biome -> resource *types* present (abundance is per-seed; §3.4)
# --------------------------------------------------------------------------- #
BIOME_RESOURCES: dict[Biome, set[str]] = {
    Biome.FOREST: {"wood", "food"},
    Biome.PLAINS: {"food", "wood"},
    Biome.MOUNTAINS: {"cobblestone", "coal", "iron_ore"},
    Biome.CAVES: {"cobblestone", "coal", "iron_ore", "diamond"},
    Biome.DESERT: {"sand"},
}

# --------------------------------------------------------------------------- #
# Tool gates: gathering a resource requires (>=) a pickaxe tier (§3.4)
# --------------------------------------------------------------------------- #
# Pickaxe tiers, ascending; a higher tier satisfies a lower gate.
PICKAXE_TIERS: list[str] = [
    "wooden_pickaxe",
    "stone_pickaxe",
    "iron_pickaxe",
    "diamond_pickaxe",
]

# resource -> minimum pickaxe required (absent => no tool needed)
GATHER_GATES: dict[str, str] = {
    "cobblestone": "wooden_pickaxe",
    "coal": "wooden_pickaxe",
    "iron_ore": "stone_pickaxe",
    "diamond": "iron_pickaxe",
}


def _tier_index(tool: str) -> int:
    return PICKAXE_TIERS.index(tool) if tool in PICKAXE_TIERS else -1


def best_pickaxe(inventory: dict[str, int]) -> str | None:
    """Highest-tier pickaxe present in an inventory, or None."""
    have = [t for t in PICKAXE_TIERS if inventory.get(t, 0) > 0]
    return have[-1] if have else None


def gather_tool_ok(resource: str, inventory: dict[str, int]) -> bool:
    """Does this inventory satisfy the tool-gate to gather ``resource``?"""
    gate = GATHER_GATES.get(resource)
    if gate is None:
        return True
    best = best_pickaxe(inventory)
    return best is not None and _tier_index(best) >= _tier_index(gate)


# --------------------------------------------------------------------------- #
# Recipes (craft). gate_item = an item that must be present (e.g. crafting_table)
# but is NOT consumed. inputs are consumed; outputs are produced.
# --------------------------------------------------------------------------- #
class Recipe:
    __slots__ = ("item", "inputs", "outputs", "requires")

    def __init__(
        self,
        item: str,
        inputs: dict[str, int],
        outputs: dict[str, int],
        requires: tuple[str, ...] = (),
    ) -> None:
        self.item = item
        self.inputs = inputs
        self.outputs = outputs
        self.requires = requires  # non-consumed prerequisites (e.g. crafting_table)


RECIPES: dict[str, Recipe] = {
    "planks": Recipe("planks", {"wood": 1}, {"planks": 4}),
    "sticks": Recipe("sticks", {"planks": 2}, {"sticks": 4}),
    "crafting_table": Recipe("crafting_table", {"planks": 4}, {"crafting_table": 1}),
    "wooden_pickaxe": Recipe(
        "wooden_pickaxe",
        {"planks": 3, "sticks": 2},
        {"wooden_pickaxe": 1},
        requires=("crafting_table",),
    ),
    "stone_pickaxe": Recipe(
        "stone_pickaxe",
        {"cobblestone": 3, "sticks": 2},
        {"stone_pickaxe": 1},
        requires=("crafting_table",),
    ),
    "furnace": Recipe(
        "furnace",
        {"cobblestone": 8},
        {"furnace": 1},
        requires=("crafting_table",),
    ),
}


def craft_check(item: str, inventory: dict[str, int]) -> tuple[bool, str | None]:
    """Can ``item`` be crafted from ``inventory``? Returns (ok, reason_if_not)."""
    recipe = RECIPES.get(item)
    if recipe is None:
        return False, f"no recipe for '{item}'"
    for req in recipe.requires:
        if inventory.get(req, 0) <= 0:
            return False, f"requires {req}"
    for name, qty in recipe.inputs.items():
        if inventory.get(name, 0) < qty:
            return False, f"need {qty} {name} (have {inventory.get(name, 0)})"
    return True, None


# --------------------------------------------------------------------------- #
# Milestone detection from a *pooled* (team) inventory (frontier is team-level)
# --------------------------------------------------------------------------- #
# Shallow ladder reached by the stub, ordered shallow -> deep.
STUB_LADDER: list[Milestone] = [
    Milestone.START,
    Milestone.WOOD,
    Milestone.WOODEN_TOOLS,
    Milestone.STONE_TOOLS,
    Milestone.IRON,
]


def detect_milestone(pooled_inventory: dict[str, int]) -> Milestone:
    """Deepest stub milestone implied by a pooled team inventory (max-frontier)."""
    inv = pooled_inventory
    reached = Milestone.START
    if inv.get("wood", 0) > 0:
        reached = Milestone.WOOD
    if inv.get("wooden_pickaxe", 0) > 0:
        reached = Milestone.WOODEN_TOOLS
    if inv.get("stone_pickaxe", 0) > 0:
        reached = Milestone.STONE_TOOLS
    if inv.get("iron_ore", 0) > 0:
        reached = Milestone.IRON
    return reached


__all__ = [
    "RESOURCES",
    "BIOME_RESOURCES",
    "PICKAXE_TIERS",
    "GATHER_GATES",
    "RECIPES",
    "Recipe",
    "STUB_LADDER",
    "best_pickaxe",
    "gather_tool_ok",
    "craft_check",
    "detect_milestone",
]

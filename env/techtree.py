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
    # --- overworld basics ---
    "wood",
    "food",  # raw food (animals/crops) -> smelt to cooked_food
    "cobblestone",
    "coal",
    "iron_ore",
    "diamond",
    "sand",
    "cactus",
    "clay",
    "water",  # river/ocean source — fills a bucket for the obsidian water-trick
    "flint",  # from gravel; pairs with iron_ingot -> flint_and_steel
    "lava_pool",  # near-lava feature; the obsidian source for both routes (§3.4)
    "obsidian",  # route (a): mined from cooled lava with a diamond_pickaxe
    # --- nether ---
    "nether_wart",
    "soul_sand",
    "basalt",
}

# Combat drops — NOT gathered or crafted in E1; obtained by ``fight`` (E5). Listed
# for documentation: recipes below consume them and detect_milestone reads them,
# but no E1 action *produces* them (so tests inject them directly).
FIGHT_DROPS: set[str] = {"blaze_rod", "ender_pearl"}

# --------------------------------------------------------------------------- #
# Biome -> resource *types* present (abundance is per-seed; §3.4)
# --------------------------------------------------------------------------- #
BIOME_RESOURCES: dict[Biome, set[str]] = {
    # --- overworld ---
    Biome.FOREST: {"wood", "food"},
    Biome.JUNGLE: {"wood", "food"},
    Biome.TAIGA: {"wood", "food"},
    Biome.PLAINS: {"food", "wood"},
    Biome.MOUNTAINS: {"cobblestone", "coal", "iron_ore"},
    # deep caves also expose lava_pool + flint(gravel) -> the two obsidian routes
    Biome.CAVES: {"cobblestone", "coal", "iron_ore", "diamond", "lava_pool", "flint"},
    Biome.DESERT: {"sand", "cactus"},
    Biome.SWAMP: {"clay", "wood"},
    Biome.OCEAN: {"food", "water"},
    # --- nether (reached via a lit portal in E2) ---
    Biome.NETHER_WASTES: {"nether_wart"},  # fortress (+blaze via fight) likely here
    Biome.SOUL_SAND_VALLEY: {"soul_sand", "nether_wart"},
    Biome.BASALT_DELTA: {"basalt", "lava_pool"},
    Biome.WARPED_FOREST: set(),  # enderman -> ender_pearl via fight (E5); no passive gather
    # --- structures / end (combat & activation only; no passive gathers) ---
    Biome.STRONGHOLD: set(),
    Biome.END: set(),
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
    # obsidian route (a): mine a cooled lava_pool with a diamond_pickaxe (§3.4).
    # Route (b) — the cheaper bucket water-trick — is the ``obsidian`` RECIPE below.
    "obsidian": "diamond_pickaxe",
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
    "stone_sword": Recipe(
        "stone_sword",
        {"cobblestone": 2, "sticks": 1},
        {"stone_sword": 1},
        requires=("crafting_table",),
    ),
    # --- iron tier (needs iron_ingot from smelting; §3.4) -------------------- #
    "iron_pickaxe": Recipe(
        "iron_pickaxe",
        {"iron_ingot": 3, "sticks": 2},
        {"iron_pickaxe": 1},
        requires=("crafting_table",),
    ),
    "iron_sword": Recipe(
        "iron_sword",
        {"iron_ingot": 2, "sticks": 1},
        {"iron_sword": 1},
        requires=("crafting_table",),
    ),
    "shield": Recipe(
        "shield",
        {"iron_ingot": 1, "planks": 6},
        {"shield": 1},
        requires=("crafting_table",),
    ),
    "bucket": Recipe(
        "bucket",
        {"iron_ingot": 3},
        {"bucket": 1},
        requires=("crafting_table",),
    ),
    "flint_and_steel": Recipe(
        "flint_and_steel",
        {"iron_ingot": 1, "flint": 1},
        {"flint_and_steel": 1},
        requires=("crafting_table",),
    ),
    # --- diamond tier (diamond is iron_pickaxe-gated in GATHER_GATES) -------- #
    "diamond_pickaxe": Recipe(
        "diamond_pickaxe",
        {"diamond": 3, "sticks": 2},
        {"diamond_pickaxe": 1},
        requires=("crafting_table",),
    ),
    "diamond_sword": Recipe(
        "diamond_sword",
        {"diamond": 2, "sticks": 1},
        {"diamond_sword": 1},
        requires=("crafting_table",),
    ),
    "diamond_armor": Recipe(  # full set = 24 diamonds (§3.4)
        "diamond_armor",
        {"diamond": 24},
        {"diamond_armor": 1},
        requires=("crafting_table",),
    ),
    # --- obsidian, route (b): the CLEVER, cheaper water-trick (reward it!) --- #
    # A water bucket poured over a lava_pool cools it into obsidian. This is
    # reachable at IRON tier (just a bucket) — no diamond_pickaxe needed — so it
    # is strictly cheaper than route (a) (the diamond_pickaxe GATHER_GATE above).
    # Rewarding agents who find this route is rewarding strategy (§3.4).
    "obsidian": Recipe(
        "obsidian",
        {"lava_pool": 1},
        {"obsidian": 1},
        requires=("bucket",),  # the water bucket is reused, not consumed
    ),
    # --- nether portal: 10 obsidian frame, lit with flint_and_steel --------- #
    "nether_portal": Recipe(
        "nether_portal",
        {"obsidian": 10},
        {"nether_portal": 1},
        requires=("flint_and_steel",),  # the lighter is reused, not consumed
    ),
    # --- nether / end consumables ------------------------------------------- #
    "blaze_powder": Recipe(  # blaze_rod is a fortress fight-drop (E5)
        "blaze_powder",
        {"blaze_rod": 1},
        {"blaze_powder": 2},
    ),
    "eye_of_ender": Recipe(  # need ~12 to locate + activate the stronghold portal
        "eye_of_ender",
        {"blaze_powder": 1, "ender_pearl": 1},
        {"eye_of_ender": 1},
    ),
    "end_portal": Recipe(  # 12 eyes activate the stronghold's frame
        "end_portal",
        {"eye_of_ender": 12},
        {"end_portal": 1},
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
# Smelting (§3.4): a furnace + fuel (coal) converts ore -> ingot and raw food ->
# cooked. The furnace is a non-consumed prerequisite (like crafting_table); one
# unit of fuel is consumed per smelt. Shape mirrors RECIPES so later streams can
# extend it (gold_ore -> gold_ingot, sand -> glass, …) without touching actions.
# --------------------------------------------------------------------------- #
SMELT_FUEL = "coal"

# smeltable input -> output item (1 input + 1 fuel -> 1 output)
SMELTS: dict[str, str] = {
    "iron_ore": "iron_ingot",
    "food": "cooked_food",
}


def smelt_check(item: str, inventory: dict[str, int]) -> tuple[bool, str | None]:
    """Can ``item`` be smelted from ``inventory``? Returns (ok, reason_if_not).

    Needs a furnace (prerequisite, not consumed) + fuel (coal, consumed) + at
    least one of the smeltable input present (§3.4).
    """
    if item not in SMELTS:
        return False, f"cannot smelt '{item or '?'}'"
    if inventory.get("furnace", 0) <= 0:
        return False, "smelting needs a furnace"
    if inventory.get(item, 0) <= 0:
        return False, f"need {item} to smelt (have 0)"
    if inventory.get(SMELT_FUEL, 0) <= 0:
        return False, f"smelting needs {SMELT_FUEL} (fuel)"
    return True, None


# --------------------------------------------------------------------------- #
# Placeable blocks (§3.3 ``place``). E1 is geometry-free, so a placed block is
# consumed + logged; E2 attaches it to region world-state (e.g. an obsidian
# portal frame). Only pure building blocks — tools/ingots/consumables are not
# placeable. Note: crafting_table/furnace are deliberately excluded: the recipe
# system treats them as non-consumed ``requires`` gates, so letting ``place``
# consume them would silently break crafting/smelting (E2 can revisit).
# --------------------------------------------------------------------------- #
PLACEABLE_BLOCKS: set[str] = {
    "obsidian",
    "cobblestone",
    "sand",
    "soul_sand",
    "basalt",
}


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
    """Deepest milestone implied by a pooled team inventory (max-frontier, §7.1).

    E1 detects every milestone whose proof is an *inventory item* obtainable in
    the deepened tech tree. Milestones that are world placement/activation or
    location state (portal lit, Nether/fortress/stronghold/End reached, dragon
    killed) need geometry the stub env doesn't expose yet — they're stubbed below
    and wired in E6. The shallow wood->iron chain is preserved unchanged so the
    Phase-0 oracle still reaches IRON by gathering ``iron_ore``.

    Checks run shallow -> deep; the deepest matching one wins (later overrides).
    """
    inv = pooled_inventory
    reached = Milestone.START

    # --- shallow chain (unchanged; the oracle relies on iron_ore -> IRON) ---- #
    if inv.get("wood", 0) > 0:
        reached = Milestone.WOOD
    if inv.get("wooden_pickaxe", 0) > 0:
        reached = Milestone.WOODEN_TOOLS
    if inv.get("stone_pickaxe", 0) > 0:
        reached = Milestone.STONE_TOOLS
    # IRON: ore in hand (shallow oracle) OR a smelted ingot (deep tree).
    if inv.get("iron_ore", 0) > 0 or inv.get("iron_ingot", 0) > 0:
        reached = Milestone.IRON

    # --- E1 deep tree: inventory-detectable milestones only (§3.4) ----------- #
    # SHIELD_BUCKET: either iron utility item (bucket OR shield). The bucket path
    # lets the cheaper water-trick obsidian route register this tier.
    if inv.get("bucket", 0) > 0 or inv.get("shield", 0) > 0:
        reached = Milestone.SHIELD_BUCKET
    if inv.get("obsidian", 0) > 0:
        reached = Milestone.OBSIDIAN
    if inv.get("blaze_rod", 0) > 0:
        reached = Milestone.BLAZE_RODS
    if inv.get("ender_pearl", 0) > 0:
        reached = Milestone.ENDER_PEARLS
    if inv.get("eye_of_ender", 0) > 0:  # ~12 needed to activate the end portal
        reached = Milestone.EYES_OF_ENDER

    # --- world-state / location milestones: STUBBED; detection lands in E6 --- #
    # These need geometry the stub env doesn't expose yet, and detect_milestone
    # only sees pooled inventory — so they must NOT be faked off inventory tokens
    # (a crafted ``nether_portal``/``end_portal`` token does not imply the portal
    # was placed + lit in the world). They stay craftable (RECIPES) but undetected.
    # TODO E6: wire location/world-state detection for:
    #   PORTAL_BUILT       — nether_portal frame placed (10 obsidian) + lit
    #   NETHER_ENTERED     — an agent has stepped through into Layer.NETHER
    #   FORTRESS_FOUND     — a fortress structure has been discovered
    #   STRONGHOLD_FOUND   — the stronghold structure has been discovered
    #   END_PORTAL_ACTIVE  — 12 eyes placed -> stronghold end portal activated
    #   END_ENTERED        — an agent has entered Layer.END
    #   DRAGON_DEFEATED    — the ender_dragon fight has been won (WIN)
    return reached


_MILESTONE_DEPTH = {m: i for i, m in enumerate(Milestone)}


def detect_frontier(
    pooled_inventory: dict[str, int], world_milestones: set[Milestone] | None = None
) -> Milestone:
    """Deepest team milestone from pooled inventory PLUS world-state milestones.

    Inventory milestones come from :func:`detect_milestone` (E1). World/location
    milestones (Nether entered, fortress/stronghold found, End portal active, End
    entered, dragon defeated) can't be read from inventory, so the env records
    them on ``World.world_milestones`` (E2/E3) and passes them here. Returns the
    deepest of the two (max-frontier, §7.1)."""
    best = detect_milestone(pooled_inventory)
    for m in world_milestones or ():
        if _MILESTONE_DEPTH[m] > _MILESTONE_DEPTH[best]:
            best = m
    return best


__all__ = [
    "RESOURCES",
    "FIGHT_DROPS",
    "BIOME_RESOURCES",
    "PICKAXE_TIERS",
    "GATHER_GATES",
    "RECIPES",
    "Recipe",
    "SMELTS",
    "SMELT_FUEL",
    "PLACEABLE_BLOCKS",
    "STUB_LADDER",
    "best_pickaxe",
    "gather_tool_ok",
    "craft_check",
    "smelt_check",
    "detect_milestone",
    "detect_frontier",
]

"""Stream 1 E1 — full tech tree + recipes + smelt/place (§3.4).

Validity-rejection tests for every new recipe, plus the new ``smelt`` / ``place``
actions, milestone detection for the deepened tree, and the tech-tree/biome
consistency invariants. Existing Phase-0 behaviour (oracle reaches IRON) is
preserved and covered by ``tests/test_env.py``.
"""

from __future__ import annotations

from random import Random

import pytest

from contracts import Action
from contracts.enums import ActionName, Biome, Layer, Milestone, Role
from env import StubEnv, techtree
from env.actions import resolve_action
from env.world import AgentState, Region, World

# --------------------------------------------------------------------------- #
# Every new E1 recipe with a *sufficient* inventory to craft it. Drives both the
# empty-inventory rejection sweep and the happy-path sweep.
# --------------------------------------------------------------------------- #
NEW_RECIPE_INPUTS: dict[str, dict[str, int]] = {
    "stone_sword": {"cobblestone": 2, "sticks": 1, "crafting_table": 1},
    "iron_pickaxe": {"iron_ingot": 3, "sticks": 2, "crafting_table": 1},
    "iron_sword": {"iron_ingot": 2, "sticks": 1, "crafting_table": 1},
    "shield": {"iron_ingot": 1, "planks": 6, "crafting_table": 1},
    "bucket": {"iron_ingot": 3, "crafting_table": 1},
    "flint_and_steel": {"iron_ingot": 1, "flint": 1, "crafting_table": 1},
    "diamond_pickaxe": {"diamond": 3, "sticks": 2, "crafting_table": 1},
    "diamond_sword": {"diamond": 2, "sticks": 1, "crafting_table": 1},
    "diamond_armor": {"diamond": 24, "crafting_table": 1},
    "obsidian": {"lava_pool": 1, "bucket": 1},  # route (b): water-trick
    "nether_portal": {"obsidian": 10, "flint_and_steel": 1},
    "blaze_powder": {"blaze_rod": 1},
    "eye_of_ender": {"blaze_powder": 1, "ender_pearl": 1},
    "end_portal": {"eye_of_ender": 12},
}


# --------------------------------------------------------------------------- #
# Minimal world/agent helpers for resolve_action-level tests (geometry-free).
# --------------------------------------------------------------------------- #
def _world() -> World:
    regions = {
        "r0": Region(
            id="r0",
            biome=Biome.MOUNTAINS,
            pos=(0.0, 0.0),
            resources={},
            layer=Layer.OVERWORLD,
            discovered=True,
        )
    }
    return World(regions=regions, start_region_id="r0")


def _resolve(inventory: dict[str, int], action: Action):
    """Resolve one action for a lone agent with ``inventory``; return (record, agent)."""
    world = _world()
    agent = AgentState(agent_id="a", role=Role.TINKERER, region_id="r0", inventory=dict(inventory))
    world.add_agent(agent)
    res = resolve_action(world, agent, action, Random(0), 0)
    return res.record, agent


# =========================================================================== #
# Recipes: existence + validity rejection + happy path
# =========================================================================== #
def test_all_new_recipes_registered():
    for name in NEW_RECIPE_INPUTS:
        assert name in techtree.RECIPES, f"missing recipe {name!r}"


@pytest.mark.parametrize("item", sorted(NEW_RECIPE_INPUTS))
def test_new_recipe_rejected_with_empty_inventory(item):
    ok, reason = techtree.craft_check(item, {})
    assert ok is False
    assert reason  # a non-empty reason string is logged (§3.3)


@pytest.mark.parametrize("item", sorted(NEW_RECIPE_INPUTS))
def test_new_recipe_crafts_with_sufficient_inputs(item):
    ok, reason = techtree.craft_check(item, NEW_RECIPE_INPUTS[item])
    assert ok is True
    assert reason is None


def test_eye_of_ender_without_ingredients_rejected():
    # Has blaze_powder but no ender_pearl -> rejected with a specific reason.
    ok, reason = techtree.craft_check("eye_of_ender", {"blaze_powder": 1})
    assert ok is False
    assert "ender_pearl" in reason


def test_diamond_armor_needs_full_24():
    ok, reason = techtree.craft_check("diamond_armor", {"diamond": 23, "crafting_table": 1})
    assert ok is False
    assert "need 24 diamond" in reason  # matches the fixtures.py sample reason


@pytest.mark.parametrize(
    "item, partial_inv, missing",
    [
        ("obsidian", {"lava_pool": 1}, "bucket"),  # route (b) needs a bucket
        ("nether_portal", {"obsidian": 10}, "flint_and_steel"),  # needs a lighter
        ("iron_pickaxe", {"iron_ingot": 3, "sticks": 2}, "crafting_table"),
    ],
)
def test_recipe_requires_rejection(item, partial_inv, missing):
    ok, reason = techtree.craft_check(item, partial_inv)
    assert ok is False
    assert f"requires {missing}" == reason


def test_craft_obsidian_water_trick_consumes_lava_keeps_bucket():
    rec, agent = _resolve(
        {"lava_pool": 1, "bucket": 1}, Action(name=ActionName.CRAFT, args={"item": "obsidian"})
    )
    assert rec.valid is True
    assert agent.inventory.get("obsidian", 0) == 1
    assert "lava_pool" not in agent.inventory  # consumed
    assert agent.inventory.get("bucket", 0) == 1  # reused, not consumed


def test_craft_nether_portal_consumes_obsidian_keeps_lighter():
    rec, agent = _resolve(
        {"obsidian": 10, "flint_and_steel": 1},
        Action(name=ActionName.CRAFT, args={"item": "nether_portal"}),
    )
    assert rec.valid is True
    assert agent.inventory.get("nether_portal", 0) == 1
    assert "obsidian" not in agent.inventory  # 10 consumed
    assert agent.inventory.get("flint_and_steel", 0) == 1  # reused


# =========================================================================== #
# Smelt
# =========================================================================== #
def test_smelt_iron_ore_to_ingot():
    rec, agent = _resolve(
        {"furnace": 1, "coal": 1, "iron_ore": 1},
        Action(name=ActionName.SMELT, args={"item": "iron_ore"}),
    )
    assert rec.valid is True
    assert agent.inventory.get("iron_ingot", 0) == 1
    assert "iron_ore" not in agent.inventory  # input consumed
    assert "coal" not in agent.inventory  # fuel consumed
    assert agent.inventory.get("furnace", 0) == 1  # furnace persists


def test_smelt_raw_food_to_cooked():
    rec, agent = _resolve(
        {"furnace": 1, "coal": 1, "food": 1},
        Action(name=ActionName.SMELT, args={"item": "food"}),
    )
    assert rec.valid is True
    assert agent.inventory.get("cooked_food", 0) == 1


def test_smelt_without_furnace_rejected():
    rec, agent = _resolve(
        {"coal": 1, "iron_ore": 1}, Action(name=ActionName.SMELT, args={"item": "iron_ore"})
    )
    assert rec.valid is False
    assert "furnace" in rec.reason
    assert agent.inventory.get("iron_ingot", 0) == 0


def test_smelt_without_fuel_rejected():
    rec, _ = _resolve(
        {"furnace": 1, "iron_ore": 1}, Action(name=ActionName.SMELT, args={"item": "iron_ore"})
    )
    assert rec.valid is False
    assert "coal" in rec.reason


def test_smelt_non_smeltable_rejected():
    rec, _ = _resolve(
        {"furnace": 1, "coal": 1, "diamond": 1},
        Action(name=ActionName.SMELT, args={"item": "diamond"}),
    )
    assert rec.valid is False
    assert "cannot smelt" in rec.reason


def test_smelt_missing_input_rejected():
    # Furnace + fuel present, but nothing to smelt.
    rec, _ = _resolve(
        {"furnace": 1, "coal": 1}, Action(name=ActionName.SMELT, args={"item": "iron_ore"})
    )
    assert rec.valid is False
    assert "iron_ore" in rec.reason


# =========================================================================== #
# Place
# =========================================================================== #
def test_place_obsidian_valid():
    rec, agent = _resolve({"obsidian": 2}, Action(name=ActionName.PLACE, args={"item": "obsidian"}))
    assert rec.valid is True
    assert rec.result.get("placed") == "obsidian"
    assert agent.inventory.get("obsidian", 0) == 1  # one consumed


def test_place_without_block_in_inventory_rejected():
    rec, _ = _resolve({}, Action(name=ActionName.PLACE, args={"item": "obsidian"}))
    assert rec.valid is False
    assert "no obsidian to place" in rec.reason


def test_place_non_placeable_item_rejected():
    rec, agent = _resolve(
        {"iron_pickaxe": 1}, Action(name=ActionName.PLACE, args={"item": "iron_pickaxe"})
    )
    assert rec.valid is False
    assert "cannot place" in rec.reason
    assert agent.inventory.get("iron_pickaxe", 0) == 1  # not consumed


def test_place_missing_arg_rejected():
    rec, _ = _resolve({"obsidian": 1}, Action(name=ActionName.PLACE, args={}))
    assert rec.valid is False
    assert "place needs an item" in rec.reason


# =========================================================================== #
# Gather gate for the diamond-pickaxe obsidian route (a)
# =========================================================================== #
def test_obsidian_gather_requires_diamond_pickaxe():
    assert techtree.gather_tool_ok("obsidian", {}) is False
    assert techtree.gather_tool_ok("obsidian", {"iron_pickaxe": 1}) is False  # too low a tier
    assert techtree.gather_tool_ok("obsidian", {"diamond_pickaxe": 1}) is True


# =========================================================================== #
# Milestone detection (inventory-detectable only; §3.4 / advisor scope)
# =========================================================================== #
def test_iron_ore_still_detects_iron():
    # The shallow oracle reaches IRON by gathering ore — must stay true.
    assert techtree.detect_milestone({"iron_ore": 1}) == Milestone.IRON


def test_iron_ingot_detects_iron():
    assert techtree.detect_milestone({"iron_ingot": 1}) == Milestone.IRON


@pytest.mark.parametrize(
    "inv, expected",
    [
        ({"bucket": 1}, Milestone.SHIELD_BUCKET),
        ({"shield": 1}, Milestone.SHIELD_BUCKET),
        ({"obsidian": 1}, Milestone.OBSIDIAN),
        ({"blaze_rod": 1}, Milestone.BLAZE_RODS),
        ({"ender_pearl": 1}, Milestone.ENDER_PEARLS),
        ({"eye_of_ender": 1}, Milestone.EYES_OF_ENDER),
    ],
)
def test_inventory_milestones_detected(inv, expected):
    assert techtree.detect_milestone(inv) == expected


def test_detect_returns_deepest_milestone():
    inv = {"stone_pickaxe": 1, "iron_ore": 2, "obsidian": 1}
    assert techtree.detect_milestone(inv) == Milestone.OBSIDIAN


@pytest.mark.parametrize(
    "inv, expected",
    [
        # Each inventory holds a shallower-tier item AND a deeper one; the deeper
        # milestone must win. Guards the shallow->deep override order in
        # detect_milestone against a future reorder of the if-blocks.
        ({"iron_ore": 1, "bucket": 1}, Milestone.SHIELD_BUCKET),
        ({"bucket": 1, "obsidian": 1}, Milestone.OBSIDIAN),
        ({"obsidian": 1, "blaze_rod": 1}, Milestone.BLAZE_RODS),
        ({"blaze_rod": 1, "ender_pearl": 1}, Milestone.ENDER_PEARLS),
        ({"ender_pearl": 1, "eye_of_ender": 1}, Milestone.EYES_OF_ENDER),
    ],
)
def test_detect_milestone_precedence_deepest_wins(inv, expected):
    assert techtree.detect_milestone(inv) == expected


@pytest.mark.parametrize("token", ["nether_portal", "end_portal"])
def test_portal_tokens_are_not_inventory_milestones(token):
    # Built-in-the-world milestones must NOT be faked off a crafted token (E6).
    assert techtree.detect_milestone({token: 1}) == Milestone.START


# =========================================================================== #
# Tech-tree / biome consistency invariants
# =========================================================================== #
def test_biome_resources_are_all_declared_resources():
    for biome, resources in techtree.BIOME_RESOURCES.items():
        for res in resources:
            assert res in techtree.RESOURCES, f"{res} (in {biome}) missing from RESOURCES"


def test_new_nether_biomes_present():
    for biome in (Biome.NETHER_WASTES, Biome.SOUL_SAND_VALLEY, Biome.WARPED_FOREST):
        assert biome in techtree.BIOME_RESOURCES


def test_smelt_outputs_are_distinct_from_inputs():
    for inp, out in techtree.SMELTS.items():
        assert inp != out


# =========================================================================== #
# Integration: smelt + place through a real StubEnv.step (full action path)
# =========================================================================== #
def _env_with_inventory(inv: dict[str, int]) -> StubEnv:
    env = StubEnv(seed="A", agents=[("agent_1", Role.MINER)])
    env.reset()
    env.world.agents["agent_1"].inventory.update(inv)
    return env


def test_smelt_through_env_step():
    env = _env_with_inventory({"furnace": 1, "coal": 1, "iron_ore": 1})
    res = env.step({"agent_1": Action(name=ActionName.SMELT, args={"item": "iron_ore"})})
    assert res.records[0].valid is True
    assert env.world.agents["agent_1"].inventory.get("iron_ingot", 0) == 1


def test_place_through_env_step():
    env = _env_with_inventory({"obsidian": 1})
    res = env.step({"agent_1": Action(name=ActionName.PLACE, args={"item": "obsidian"})})
    assert res.records[0].valid is True
    assert res.records[0].result.get("placed") == "obsidian"

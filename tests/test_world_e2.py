"""Stream 1 E2 — multi-layer world graph + cross-layer transitions (§3.1/§3.2).

Graph invariants (counts/connectivity/consistency), determinism, the explicit
portal (Overworld->Nether) and End (stronghold->End) transitions with their
rejection paths, and the new coord-free perception (landmarks/mobs). The richer
observation staying coord-clean is covered in obs_guard/coord_leak_test.py.
"""

from __future__ import annotations

from random import Random

import pytest

from contracts import Action
from contracts.enums import ActionName, Biome, Layer, Role, Structure, TimeOfDay
from env import make_world, techtree
from env.actions import resolve_action
from env.observation import perceived_mobs, serialize_observation
from env.seeds import ALL_SEEDS
from env.world import AgentState, Region, World


def _agent_at(world: World, region_id: str, inventory=None) -> AgentState:
    agent = AgentState(
        agent_id="a", role=Role.TINKERER, region_id=region_id, inventory=dict(inventory or {})
    )
    world.add_agent(agent)
    return agent


def _do(world: World, agent: AgentState, action: Action):
    return resolve_action(world, agent, action, Random(0), 0).record


def _by_layer(world: World) -> dict[Layer, list[Region]]:
    out: dict[Layer, list[Region]] = {lyr: [] for lyr in Layer}
    for region in world.regions.values():
        out[region.layer].append(region)
    return out


def _bfs(world: World, start: str) -> set[str]:
    seen, stack = {start}, [start]
    while stack:
        for rid, _d in world.neighbors(stack.pop()):
            if rid not in seen:
                seen.add(rid)
                stack.append(rid)
    return seen


# =========================================================================== #
# Graph invariants
# =========================================================================== #
def test_layer_counts_in_spec_range():
    layers = _by_layer(make_world("A"))
    assert len(layers[Layer.OVERWORLD]) == 24  # §3.7: 20–40
    assert len(layers[Layer.NETHER]) == 10  # §3.7: 8–15
    assert len(layers[Layer.END]) == 1


def test_layer_anchors_are_set_and_consistent():
    world = make_world("A")
    assert world.start_region_id == "r_00"
    assert world.regions["r_00"].biome == Biome.FOREST
    assert world.regions["r_00"].layer == Layer.OVERWORLD
    assert world.regions["r_00"].discovered is True
    assert world.regions[world.nether_entry_id].layer == Layer.NETHER
    assert world.regions[world.end_region_id].layer == Layer.END
    assert world.regions[world.stronghold_id].structure == Structure.STRONGHOLD


def test_only_start_region_discovered_at_genesis():
    world = make_world("A")
    discovered = [r.id for r in world.regions.values() if r.discovered]
    assert discovered == ["r_00"]


def test_structures_placed_one_fortress_one_stronghold():
    world = make_world("A")
    fortresses = [r for r in world.regions.values() if r.structure == Structure.FORTRESS]
    strongholds = [r for r in world.regions.values() if r.structure == Structure.STRONGHOLD]
    assert len(fortresses) == 1 and fortresses[0].layer == Layer.NETHER
    assert len(strongholds) == 1 and strongholds[0].layer == Layer.OVERWORLD


def test_deep_resources_present_in_world():
    # E1's deep resources are actually placed somewhere (lava_pool, nether mats).
    world = make_world("A")
    all_res = set()
    for r in world.regions.values():
        all_res |= set(r.resources)
    for res in ("lava_pool", "iron_ore", "diamond", "coal", "nether_wart", "basalt"):
        assert res in all_res, f"{res} not placed in any region"


def test_biome_resources_are_subset_of_declared_resources():
    # The E1 BIOME ⊆ RESOURCES invariant must still hold for generated worlds.
    for seed in ALL_SEEDS:
        for r in make_world(seed).regions.values():
            for res in r.resources:
                assert res in techtree.RESOURCES
                assert res in techtree.BIOME_RESOURCES.get(r.biome, set())


def test_each_layer_is_connected():
    world = make_world("A")
    layers = _by_layer(world)
    ow_ids = {r.id for r in layers[Layer.OVERWORLD]}
    ne_ids = {r.id for r in layers[Layer.NETHER]}
    assert _bfs(world, world.start_region_id) == ow_ids  # all OW reachable by move
    assert _bfs(world, world.nether_entry_id) == ne_ids  # all Nether reachable


def test_neighbors_never_cross_layers():
    world = make_world("A")
    for rid, region in world.regions.items():
        for nb_id, _d in world.neighbors(rid):
            assert world.regions[nb_id].layer == region.layer


# =========================================================================== #
# Determinism (§3.6) + per-seed distinctness
# =========================================================================== #
def _fingerprint(world: World):
    return sorted(
        (r.id, r.biome.value, r.layer.value, round(r.pos[0], 6), round(r.pos[1], 6),
         tuple(sorted(r.resources.items())), r.structure.value if r.structure else None)
        for r in world.regions.values()
    )


def test_same_seed_same_world():
    assert _fingerprint(make_world("A")) == _fingerprint(make_world("A"))


def test_distinct_seeds_distinct_layouts():
    assert _fingerprint(make_world("A")) != _fingerprint(make_world("B"))


# =========================================================================== #
# Spine geometry (preserves the Phase-0 oracle path + move-toward-nothing)
# =========================================================================== #
def test_move_north_from_start_reaches_mountains_with_iron():
    world = make_world("A")
    agent = _agent_at(world, "r_00")
    rec = _do(world, agent, Action(name=ActionName.MOVE, args={"direction": "N"}))
    assert rec.valid is True
    arrived = world.regions[agent.region_id]
    assert arrived.biome == Biome.MOUNTAINS
    assert arrived.resources.get("iron_ore", 0) > 0


def test_move_south_from_start_hits_nothing():
    world = make_world("A")
    agent = _agent_at(world, "r_00")
    rec = _do(world, agent, Action(name=ActionName.MOVE, args={"direction": "S"}))
    assert rec.valid is False
    assert "no region toward" in (rec.reason or "")


# =========================================================================== #
# Cross-layer transitions: Nether (happy + rejections)
# =========================================================================== #
def test_light_and_enter_nether_portal():
    world = make_world("A")
    agent = _agent_at(world, "r_00", inventory={"nether_portal": 1})
    lit = _do(world, agent, Action(name=ActionName.PLACE, args={"item": "nether_portal"}))
    assert lit.valid is True
    assert world.regions["r_00"].portal_to == world.nether_entry_id
    assert "nether_portal" not in agent.inventory  # token consumed
    moved = _do(world, agent, Action(name=ActionName.MOVE, args={"to": "nether"}))
    assert moved.valid is True
    assert agent.region_id == world.nether_entry_id
    assert world.regions[agent.region_id].layer == Layer.NETHER


def test_enter_nether_without_lit_portal_rejected():
    world = make_world("A")
    agent = _agent_at(world, "r_00")
    rec = _do(world, agent, Action(name=ActionName.MOVE, args={"to": "nether"}))
    assert rec.valid is False
    assert "no active portal" in (rec.reason or "")


def test_place_nether_portal_without_token_rejected():
    world = make_world("A")
    agent = _agent_at(world, "r_00")  # no token
    rec = _do(world, agent, Action(name=ActionName.PLACE, args={"item": "nether_portal"}))
    assert rec.valid is False
    assert "no nether_portal to place" in (rec.reason or "")


def test_nether_portal_must_be_lit_in_overworld():
    world = make_world("A")
    agent = _agent_at(world, world.nether_entry_id, inventory={"nether_portal": 1})  # in Nether
    rec = _do(world, agent, Action(name=ActionName.PLACE, args={"item": "nether_portal"}))
    assert rec.valid is False
    assert "Overworld" in (rec.reason or "")


def test_nether_portal_round_trip_returns_to_overworld():
    world = make_world("A")
    agent = _agent_at(world, "r_00", inventory={"nether_portal": 1})
    _do(world, agent, Action(name=ActionName.PLACE, args={"item": "nether_portal"}))
    _do(world, agent, Action(name=ActionName.MOVE, args={"to": "nether"}))
    back = _do(world, agent, Action(name=ActionName.MOVE, args={"to": "overworld"}))
    assert back.valid is True
    assert agent.region_id == "r_00"
    assert world.regions[agent.region_id].layer == Layer.OVERWORLD


def test_portal_move_rejects_mismatched_target_layer():
    # A nether portal here leads to the Nether; asking for a *different* layer is
    # rejected (the env, not the caller, decides where a portal goes). The matching
    # layer keyword and the generic "portal" keyword both still work.
    world = make_world("A")
    agent = _agent_at(world, "r_00", inventory={"nether_portal": 1})
    _do(world, agent, Action(name=ActionName.PLACE, args={"item": "nether_portal"}))

    bad = _do(world, agent, Action(name=ActionName.MOVE, args={"to": "end"}))
    assert bad.valid is False
    assert "not the end" in (bad.reason or "")
    assert agent.region_id == "r_00"  # did not move

    good = _do(world, agent, Action(name=ActionName.MOVE, args={"to": "nether"}))
    assert good.valid is True
    assert agent.region_id == world.nether_entry_id


def test_generic_portal_keyword_follows_whatever_is_linked():
    world = make_world("A")
    agent = _agent_at(world, "r_00", inventory={"nether_portal": 1})
    _do(world, agent, Action(name=ActionName.PLACE, args={"item": "nether_portal"}))
    moved = _do(world, agent, Action(name=ActionName.MOVE, args={"to": "portal"}))
    assert moved.valid is True
    assert world.regions[agent.region_id].layer == Layer.NETHER


# =========================================================================== #
# Cross-layer transitions: End (happy + rejections)
# =========================================================================== #
def test_activate_and_enter_end_portal_at_stronghold():
    world = make_world("A")
    agent = _agent_at(world, world.stronghold_id, inventory={"end_portal": 1})
    act = _do(world, agent, Action(name=ActionName.PLACE, args={"item": "end_portal"}))
    assert act.valid is True
    assert world.regions[world.stronghold_id].portal_to == world.end_region_id
    moved = _do(world, agent, Action(name=ActionName.MOVE, args={"to": "end"}))
    assert moved.valid is True
    assert agent.region_id == world.end_region_id
    assert world.regions[agent.region_id].layer == Layer.END


def test_end_portal_rejects_non_end_target():
    # The reviewer's repro: move{"to":"overworld"} on the End portal must NOT walk
    # the agent into the End. Only "end" (or generic "portal") follows it.
    world = make_world("A")
    agent = _agent_at(world, world.stronghold_id, inventory={"end_portal": 1})
    _do(world, agent, Action(name=ActionName.PLACE, args={"item": "end_portal"}))
    rec = _do(world, agent, Action(name=ActionName.MOVE, args={"to": "overworld"}))
    assert rec.valid is False
    assert agent.region_id == world.stronghold_id  # stayed put
    follow = _do(world, agent, Action(name=ActionName.MOVE, args={"to": "end"}))
    assert follow.valid is True
    assert world.regions[agent.region_id].layer == Layer.END


def test_activate_end_portal_away_from_stronghold_rejected():
    world = make_world("A")
    agent = _agent_at(world, "r_00", inventory={"end_portal": 1})  # not the stronghold
    rec = _do(world, agent, Action(name=ActionName.PLACE, args={"item": "end_portal"}))
    assert rec.valid is False
    assert "stronghold" in (rec.reason or "")


def test_enter_end_without_active_portal_rejected():
    world = make_world("A")
    agent = _agent_at(world, world.stronghold_id)  # not activated
    rec = _do(world, agent, Action(name=ActionName.MOVE, args={"to": "end"}))
    assert rec.valid is False
    assert "no active portal" in (rec.reason or "")


# =========================================================================== #
# Perception: mobs + landmarks (coord-free)
# =========================================================================== #
@pytest.mark.parametrize(
    "layer, structure, tod, expected",
    [
        (Layer.END, None, TimeOfDay.DAY, ["ender_dragon"]),
        (Layer.NETHER, Structure.FORTRESS, TimeOfDay.DAY, ["blaze"]),
        (Layer.NETHER, None, TimeOfDay.DAY, ["piglin"]),
        (Layer.OVERWORLD, None, TimeOfDay.NIGHT, ["zombie"]),
        (Layer.OVERWORLD, None, TimeOfDay.DAY, []),
    ],
)
def test_perceived_mobs(layer, structure, tod, expected):
    region = Region(id="r_99", biome=Biome.PLAINS, pos=(0.0, 0.0), layer=layer, structure=structure)
    assert perceived_mobs(region, tod) == expected


def test_landmarks_surface_discovered_neighbor_features():
    world = make_world("A")
    # r_07 (caves) is adjacent to the start and carries lava_pool; reveal it.
    world.regions["r_07"].discovered = True
    types = [t for (t, _b, _band) in world.perceived_landmarks("r_00")]
    assert "lava_pool" in types


def test_landmarks_empty_until_neighbors_discovered():
    world = make_world("A")  # only r_00 discovered
    assert world.perceived_landmarks("r_00") == []


def test_observation_populates_landmarks_after_scout():
    world = make_world("A")
    _agent_at(world, "r_00")
    world.regions["r_07"].discovered = True
    obs = serialize_observation(world, "a", round_idx=0, day_length=100)
    assert any(lm.type == "lava_pool" for lm in obs.known_landmarks)

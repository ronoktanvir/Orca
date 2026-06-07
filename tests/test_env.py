"""Stub env: reset/step return valid contracts, validity rejection, determinism (F3)."""

from __future__ import annotations

from contracts import Action, Observation
from contracts.enums import ActionName, Bearing, Milestone, Role
from env import StubEnv, make_world


def _fresh_env(**kw):
    env = StubEnv(seed="A", agents=[("agent_1", Role.MINER)], **kw)
    env.reset()
    return env


def test_reset_returns_valid_observations():
    env = StubEnv(seed="A", agents=[("agent_1", Role.MINER)])
    obs_map = env.reset()
    assert set(obs_map) == {"agent_1"}
    assert isinstance(obs_map["agent_1"], Observation)
    assert obs_map["agent_1"].self_view.current_biome.value == "forest"


def test_step_accepts_action_and_advances_round():
    env = _fresh_env()
    assert env.round_idx == 0
    env.step({"agent_1": Action(name=ActionName.GATHER, args={"resource": "wood"})})
    assert env.round_idx == 1
    assert env.world.agents["agent_1"].inventory.get("wood", 0) >= 1


def test_validity_rejection_gather_without_tool():
    # iron_ore needs a stone pickaxe; gathering it bare must be rejected + logged.
    env = _fresh_env()
    # move to the mountains first (scout then move N).
    env.step({"agent_1": Action(name=ActionName.SCOUT)})
    env.step({"agent_1": Action(name=ActionName.MOVE, args={"direction": "N"})})
    res = env.step({"agent_1": Action(name=ActionName.GATHER, args={"resource": "iron_ore"})})
    rec = res.records[0]
    assert rec.valid is False
    assert "stone_pickaxe" in (rec.reason or "")
    assert env.world.agents["agent_1"].inventory.get("iron_ore", 0) == 0


def test_validity_rejection_craft_without_inputs():
    env = _fresh_env()
    res = env.step({"agent_1": Action(name=ActionName.CRAFT, args={"item": "wooden_pickaxe"})})
    rec = res.records[0]
    assert rec.valid is False
    assert rec.reason  # a reason string is logged


def test_validity_rejection_move_toward_nothing():
    # From the start region, after scouting, moving S leads nowhere in the layout.
    env = _fresh_env()
    env.step({"agent_1": Action(name=ActionName.SCOUT)})
    res = env.step({"agent_1": Action(name=ActionName.MOVE, args={"direction": "S"})})
    rec = res.records[0]
    assert rec.valid is False
    assert "no region toward" in (rec.reason or "")


def test_unsupported_action_rejected():
    # FIGHT stays deferred to E4/E5 and must still reject with "not supported".
    # (SMELT/PLACE are implemented in E1 — covered by tests/test_techtree_e1.py.)
    env = _fresh_env()
    res = env.step({"agent_1": Action(name=ActionName.FIGHT, args={"target": "zombie"})})
    assert res.records[0].valid is False
    assert "not supported" in (res.records[0].reason or "")


def test_frontier_is_max_and_monotonic():
    env = _fresh_env()
    env.step({"agent_1": Action(name=ActionName.GATHER, args={"resource": "wood"})})
    assert env.frontier == Milestone.WOOD
    # Crafting wood into planks empties wood, but frontier must not regress.
    env.step({"agent_1": Action(name=ActionName.CRAFT, args={"item": "planks"})})
    assert env.frontier == Milestone.WOOD  # still WOOD (max-frontier, §7.1)


def test_determinism_same_seed_same_outcome():
    def gather_total(seed):
        env = StubEnv(seed=seed, agents=[("agent_1", Role.MINER)])
        env.reset()
        env.step({"agent_1": Action(name=ActionName.GATHER, args={"resource": "wood"})})
        return env.world.agents["agent_1"].inventory.get("wood", 0)

    assert gather_total("A") == gather_total("A")  # reproducible


def test_world_has_full_multilayer_graph():
    # E2 grew the 5-node stub into a full multi-layer graph (§3.1/§3.7):
    # 24 Overworld + 10 Nether + 1 End.
    world = make_world("A")
    from contracts.enums import Layer

    by_layer = {lyr: 0 for lyr in Layer}
    for region in world.regions.values():
        by_layer[region.layer] += 1
    assert by_layer[Layer.OVERWORLD] == 24
    assert by_layer[Layer.NETHER] == 10
    assert by_layer[Layer.END] == 1
    assert len(world.regions) == 35


def test_exits_hide_biome_until_discovered():
    world = make_world("A")
    # Before discovery, neighbor biome hints are UNKNOWN.
    hints = [hint for (_d, _b, hint) in world.exits_of("r_00")]
    assert all(h.value == "unknown" for h in hints)

"""Stream 1 E5 — superadditive cooperative combat (§3.5).

Co-location decides the hard fights: solo is unreliable (~0.2), a co-located trio
wins reliably (~0.85). That gap is THE cooperation incentive (delegation pays).
Location gating still rejects wrong-place fights; the full-DAG oracle still reaches
the dragon on every seed (it retries the now-stochastic fights). The shallow iron
path and the coord-leak guard stay green (covered elsewhere; re-asserted here).
"""

from __future__ import annotations

import pytest

from contracts import Action
from contracts.enums import ActionName, Biome, Layer, Milestone, Role, Structure
from env import StubEnv
from env.actions import resolve_action
from env.oracle import FullDagOracle, validate_winnable
from env.rng import make_rng
from env.seeds import ALL_SEEDS
from env.world import AgentState, Region, World


def _fortress_world(n_agents: int) -> tuple[World, AgentState]:
    """A 1-region Nether fortress with ``n_agents`` co-located. Returns (world, fighter)."""
    region = Region(
        id="r0", biome=Biome.NETHER_WASTES, pos=(0.0, 0.0),
        layer=Layer.NETHER, structure=Structure.FORTRESS, discovered=True,
    )
    world = World(regions={"r0": region}, start_region_id="r0")
    agents = [AgentState(agent_id=f"a{i}", role=Role.SUPPORT, region_id="r0") for i in range(n_agents)]
    for a in agents:
        world.add_agent(a)
    return world, agents[0]


def _fight(world, fighter, target, t):
    rng = make_rng("S", 0, t, fighter.agent_id)
    return resolve_action(world, fighter, Action(name=ActionName.FIGHT, args={"target": target}), rng, t).record


def _success_rate(n_agents: int, trials: int = 400) -> float:
    wins = 0
    for t in range(trials):
        world, fighter = _fortress_world(n_agents)
        rec = _fight(world, fighter, "blaze", t)
        if rec.result.get("defeated") == "blaze":
            wins += 1
    return wins / trials


# =========================================================================== #
# Superadditive success: solo low, trio high
# =========================================================================== #
def test_solo_blaze_success_is_low():
    assert _success_rate(1) < 0.35  # ~0.2 (§3.5)


def test_trio_blaze_success_is_high():
    assert _success_rate(3) > 0.70  # ~0.85 (§3.5)


def test_cooperation_strictly_helps():
    solo, pair, trio = _success_rate(1), _success_rate(2), _success_rate(3)
    assert solo < pair < trio  # monotone superadditive
    assert trio - solo > 0.4   # the gap that makes delegation matter


def test_dragon_is_also_superadditive():
    # Solo dragon attempts fail most of the time; co-location is what wins it.
    def dragon_rate(n):
        wins = 0
        for t in range(200):
            region = Region(id="e", biome=Biome.END, pos=(0.0, 0.0), layer=Layer.END, discovered=True)
            world = World(regions={"e": region}, start_region_id="e")
            ags = [AgentState(agent_id=f"a{i}", role=Role.SUPPORT, region_id="e") for i in range(n)]
            for a in ags:
                world.add_agent(a)
            if _fight(world, ags[0], "ender_dragon", t).result.get("win"):
                wins += 1
        return wins / 200
    assert dragon_rate(1) < 0.35 < 0.70 < dragon_rate(3)


# =========================================================================== #
# Co-location gating: same seeded draw, apart fails / together wins
# =========================================================================== #
def test_colocation_flips_a_losing_fight_to_a_win():
    # For the same RNG draw, there exists a round where a solo fight fails but a
    # co-located trio wins — co-location is what gates the outcome.
    flipped = 0
    for t in range(60):
        solo_world, solo = _fortress_world(1)
        trio_world, lead = _fortress_world(3)
        solo_lost = _fight(solo_world, solo, "blaze", t).result.get("defeated") is False
        trio_won = _fight(trio_world, lead, "blaze", t).result.get("defeated") == "blaze"
        if solo_lost and trio_won:
            flipped += 1
    assert flipped > 0, "co-location should flip at least one losing solo fight into a win"


def test_failed_fight_is_valid_not_invalid():
    # Find a round the solo fighter loses; the record must be valid (an attempt),
    # just without a drop — not an invalid action.
    for t in range(60):
        world, fighter = _fortress_world(1)
        rec = _fight(world, fighter, "blaze", t)
        if rec.result.get("defeated") is False:
            assert rec.valid is True
            assert fighter.inventory.get("blaze_rod", 0) == 0
            assert rec.result.get("n_colocated") == 1
            return
    pytest.fail("expected at least one failed solo fight in 60 rounds")


def test_enderman_is_a_regular_fight():
    # Endermen are not superadditive (§3.5 names only blaze + dragon): a lone
    # fighter in the Nether always gets the pearl.
    world, fighter = _fortress_world(1)
    rec = _fight(world, fighter, "enderman", t=0)
    assert rec.result.get("defeated") == "enderman"
    assert fighter.inventory.get("ender_pearl", 0) == 1


# =========================================================================== #
# Location gating still rejects wrong-place fights (reason logged)
# =========================================================================== #
def test_blaze_outside_fortress_rejected():
    region = Region(id="r0", biome=Biome.PLAINS, pos=(0.0, 0.0), layer=Layer.OVERWORLD, discovered=True)
    world = World(regions={"r0": region}, start_region_id="r0")
    fighter = AgentState(agent_id="a", role=Role.SUPPORT, region_id="r0")
    world.add_agent(fighter)
    rec = _fight(world, fighter, "blaze", t=0)
    assert rec.valid is False
    assert "fortress" in (rec.reason or "")


def test_dragon_outside_end_rejected():
    world, fighter = _fortress_world(1)  # in the Nether, not the End
    rec = _fight(world, fighter, "ender_dragon", t=0)
    assert rec.valid is False
    assert "End" in (rec.reason or "")


# =========================================================================== #
# Win path intact: oracle still reaches the dragon on every seed (now stochastic)
# =========================================================================== #
@pytest.mark.parametrize("seed", ALL_SEEDS)
def test_oracle_still_wins_every_seed(seed):
    assert validate_winnable(seed) is True


def test_oracle_reaches_dragon_with_zero_invalids():
    env = StubEnv(seed="A", agents=[("oracle", Role.TINKERER)], t_max=8000, stop_at_milestone=None)
    env.reset()
    oracle = FullDagOracle("oracle")
    won = oracle.solve(env)
    assert won is True
    assert env.frontier == Milestone.DRAGON_DEFEATED
    assert oracle.invalid_actions == 0  # failed fights are valid attempts, not invalid


def test_oracle_run_still_deterministic():
    def run():
        env = StubEnv(seed="A", agents=[("o", Role.TINKERER)], t_max=8000, stop_at_milestone=None)
        env.reset()
        o = FullDagOracle("o")
        return o.solve(env), env.round_idx
    assert run() == run()

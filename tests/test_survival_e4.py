"""Stream 1 E4 — hunger / EAT / death + respawn (§3.5).

Hunger drains each round (faster when moving/fighting); at 0 hunger health drains;
at 0 health the agent dies, drops its non-equipped inventory, and respawns at the
start region. EAT (cooked_food) restores hunger. Rates are gentle so the scripted
oracle survives its run without eating — death is forced here to test the mechanic.
The iron path and the full-DAG oracle stay intact.
"""

from __future__ import annotations

import pytest

from contracts import Action
from contracts.enums import ActionName, Milestone, Role
from env import StubEnv
from env.oracle import validate_winnable
from env.seeds import ALL_SEEDS
from env.stub_env import _RESPAWN_COST


def _env(seed="A", t_max=80):
    env = StubEnv(seed=seed, agents=[("a", Role.MINER)], t_max=t_max, stop_at_milestone=None)
    env.reset()
    return env


# =========================================================================== #
# Hunger drain
# =========================================================================== #
def test_hunger_drains_each_round():
    env = _env()
    a = env.world.agents["a"]
    assert a.hunger == 1.0
    for _ in range(10):
        env.step({"a": Action(name=ActionName.WAIT)})
    assert a.hunger < 1.0          # it drains
    assert a.hunger > 0.9          # ...but gently (a competent agent survives)


def test_moving_drains_more_than_waiting():
    # Same seed -> same RNG jitter, so the only difference is the active base rate.
    def final_hunger(make_action):
        env = _env()
        a = env.world.agents["a"]
        for _ in range(20):
            env.step({"a": make_action()})
        return a.hunger

    waited = final_hunger(lambda: Action(name=ActionName.WAIT))
    moved = final_hunger(lambda: Action(name=ActionName.MOVE, args={"direction": "N"}))
    assert moved < waited


# =========================================================================== #
# EAT
# =========================================================================== #
def test_eat_restores_hunger_and_consumes_food():
    env = _env()
    a = env.world.agents["a"]
    a.hunger = 0.3
    a.inventory = {"cooked_food": 2}
    res = env.step({"a": Action(name=ActionName.EAT, args={"food": "cooked_food"})})
    assert res.records[0].valid is True
    assert a.hunger > 0.75                       # +0.5 restore, minus a tiny tick
    assert a.inventory.get("cooked_food", 0) == 1  # one consumed


def test_eat_without_cooked_food_rejected():
    env = _env()
    res = env.step({"a": Action(name=ActionName.EAT, args={})})
    assert res.records[0].valid is False
    assert "cooked_food" in (res.records[0].reason or "")


# =========================================================================== #
# Death + respawn
# =========================================================================== #
def test_starvation_causes_death_dropping_nonequipped_inventory():
    env = _env()
    a = env.world.agents["a"]
    a.inventory = {"iron_pickaxe": 1, "shield": 1, "diamond": 5, "wood": 3, "obsidian": 9}
    a.hunger = 0.0
    a.health = _health_one_tick_from_death()
    env.step({"a": Action(name=ActionName.WAIT)})
    assert a.alive is False
    assert a.deaths == 1
    assert a.inventory == {"iron_pickaxe": 1, "shield": 1}  # gear kept; resources dropped


def test_respawn_at_start_after_cost_with_full_bars():
    env = _env()
    a = env.world.agents["a"]
    a.hunger = 0.0
    a.health = _health_one_tick_from_death()
    env.step({"a": Action(name=ActionName.WAIT)})  # dies
    assert a.alive is False
    a.region_id = "r_06"  # stray far so respawn-at-start is observable
    for _ in range(_RESPAWN_COST):
        env.step({"a": Action(name=ActionName.WAIT)})
    assert a.alive is True
    assert a.health == 1.0 and a.hunger == 1.0
    assert a.region_id == env.world.start_region_id


def test_dead_agent_does_not_act():
    env = _env()
    a = env.world.agents["a"]
    a.hunger = 0.0
    a.health = _health_one_tick_from_death()
    env.step({"a": Action(name=ActionName.WAIT)})  # dies
    n_before = len(env.all_records)
    env.step({"a": Action(name=ActionName.GATHER, args={"resource": "wood"})})
    assert len(env.all_records) == n_before  # the dead agent's action was not resolved


def _health_one_tick_from_death() -> float:
    from env.stub_env import _HEALTH_DRAIN_STARVING
    return _HEALTH_DRAIN_STARVING / 2  # one starving tick takes it to 0


# =========================================================================== #
# Determinism + win path intact
# =========================================================================== #
def test_survival_is_deterministic():
    def run():
        env = _env()
        a = env.world.agents["a"]
        for _ in range(30):
            env.step({"a": Action(name=ActionName.WAIT)})
        return a.hunger
    assert run() == run()


def test_iron_path_intact_with_hunger_on():
    from agents.scripted import ShallowOracle

    env = StubEnv(seed="A", agents=[("a", Role.MINER)])  # default: stop at iron
    env.reset()
    oracle = ShallowOracle("a")
    while not env.done:
        env.step({"a": oracle.act(env.observe("a"))})
    assert env.frontier == Milestone.IRON
    assert env.world.agents["a"].hunger > 0.9  # barely touched in ~14 rounds


@pytest.mark.parametrize("seed", ALL_SEEDS)
def test_oracle_still_reaches_dragon_without_starving(seed):
    assert validate_winnable(seed) is True

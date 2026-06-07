"""Stream 1 E3 — full-DAG oracle, winnability validator, greedy contrast (§3.7).

The oracle-wins / greedy-stalls gap is the headline result, so these are the real
safety net: the oracle must reach ``dragon_defeated`` on seed A with zero invalid
actions, every seed must validate winnable, and the greedy baseline must stall
mid-DAG. The shallow iron path stays intact (covered by test_oracle.py + run.py).
"""

from __future__ import annotations

import pytest

from contracts.enums import Milestone, Role
from env import StubEnv
from env.oracle import FullDagOracle, GreedyAgent, validate_winnable
from env.seeds import ALL_SEEDS, generate, make_world


def _run(agent_cls, seed="A", t_max=8000):
    env = StubEnv(seed=seed, agents=[("a", Role.TINKERER)], t_max=t_max, stop_at_milestone=None)
    env.reset()
    agent = agent_cls("a")
    won = agent.solve(env)
    return env, agent, won


# =========================================================================== #
# Oracle reaches the dragon
# =========================================================================== #
def test_oracle_reaches_dragon_on_seed_A():
    env, oracle, won = _run(FullDagOracle, "A")
    assert won is True
    assert Milestone.DRAGON_DEFEATED in env.world.world_milestones
    assert env.frontier == Milestone.DRAGON_DEFEATED  # frontier tracks the win


def test_oracle_emits_zero_invalid_actions():
    env, oracle, _won = _run(FullDagOracle, "A")
    assert oracle.invalid_actions == 0
    invalids = [r for r in env.all_records if not r.valid]
    assert invalids == [], f"oracle emitted invalids: {[(r.action.name, r.reason) for r in invalids]}"


def test_oracle_walks_the_full_dag_in_order():
    # Every milestone on the ladder from PORTAL_BUILT to the win is achieved.
    env, _oracle, _won = _run(FullDagOracle, "A")
    deep = {
        Milestone.PORTAL_BUILT, Milestone.NETHER_ENTERED, Milestone.FORTRESS_FOUND,
        Milestone.STRONGHOLD_FOUND, Milestone.END_PORTAL_ACTIVE, Milestone.END_ENTERED,
        Milestone.DRAGON_DEFEATED,
    }
    assert deep <= env.world.world_milestones
    # inventory-detectable deep milestones were also reached en route.
    inv_reached = {m.value for m in env.milestone_timeline and [e.milestone for e in env.milestone_timeline]}
    for name in ("obsidian", "blaze_rods", "ender_pearls", "eyes_of_ender"):
        assert name in inv_reached


# =========================================================================== #
# Validator: every seed is winnable (train + held-out)
# =========================================================================== #
@pytest.mark.parametrize("seed", ALL_SEEDS)
def test_validate_winnable_all_seeds(seed):
    assert validate_winnable(seed) is True


def test_validate_winnable_covers_train_and_heldout():
    assert all(validate_winnable(s) for s in ("A", "B", "C"))


# =========================================================================== #
# Greedy contrast: stalls mid-DAG
# =========================================================================== #
def test_greedy_stalls_mid_dag():
    env, greedy, won = _run(GreedyAgent, "A")
    assert won is False
    assert Milestone.DRAGON_DEFEATED not in env.world.world_milestones
    # It is "mid-DAG", not stuck at zero: it does reach the iron tier.
    assert env.frontier == Milestone.IRON
    assert greedy.invalid_actions == 0  # greedy is valid, just myopic


def test_greedy_never_enters_nether():
    env, _greedy, _won = _run(GreedyAgent, "A")
    assert Milestone.NETHER_ENTERED not in env.world.world_milestones
    assert Milestone.PORTAL_BUILT not in env.world.world_milestones


# =========================================================================== #
# Determinism + generator entry + shallow iron path intact
# =========================================================================== #
def test_oracle_run_is_deterministic():
    e1, o1, w1 = _run(FullDagOracle, "A")
    e2, o2, w2 = _run(FullDagOracle, "A")
    assert (w1, o1.invalid_actions, e1.round_idx) == (w2, o2.invalid_actions, e2.round_idx)


def test_generate_is_make_world_alias():
    assert generate is make_world
    assert generate("A").start_region_id == "r_00"


def test_shallow_iron_path_unchanged():
    # The frontier still reaches IRON from pooled inventory alone (no world state),
    # so the Phase-0 oracle / run.py path is untouched by the deep-milestone wiring.
    from env import techtree

    assert techtree.detect_frontier({"iron_ore": 1}, set()) == Milestone.IRON
    assert techtree.detect_frontier({"iron_ore": 1}, None) == Milestone.IRON

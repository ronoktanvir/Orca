"""The scripted placeholder agent deterministically reaches iron (F4 / §12)."""

from __future__ import annotations

from agents.scripted import ShallowOracle
from contracts.enums import Milestone, Role
from env import StubEnv


def _run(seed="A", t_max=200):
    env = StubEnv(seed=seed, agents=[("agent_1", Role.MINER)], t_max=t_max)
    env.reset()
    oracle = ShallowOracle("agent_1")
    actions = []
    while not env.done:
        obs = env.observe("agent_1")
        act = oracle.act(obs)
        actions.append(act.name.value)
        env.step({"agent_1": act})
    return env, actions


def test_oracle_reaches_iron():
    env, _actions = _run()
    assert env.frontier == Milestone.IRON
    assert env.terminated_reason == "frontier_target"
    assert env.world.agents["agent_1"].inventory.get("iron_ore", 0) >= 1


def test_oracle_is_deterministic():
    env1, actions1 = _run()
    env2, actions2 = _run()
    assert actions1 == actions2
    assert env1.round_idx == env2.round_idx


def test_oracle_does_not_emit_invalid_actions():
    env, _ = _run()
    invalids = [r for r in env.all_records if not r.valid]
    assert invalids == [], f"oracle emitted invalid actions: {[(r.action.name, r.reason) for r in invalids]}"


def test_oracle_finishes_well_within_t_max():
    env, _ = _run()
    assert env.round_idx < 50  # reaches iron quickly on the stub seed

"""Reward computer: max-frontier ladder + penalties + advisory dials at 0 (§7)."""

from __future__ import annotations

from contracts.enums import ActionName, Milestone, Role
from reward import MILESTONE_VALUE, frontier_value, reward_computer
from tests.fixtures import sample_episode_trace


def test_ladder_is_monotonic_and_anchored():
    values = [MILESTONE_VALUE[m] for m in Milestone]
    assert values == sorted(values)  # monotonic non-decreasing in depth
    # §7.1 anchors
    assert MILESTONE_VALUE[Milestone.WOOD] == 0.05
    assert MILESTONE_VALUE[Milestone.IRON] == 0.20
    assert MILESTONE_VALUE[Milestone.PORTAL_BUILT] == 0.30
    assert MILESTONE_VALUE[Milestone.DRAGON_DEFEATED] == 1.00


def test_reward_computer_from_trace():
    trace = sample_episode_trace()  # frontier IRON, 1 valid + 1 invalid action
    m = reward_computer(trace, agent_roles={"agent_1": Role.MINER})
    assert m.frontier_milestone == Milestone.IRON
    assert m.frontier_value == 0.20
    # 1 of 2 actions invalid -> invalid_rate 0.5; penalty subtracts from base
    assert m.invalid_rate == 0.5
    assert m.team_reward < m.frontier_value
    assert m.team_reward >= 0.0
    # advisory dials are not summed into team reward (§6.4)
    assert all(s.performance_score == 0.0 for s in m.agent_stats)


def test_team_reward_clipped_nonnegative():
    trace = sample_episode_trace()
    m = reward_computer(
        trace,
        agent_roles={"agent_1": Role.MINER},
        weights={"deaths": 10.0, "invalid": 10.0, "idle": 10.0},  # absurd penalties
    )
    assert m.team_reward == 0.0  # clipped, never negative


def test_per_agent_stats_counts():
    trace = sample_episode_trace()
    m = reward_computer(trace, agent_roles={"agent_1": Role.MINER})
    stats = {s.agent_id: s for s in m.agent_stats}["agent_1"]
    assert stats.actions_taken == 2
    assert stats.invalid_actions == 1
    assert stats.items_gathered.get("wood", 0) == 3


def test_frontier_value_helper():
    assert frontier_value(Milestone.START) == 0.0
    assert frontier_value(Milestone.IRON) == 0.20

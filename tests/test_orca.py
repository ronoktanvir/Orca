"""Tests for the real Orca (§6) — bandit math, scoring purity, coach, gate, phasing.

No test hits the LLM API: the coach is exercised with a deterministic mock client
and with its offline heuristic fallback.
"""

from __future__ import annotations

import json

from contracts import AgentStats, BehaviorCard, EpisodeMetrics
from contracts.enums import Milestone, Role
from orca.bandit import EpsilonGreedyBandit
from orca.coach import CoachOutput, run_coach
from orca.digest import build_digest
from orca.gate import AcceptGate, GateDecision
from orca.orca import Orca, Proposal
from orca.scoring import learning_signal, performance_score, score_agent, score_agents
from orca.situations import S1, SITUATION_ARMS
from train.phases import Phase, current_phase

from tests.fixtures import sample_episode_metrics, sample_episode_trace


# --------------------------------------------------------------------------- #
# Bandit (O2, §6.3)
# --------------------------------------------------------------------------- #
def test_bandit_running_mean_update():
    b = EpsilonGreedyBandit(arms={"S": ["a", "b"]}, epsilon=0.0, seed=1)
    b.update("S", "a", 1.0)
    b.update("S", "a", 0.0)  # running mean of [1, 0] = 0.5
    b.update("S", "a", 0.5)  # running mean of [1, 0, 0.5] = 0.5
    assert abs(b.values()["S"]["a"] - 0.5) < 1e-9
    assert b.values()["S"]["b"] == 0.0


def test_bandit_optimistic_init_then_overwritten():
    b = EpsilonGreedyBandit(arms={"S": ["a", "b"]}, epsilon=0.0, seed=1, optimistic=1.0)
    # every arm starts optimistic, so greedy first picks an untried arm
    assert b.values()["S"]["a"] == 1.0
    b.update("S", "a", 0.3)  # first real obs overwrites the optimistic seed
    assert abs(b.values()["S"]["a"] - 0.3) < 1e-9


def test_bandit_greedy_picks_highest_value_arm():
    b = EpsilonGreedyBandit(arms={"S": ["a", "b", "c"]}, epsilon=0.0, seed=2)
    for _ in range(5):
        b.update("S", "a", 0.2)
        b.update("S", "b", 0.9)
        b.update("S", "c", 0.5)
    assert b.greedy("S") == "b"
    assert b.choose("S") == "b"  # epsilon 0 -> greedy


def test_bandit_learns_better_arm_over_episodes():
    b = EpsilonGreedyBandit(arms={"S": ["lo", "hi"]}, epsilon=0.2, seed=7, optimistic=1.0)
    rng_rewards = {"lo": 0.3, "hi": 0.7}
    for _ in range(60):
        arm = b.choose("S")
        b.update("S", arm, rng_rewards[arm])
    vals = b.values()["S"]
    assert vals["hi"] > vals["lo"]


# --------------------------------------------------------------------------- #
# Scoring (O3, §7.3) — purely from agent_stats
# --------------------------------------------------------------------------- #
def _stats(**kw) -> AgentStats:
    base = dict(agent_id="a", role=Role.MINER, actions_taken=10)
    base.update(kw)
    return AgentStats(**base)


def test_scoring_is_deterministic_and_pure():
    s = _stats(invalid_actions=2, idle_rounds=1, items_gathered={"wood": 3})
    p1, l1 = score_agent(s)
    p2, l2 = score_agent(s)
    assert (p1, l1) == (p2, l2)  # deterministic
    assert 0.0 <= p1 <= 1.0 and -1.0 <= l1 <= 1.0


def test_scoring_ranks_clean_above_failing():
    clean = _stats(invalid_actions=0, idle_rounds=0, items_gathered={"wood": 4, "iron_ore": 2}, items_crafted={"stone_pickaxe": 2})
    failing = _stats(invalid_actions=8, idle_rounds=1)
    assert performance_score(clean) > performance_score(failing)
    # learning_signal grows with dysfunction
    assert learning_signal(failing) > learning_signal(clean)


def test_scoring_no_opinion_is_objective_only():
    s = _stats(invalid_actions=3)
    objective = performance_score(s, opinion=None)
    # supplying an opinion only nudges the score (lightly), never replaces it
    high_op = performance_score(s, opinion=1.0)
    low_op = performance_score(s, opinion=0.0)
    assert low_op <= objective <= high_op
    assert abs(high_op - objective) <= 0.15 + 1e-9  # opinion_blend cap


def test_scoring_idle_agent_scores_zero():
    assert performance_score(_stats(actions_taken=0)) == 0.0
    assert learning_signal(_stats(actions_taken=0)) == 0.0


def test_score_agents_fills_dials_without_touching_reward():
    metrics = sample_episode_metrics()
    scored = score_agents(metrics.agent_stats)
    assert all(0.0 <= st.performance_score <= 1.0 for st in scored)
    # team_reward is never derived here — anti-circularity (§6.4)
    assert metrics.team_reward == sample_episode_metrics().team_reward


# --------------------------------------------------------------------------- #
# Digest (O1, §6.1)
# --------------------------------------------------------------------------- #
def test_digest_renders_and_flags_invalids():
    d = build_digest(sample_episode_trace(), sample_episode_metrics())
    text = d.render()
    assert "EPISODE 0" in text and "frontier=iron" in text
    assert d.agents and d.agents[0].agent_id == "agent_1"


# --------------------------------------------------------------------------- #
# Coach (O4, §6.4) — mock LLM + heuristic fallback
# --------------------------------------------------------------------------- #
class _MockLLM:
    """Returns a fixed valid CoachOutput JSON; records that it was called."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = 0

    def complete(self, prompt, schema=None, **kwargs) -> str:
        self.calls += 1
        return json.dumps(self.payload)


def _failing_trace_metrics():
    """An episode where agent_1 repeatedly fails to mine iron (no stone pickaxe)."""
    from contracts import Action, EpisodeTrace
    from contracts.episode import ActionRecord, MilestoneEvent
    from contracts.enums import ActionName

    recs = [
        ActionRecord(
            round=r,
            agent_id="agent_1",
            action=Action(name=ActionName.GATHER, args={"resource": "iron_ore"}),
            valid=False,
            reason="need stone_pickaxe (have 0)",
        )
        for r in range(5)
    ]
    trace = EpisodeTrace(
        episode_idx=2,
        seed="A",
        n_rounds=5,
        agent_ids=["agent_1"],
        config={"arms": {S1: "mining_heavy"}},
        behavior_cards=[],
        action_records=recs,
        messages=[],
        milestone_timeline=[MilestoneEvent(milestone=Milestone.WOOD, round=0)],
        frontier_reached=Milestone.WOOD,
        terminated_reason="t_max",
        observations=[],
    )
    metrics = EpisodeMetrics(
        episode_idx=2,
        seed="A",
        frontier_milestone=Milestone.WOOD,
        frontier_value=0.05,
        team_reward=0.0,
        invalid_rate=1.0,
        n_rounds=5,
        agent_stats=[AgentStats(agent_id="agent_1", role=Role.MINER, actions_taken=5, invalid_actions=5)],
    )
    return trace, metrics


def test_coach_heuristic_writes_execution_fix():
    trace, metrics = _failing_trace_metrics()
    prop = run_coach(trace, metrics, cards={})
    assert "agent_1" in prop.behavior_cards
    card = prop.behavior_cards["agent_1"]
    assert card.version == 1  # bumped
    assert any("prerequisit" in d.lower() for d in card.directives)
    assert "execution" in prop.rationale


def test_coach_uses_mock_llm_no_api():
    trace, metrics = _failing_trace_metrics()
    llm = _MockLLM(
        {
            "team_reasoning": "agent_1 lacks a stone pickaxe.",
            "agents": [
                {
                    "agent_id": "agent_1",
                    "credit": "execution",
                    "reasoning": "missing tool",
                    "new_directives": ["craft a stone pickaxe before mining iron"],
                    "learning_signal": 0.8,
                }
            ],
        }
    )
    prop = run_coach(trace, metrics, cards={}, llm=llm)
    assert llm.calls == 1
    assert prop.notes == "coach(llm)"
    assert "craft a stone pickaxe before mining iron" in prop.behavior_cards["agent_1"].directives


def test_coach_falls_back_to_heuristic_on_bad_llm():
    trace, metrics = _failing_trace_metrics()

    class _BadLLM:
        def complete(self, *a, **k):
            return "not json at all"

    prop = run_coach(trace, metrics, cards={}, llm=_BadLLM())
    assert prop.notes == "coach(heuristic)"
    assert "agent_1" in prop.behavior_cards


def test_coach_clean_episode_proposes_nothing():
    prop = run_coach(sample_episode_trace(), sample_episode_metrics(), cards={})
    assert prop.is_empty()  # no bottleneck -> no edits


# --------------------------------------------------------------------------- #
# Accept-gate (O5, §6.5)
# --------------------------------------------------------------------------- #
def test_gate_consider_keeps_and_ratchets():
    gate = AcceptGate(epsilon=0.02, baseline=0.5)
    d = gate.consider(0.6)
    assert d.accepted and gate.best == 0.6
    d2 = gate.consider(0.59)  # within epsilon of best -> kept
    assert d2.accepted
    d3 = gate.consider(0.4)  # well below best -> rolled back
    assert not d3.accepted and d3.rolled_back
    assert gate.best == 0.6  # best never regresses


class _FakeOrca:
    def __init__(self):
        self.cards: dict[str, BehaviorCard] = {}

    def snapshot(self):
        return {k: v.model_copy(deep=True) for k, v in self.cards.items()}

    def restore(self, snap):
        self.cards = {k: v.model_copy(deep=True) for k, v in snap.items()}

    def commit(self, proposal):
        self.cards.update(proposal.behavior_cards)


def _metrics_with_reward(r: float) -> EpisodeMetrics:
    return sample_episode_metrics().model_copy(update={"team_reward": r})


def test_gate_evaluate_keeps_good_edit():
    orca = _FakeOrca()
    gate = AcceptGate(epsilon=0.02, baseline=0.2)
    prop = Proposal(behavior_cards={"agent_1": BehaviorCard(agent_id="agent_1", role=Role.MINER, version=1)})
    d = gate.evaluate(orca, prop, lambda: [_metrics_with_reward(0.5), _metrics_with_reward(0.5)])
    assert d.accepted and not d.rolled_back
    assert "agent_1" in orca.cards  # kept


def test_gate_evaluate_rolls_back_bad_edit_and_restores_cards():
    orca = _FakeOrca()
    orca.cards = {"agent_2": BehaviorCard(agent_id="agent_2", role=Role.MINER, version=3)}
    gate = AcceptGate(epsilon=0.02, baseline=0.5)
    prop = Proposal(behavior_cards={"agent_1": BehaviorCard(agent_id="agent_1", role=Role.MINER, version=1)})
    d = gate.evaluate(orca, prop, lambda: [_metrics_with_reward(0.1)])
    assert d.rolled_back and not d.accepted
    assert "agent_1" not in orca.cards  # rolled back
    assert orca.cards["agent_2"].version == 3  # original preserved


def test_gate_empty_proposal_is_noop_keep():
    orca = _FakeOrca()
    gate = AcceptGate(epsilon=0.02, baseline=0.4)
    d = gate.evaluate(orca, Proposal(), lambda: [_metrics_with_reward(0.9)])
    assert d.accepted and d.note == "empty-proposal"


# --------------------------------------------------------------------------- #
# Phasing (O6, §6.6)
# --------------------------------------------------------------------------- #
def test_phasing_transitions():
    assert current_phase(0, 15, False) == Phase.PHASE_0
    assert current_phase(14, 15, False) == Phase.PHASE_0
    assert current_phase(15, 15, False) == Phase.PHASE_1
    assert current_phase(3, 15, True) == Phase.PHASE_2  # first win -> phase 2 regardless


# --------------------------------------------------------------------------- #
# Orca integration (O2 wiring)
# --------------------------------------------------------------------------- #
def test_orca_choose_config_has_all_situations():
    orca = Orca([("agent_1", Role.MINER), ("agent_2", Role.EXPLORER)], epsilon=0.0)
    cfg = orca.choose_config()
    assert set(cfg.arms.keys()) == set(SITUATION_ARMS.keys())
    assert len(cfg.roster) == 2
    assert set(cfg.behavior_cards.keys()) == {"agent_1", "agent_2"}


def test_orca_observe_outcome_updates_bandit_and_tracks_seed():
    orca = Orca([("agent_1", Role.MINER)], epsilon=0.0)
    cfg = orca.choose_config()
    orca.observe_outcome(cfg, _metrics_with_reward(0.42))
    assert "A" in orca.trained_seeds  # sample metrics seed is "A"
    # the chosen arms now carry the observed reward
    chosen_s1 = cfg.arms[S1]
    assert abs(orca.bandit.values()[S1][chosen_s1] - 0.42) < 1e-9


def test_orca_freeze_stops_learning():
    orca = Orca([("agent_1", Role.MINER)], epsilon=0.2)
    orca.freeze()
    before = orca.bandit.values()
    orca.observe_outcome(orca.choose_config(), _metrics_with_reward(0.9))
    assert orca.bandit.values() == before  # no update after freeze
    assert orca.bandit.epsilon == 0.0
    assert orca.trained_seeds == set()

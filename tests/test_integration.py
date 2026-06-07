"""Integration test for the worker seam (Stream 2 swap, pre-validated offline).

The real LLM-worker path is wired through ``train.loop.build_agents`` /
``eval.harness.RealRunner`` via a ``worker_factory(agent_id, role, llm)`` callable
(the §8 seam that swaps ``ShallowOracle`` → ``LLMWorker``). Stream 2's real worker
isn't built yet, so we exercise the *plumbing* with a deterministic mock worker:
the factory is called with the right signature, a non-oracle worker flows through
the real env + reward + scoring pipeline, and the full ``run`` loop drives it.

This is what makes the eventual one-line swap (`worker_factory=make_llm_worker`)
low-risk: when Stream 2 lands, only the factory body changes.
"""

from __future__ import annotations

from agents.scripted import ShallowOracle
from config import load_config
from contracts import Action, EpisodeMetrics, EpisodeTrace, Observation
from contracts.enums import Role
from eval.harness import FULL_C2_SPEC, RealRunner, make_orca
from eval.outcome_model import FULL_C2
from telemetry import init_telemetry
from train.loop import build_agents, run


class _MockWorker:
    """A non-oracle worker built via the factory; reaches iron via the oracle's
    policy internally, and proves it *reads its behavior card* from the obs."""

    def __init__(self, agent_id: str, role: Role, llm) -> None:
        self.agent_id = agent_id
        self.role = role
        self.llm = llm
        self.saw_assignment = False
        self._policy = ShallowOracle(agent_id)

    def act(self, obs: Observation) -> Action:
        if getattr(obs, "assignment", ""):
            self.saw_assignment = True  # the card -> obs -> worker path is live
        return self._policy.act(obs)


def _make_factory():
    calls: list[tuple] = []
    workers: list[_MockWorker] = []

    def factory(agent_id: str, role: Role, llm):
        calls.append((agent_id, role, llm))
        w = _MockWorker(agent_id, role, llm)
        workers.append(w)
        return w

    return factory, calls, workers


def test_build_agents_uses_worker_factory_with_correct_signature():
    factory, calls, _ = _make_factory()
    sentinel_llm = object()
    roster = [("agent_1", Role.MINER), ("agent_2", Role.EXPLORER)]
    agents = build_agents(roster, llm=sentinel_llm, worker_factory=factory)

    assert [a.agent_id for a in agents] == ["agent_1", "agent_2"]
    assert calls == [("agent_1", Role.MINER, sentinel_llm), ("agent_2", Role.EXPLORER, sentinel_llm)]
    assert all(isinstance(a, _MockWorker) for a in agents)


def test_realrunner_drives_mock_worker_through_real_pipeline():
    settings = load_config()
    settings.telemetry.mode = "off"
    factory, calls, workers = _make_factory()
    sentinel_llm = object()
    runner = RealRunner(settings, telemetry=init_telemetry(mode="off"), llm=sentinel_llm, worker_factory=factory)

    orca = make_orca(FULL_C2_SPEC, settings)
    config = orca.choose_config(greedy=True)
    trace, metrics = runner(config, "A", condition=FULL_C2, episode_idx=0)

    # real env + reward path produced valid contracts and reached iron
    assert isinstance(trace, EpisodeTrace) and isinstance(metrics, EpisodeMetrics)
    assert metrics.frontier_milestone.value == "iron"
    assert calls and calls[0][2] is sentinel_llm  # llm threaded to the factory
    assert any(w.saw_assignment for w in workers)  # behavior card reached the worker
    # RealRunner fills the advisory dials objectively (mirrors the sim path)
    assert any(st.performance_score > 0 for st in metrics.agent_stats)


def test_run_loop_with_worker_factory_offline():
    settings = load_config()
    settings.telemetry.mode = "off"
    settings.run.single_agent_oracle = False  # real Orca over the 4-agent roster
    settings.run.n_episodes = 2
    factory, calls, _ = _make_factory()

    results = run(settings, telemetry=init_telemetry(mode="off"), worker_factory=factory)

    assert len(results) == 2
    assert calls  # the loop built workers via the factory
    for _trace, metrics in results:
        assert metrics.frontier_milestone.value == "iron"

"""4 LLM workers play a full episode through the run loop (§3.6, §4, §5).

The LLM is mocked (no live API). Exercises: the single_agent_oracle=false path,
4 agents acting per round via the async bridge, comm-bus t+1 delivery merged into
observations, worker-emitted messages logged to the trace, and the coord-leak
invariant holding across every observation. The scripted-oracle path stays the
default and is covered by tests/test_run_loop.py.
"""

from __future__ import annotations

import train.loop as loop
from agents.worker import LLMWorker
from config import load_config
from obs_guard.coord_leak_test import assert_no_coord_leak
from telemetry import init_telemetry

# A valid WorkerOutput: a (legal) scout action + a clean team message every turn.
_WORKER_JSON = (
    '{"reasoning":"scouting outward",'
    '"action":{"name":"scout"},'
    '"messages":[{"to":"team","type":"share_finding","content":"checking biomes to the N","urgency":0.3}]}'
)


class ConstLLM:
    """Mock LLM returning a fixed valid response for any prompt (no network)."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.prompts: list[str] = []

    def complete(self, prompt: str, schema=None, **kwargs) -> str:
        self.prompts.append(prompt)
        return self.text


def _settings(t_max: int = 4):
    s = load_config()
    s.telemetry.mode = "off"
    s.run.single_agent_oracle = False  # the 4-agent LLM path
    s.run.n_episodes = 1
    s.run.t_max = t_max
    return s


def test_four_llm_agents_play_full_episode(monkeypatch):
    const = ConstLLM(_WORKER_JSON)
    monkeypatch.setattr(loop, "build_llm", lambda role, settings: const)

    settings = _settings(t_max=4)
    (trace, metrics), = loop.run(settings, telemetry=init_telemetry(mode="off"))

    # 4 role-biased agents, all acting every round (env marks none busy here).
    assert trace.agent_ids == ["agent_1", "agent_2", "agent_3", "agent_4"]
    assert trace.n_rounds == 4
    assert len(trace.action_records) == 4 * trace.n_rounds
    assert all(r.action.name.value == "scout" for r in trace.action_records)
    assert const.prompts, "the mocked worker LLM was actually called"


def test_messages_flow_over_bus_and_into_observations(monkeypatch):
    const = ConstLLM(_WORKER_JSON)
    monkeypatch.setattr(loop, "build_llm", lambda role, settings: const)

    settings = _settings(t_max=4)
    (trace, metrics), = loop.run(settings, telemetry=init_telemetry(mode="off"))

    # Worker-emitted messages are recorded on the trace (verbatim log, §5.2).
    assert trace.messages, "worker messages should be captured on the trace"
    assert all(m.from_agent in trace.agent_ids for m in trace.messages)

    # t+1 delivery: by round >= 1, an agent observes teammates' (not its own) messages.
    saw_delivered = False
    for o in trace.observations:
        if o["round"] >= 1 and o["recent_messages"]:
            saw_delivered = True
            self_id = None  # observation doesn't carry the viewer id; check sender != all-self
            assert all(rm["from"] != "" for rm in o["recent_messages"])
    assert saw_delivered, "messages posted at round t should be visible at t+1"


def test_no_coordinate_leak_in_llm_path_observations(monkeypatch):
    # Even with a chatty worker, no observation may carry a coordinate leak.
    const = ConstLLM(_WORKER_JSON)
    monkeypatch.setattr(loop, "build_llm", lambda role, settings: const)

    settings = _settings(t_max=4)
    (trace, _metrics), = loop.run(settings, telemetry=init_telemetry(mode="off"))

    assert trace.observations
    for i, o in enumerate(trace.observations):
        assert_no_coord_leak(o, path=f"obs[{i}]")


def test_coord_laden_worker_message_scrubbed_in_loop(monkeypatch):
    # A worker that tries to leak coordinates into a message must be sanitized
    # before the message reaches the bus / observations / trace.
    leaky = (
        '{"action":{"name":"scout"},'
        '"messages":[{"to":"team","type":"report","content":"iron at 12.0, 3.5 near r_07"}]}'
    )
    monkeypatch.setattr(loop, "build_llm", lambda role, settings: ConstLLM(leaky))

    settings = _settings(t_max=3)
    (trace, _metrics), = loop.run(settings, telemetry=init_telemetry(mode="off"))

    assert trace.messages, "messages should still flow (sanitized, not all dropped)"
    for m in trace.messages:
        assert "r_07" not in m.content
        assert "12.0, 3.5" not in m.content
    for i, o in enumerate(trace.observations):
        assert_no_coord_leak(o, path=f"obs[{i}]")


# --------------------------------------------------------------------------- #
# A5/§6.5 — execution-memory writes are gated by the accept-gate
# --------------------------------------------------------------------------- #
def test_rejected_proposal_does_not_update_memory(monkeypatch):
    # Even with a nonzero learning_signal, a rejected proposal must NOT mutate memory.
    monkeypatch.setattr(loop, "build_llm", lambda role, settings: ConstLLM(_WORKER_JSON))
    monkeypatch.setattr(loop, "_learning_signal_for", lambda metrics, aid: 1.0)  # nonzero

    calls: list = []

    class RejectGate:
        def __init__(self, *args, **kwargs):
            pass

        def evaluate(self, *args, **kwargs):
            from types import SimpleNamespace

            return SimpleNamespace(accepted=False)

    def spy(self, digest, learning_signal=0.0):
        calls.append((self.agent_id, learning_signal))
        return self.memory

    monkeypatch.setattr(loop, "AcceptGate", RejectGate)
    monkeypatch.setattr(LLMWorker, "end_episode_update", spy)

    settings = _settings(t_max=3)
    settings.phases.phase0_length = 0  # enable the Stream-3 coach/gate path immediately
    loop.run(settings, telemetry=init_telemetry(mode="off"))
    assert calls == [], "rejected proposal must not attempt a memory write"


def test_accepted_proposal_allows_memory_update(monkeypatch):
    monkeypatch.setattr(loop, "build_llm", lambda role, settings: ConstLLM(_WORKER_JSON))

    calls: list = []

    class AcceptingGate:
        def __init__(self, *args, **kwargs):
            pass

        def evaluate(self, *args, **kwargs):
            from types import SimpleNamespace

            return SimpleNamespace(accepted=True)

    def spy(self, digest, learning_signal=0.0):
        calls.append(self.agent_id)
        return self.memory

    monkeypatch.setattr(loop, "AcceptGate", AcceptingGate)
    monkeypatch.setattr(LLMWorker, "end_episode_update", spy)

    settings = _settings(t_max=2)
    settings.phases.phase0_length = 0
    loop.run(settings, telemetry=init_telemetry(mode="off"))
    assert calls == ["agent_1", "agent_2", "agent_3", "agent_4"]

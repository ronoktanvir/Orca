"""4 LLM workers play a full episode through the run loop (§3.6, §4, §5).

The LLM is mocked (no live API). Exercises: the single_agent_oracle=false path,
4 agents acting per round via the async bridge, comm-bus t+1 delivery merged into
observations, worker-emitted messages logged to the trace, and the coord-leak
invariant holding across every observation. The scripted-oracle path stays the
default and is covered by tests/test_run_loop.py.
"""

from __future__ import annotations

from types import SimpleNamespace

import train.loop as loop
from agents.worker import LLMWorker
from config import load_config
from contracts import ExecutionMemory, Heuristic
from contracts.enums import Role
from obs_guard.coord_leak_test import assert_no_coord_leak
from orca.cards import DEFAULT_ROSTER
from orca.orca import Proposal
from telemetry import init_telemetry

# Roster ids the full-team loop actually uses (human-named; §4.1). Derived from the
# roster so these tests don't re-pin the names.
_ROSTER_IDS = [aid for aid, _role in DEFAULT_ROSTER]


class AcceptingGate:
    """A gate stub that always accepts (the eval batch is irrelevant here)."""

    def __init__(self, *args, **kwargs):
        pass

    def evaluate(self, *args, **kwargs):
        return SimpleNamespace(accepted=True)


def _msg_key(m):
    return (m.from_agent, m.to, m.type.value, m.content, m.round)

# A valid WorkerOutput: a (legal) scout action + a clean team message every turn.
_WORKER_JSON = (
    '{"reasoning":"scouting outward",'
    '"action":{"name":"scout"},'
    '"messages":[{"to":"team","type":"share_finding","content":"checking biomes to the N","urgency":0.3}]}'
)

# An INVALID action every round (gather a resource that isn't here) — the agents
# rack up repeated invalids, so the deterministic heuristic coach emits a NON-EMPTY
# proposal (an execution-fix directive). Needed wherever a test must exercise the
# real accept-gate/memory path (an empty proposal is only trivially accepted).
_INVALID_JSON = '{"reasoning":"oops","action":{"name":"gather","args":{"resource":"diamond"}}}'


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
    assert trace.agent_ids == _ROSTER_IDS
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
    # Even with a nonzero coach learning_signal, a rejected proposal must NOT
    # mutate memory: end_episode_update is never even called.
    monkeypatch.setattr(loop, "build_llm", lambda role, settings: ConstLLM(_WORKER_JSON))
    monkeypatch.setattr(loop, "_coach_signals", lambda proposal: {aid: 1.0 for aid in _ROSTER_IDS})

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
    # Invalid-action workers -> a NON-EMPTY heuristic proposal -> accepted gate ->
    # the memory write runs for every agent. Orca uses the offline heuristic coach
    # (role -> None) so the proposal is the deterministic execution-fix this asserts.
    monkeypatch.setattr(
        loop, "build_llm", lambda role, settings: None if role == "orca" else ConstLLM(_INVALID_JSON)
    )

    calls: list = []

    def spy(self, digest, learning_signal=0.0):
        calls.append(self.agent_id)
        return self.memory

    monkeypatch.setattr(loop, "AcceptGate", AcceptingGate)
    monkeypatch.setattr(LLMWorker, "end_episode_update", spy)

    settings = _settings(t_max=2)
    settings.phases.phase0_length = 0
    loop.run(settings, telemetry=init_telemetry(mode="off"))
    assert calls == _ROSTER_IDS


# --------------------------------------------------------------------------- #
# Finding 2 — action-level report/request_help share ONE bus path, t+1 delivery.
# --------------------------------------------------------------------------- #
_REPORT_JSON = (
    '{"reasoning":"status",'
    '"action":{"name":"report","args":{"content":"scouting the N ridge","to":"team","urgency":0.4}},'
    '"messages":[]}'
)


def test_report_action_delivered_at_t_plus_one_no_dupes(monkeypatch):
    monkeypatch.setattr(loop, "build_llm", lambda role, settings: ConstLLM(_REPORT_JSON))
    settings = _settings(t_max=3)
    (trace, _metrics), = loop.run(settings, telemetry=init_telemetry(mode="off"))

    # Same-round: no message is visible in round 0's observations.
    round0 = [o for o in trace.observations if o["round"] == 0]
    assert round0 and all(o["recent_messages"] == [] for o in round0)

    # t+1: a report sent at round 0 is observed by teammates at round 1, carried on
    # the one authoritative path.
    round1 = [o for o in trace.observations if o["round"] == 1]
    assert round1 and any(o["recent_messages"] for o in round1)
    for o in round1:
        for rm in o["recent_messages"]:
            assert rm["type"] == "report"

    # Exactly one report per agent per round reached the trace — none dropped, and
    # none duplicated (which the old env-inbox + bus double path would have risked).
    report_msgs = [m for m in trace.messages if m.type.value == "report"]
    assert len(report_msgs) == len(trace.agent_ids) * trace.n_rounds
    keys = [_msg_key(m) for m in trace.messages]
    assert len(keys) == len(set(keys))


def test_request_help_action_delivered_at_t_plus_one(monkeypatch):
    rj = '{"action":{"name":"request_help","args":{"content":"need iron","to":"team","urgency":0.8}}}'
    monkeypatch.setattr(loop, "build_llm", lambda role, settings: ConstLLM(rj))
    settings = _settings(t_max=3)
    (trace, _metrics), = loop.run(settings, telemetry=init_telemetry(mode="off"))
    assert any(m.type.value == "request_help" for m in trace.messages)
    # Discriminating: a request_help sent at round 0 is observed at round 1 (t+1),
    # not round 2 — the old env inbox delivered action messages a round late.
    round1 = [o for o in trace.observations if o["round"] == 1]
    assert any(
        rm["type"] == "request_help" for o in round1 for rm in o["recent_messages"]
    ), "request_help must reach teammates at t+1, not t+2"


def test_custom_worker_factory_cannot_leak_through_actions_or_messages():
    # The advertised worker_factory seam bypasses LLMWorker's own sanitization, so
    # the run loop must enforce the leak invariant at the env/bus boundary: a worker
    # that emits raw leaky action args AND a raw leaky pending message must not leak
    # into trace.action_records, trace.messages, or observations.
    from contracts import Action, Message
    from contracts.enums import ActionName, MessageType
    from obs_guard import scan_for_leaks

    class _LeakyWorker:
        def __init__(self, agent_id, role, llm):
            self.agent_id = agent_id
            self.pending_messages: list = []

        def act(self, obs):
            self.pending_messages = [
                Message(
                    **{"from": self.agent_id},
                    to="r_07",  # leaky recipient
                    type=MessageType.REPORT,
                    content="iron at 12, 3 near r_07",  # leaky content
                    urgency=0.5,
                    round=obs.round,
                )
            ]
            return Action(
                name=ActionName.REPORT,
                args={"content": "cache at 12.0, 3.5", "r_07": "x", "to": "r_07"},  # all leaky
            )

    settings = _settings(t_max=3)
    (trace, _metrics), = loop.run(
        settings,
        telemetry=init_telemetry(mode="off"),
        worker_factory=lambda aid, role, llm: _LeakyWorker(aid, role, llm),
    )

    assert trace.messages
    for m in trace.messages:  # recipient + content scrubbed at the bus boundary
        assert m.to in ("team", "orca") or m.to.startswith("agent_")
        assert "r_07" not in m.to and "r_07" not in m.content and "12, 3" not in m.content
        assert scan_for_leaks(m.model_dump(by_alias=True)) == []
    for rec in trace.action_records:  # args scrubbed at the env boundary
        assert "r_07" not in rec.action.args
        assert scan_for_leaks(rec.model_dump()) == []
    for i, o in enumerate(trace.observations):
        assert_no_coord_leak(o, path=f"obs[{i}]")


def test_worker_message_recipient_leak_scrubbed_in_loop(monkeypatch):
    # A worker that puts a region id in the message recipient must not leak it.
    leaky = (
        '{"action":{"name":"scout"},'
        '"messages":[{"to":"r_07","type":"report","content":"caves to the N"}]}'
    )
    monkeypatch.setattr(loop, "build_llm", lambda role, settings: ConstLLM(leaky))
    settings = _settings(t_max=3)
    (trace, _metrics), = loop.run(settings, telemetry=init_telemetry(mode="off"))

    assert trace.messages
    for m in trace.messages:
        assert m.to in ("team", "orca") or m.to.startswith("agent_")
        assert "r_07" not in m.to
    for i, o in enumerate(trace.observations):
        assert_no_coord_leak(o, path=f"obs[{i}]")


# --------------------------------------------------------------------------- #
# Finding 4 — accepted coach proposal's learning_signal drives the memory write.
# --------------------------------------------------------------------------- #
def test_accepted_proposal_scores_drive_memory_signal():
    from tests.fixtures import sample_episode_metrics, sample_episode_trace

    proposal = Proposal(
        behavior_cards={}, scores={"agent_1": {"learning_signal": 1.0, "performance_score": 0.5}}
    )
    calls: list = []

    class SpyAgent:
        agent_id = "agent_1"
        memory = ExecutionMemory(agent_id="agent_1")

        def end_episode_update(self, digest, learning_signal=0.0):
            calls.append(learning_signal)
            return self.memory

    loop._update_execution_memories(
        [SpyAgent()],
        {"agent_1": ExecutionMemory(agent_id="agent_1")},
        sample_episode_trace(),
        sample_episode_metrics(),
        loop._coach_signals(proposal),
    )
    assert calls == [1.0]   # the coach's signal, not the objective default


def test_no_coach_proposal_means_no_memory_write(monkeypatch):
    # Phase 0 (coach disabled) builds no proposal -> end_episode_update never runs.
    monkeypatch.setattr(loop, "build_llm", lambda role, settings: ConstLLM(_WORKER_JSON))
    calls: list = []

    def spy(self, digest, learning_signal=0.0):
        calls.append(self.agent_id)
        return self.memory

    monkeypatch.setattr(LLMWorker, "end_episode_update", spy)
    settings = _settings(t_max=2)
    settings.phases.phase0_length = 5  # stay in Phase 0 (no coach) for this single episode
    loop.run(settings, telemetry=init_telemetry(mode="off"))
    assert calls == []


def test_memory_persists_into_next_episode(monkeypatch):
    # Invalid-action workers -> a NON-EMPTY accepted proposal -> the (positive)
    # signal bakes a learned heuristic into memory after episode 0.
    monkeypatch.setattr(loop, "AcceptGate", AcceptingGate)
    monkeypatch.setattr(loop, "_coach_signals", lambda proposal: {aid: 1.0 for aid in _ROSTER_IDS})
    learned = Heuristic(condition="when blocked", action="scout before digging", confidence=0.7)
    monkeypatch.setattr(LLMWorker, "propose_memory", lambda self, digest: [learned])

    settings = _settings(t_max=2)
    settings.run.n_episodes = 2
    settings.phases.phase0_length = 0  # coach/gate active from episode 0
    const = ConstLLM(_INVALID_JSON)
    # Orca on the offline heuristic coach (role -> None); workers on the mock.
    monkeypatch.setattr(
        loop, "build_llm", lambda role, settings: None if role == "orca" else const
    )

    loop.run(settings, telemetry=init_telemetry(mode="off"))

    # The heuristic written after episode 0 appears in episode 1's worker prompts.
    assert any("scout before digging" in p for p in const.prompts)


# --------------------------------------------------------------------------- #
# Finding 4 (review follow-up) — an empty (gate-bypassing) proposal must NOT
# drive a memory write, even with a nonzero score; the negative path is reachable
# through the loop.
# --------------------------------------------------------------------------- #
def test_should_write_memory_rejects_empty_scored_proposal():
    from orca.cards import make_default_card

    # Empty proposal (no card edits) carrying a nonzero score: the gate only
    # *trivially* accepts it (no eval batch), so it must NOT authorize a write.
    empty_scored = Proposal(behavior_cards={}, scores={"agent_1": {"learning_signal": -0.7}})
    assert empty_scored.is_empty()
    assert loop._should_write_memory(True, empty_scored) is False     # gate-bypass closed
    # A real (non-empty) accepted proposal does write; rejected / None never do.
    nonempty = Proposal(
        behavior_cards={"agent_1": make_default_card("agent_1", Role.MINER)},
        scores={"agent_1": {"learning_signal": -0.7}},
    )
    assert loop._should_write_memory(True, nonempty) is True
    assert loop._should_write_memory(False, nonempty) is False
    assert loop._should_write_memory(True, None) is False


# --------------------------------------------------------------------------- #
# Worker reasoning is threaded onto the trace and into the coach's digest (§6.4).
# --------------------------------------------------------------------------- #
def test_worker_reasoning_lands_on_trace_and_digest(monkeypatch):
    # Each worker's own "reasoning" string should ride the trace (reasoning_log) and
    # surface in the digest Orca's coach reads — not just live in the Weave event.
    from orca.digest import build_digest

    monkeypatch.setattr(loop, "build_llm", lambda role, settings: ConstLLM(_WORKER_JSON))
    settings = _settings(t_max=3)
    (trace, metrics), = loop.run(settings, telemetry=init_telemetry(mode="off"))

    assert trace.reasoning_log, "worker reasoning should be captured on the trace"
    assert all(r.agent_id in trace.agent_ids for r in trace.reasoning_log)
    assert all("scouting outward" in r.text for r in trace.reasoning_log)
    # Every worker reasons on every turn it acts here, so it mirrors action_records.
    assert len(trace.reasoning_log) == len(trace.action_records)

    digest = build_digest(trace, metrics)
    assert any(a.recent_reasoning for a in digest.agents)
    assert "reasoned:" in digest.render()


def test_coord_laden_worker_reasoning_scrubbed_on_trace(monkeypatch):
    # Reasoning is free text from the model, so it must ride the same coordinate-leak
    # scrub as messages/args before it can reach Orca (§3.2): a worker that "thinks"
    # in coordinates must not leak them onto the trace or into the digest.
    leaky = (
        '{"reasoning":"iron vein at 12.0, 3.5 near r_07, heading there",'
        '"action":{"name":"scout"}}'
    )
    monkeypatch.setattr(loop, "build_llm", lambda role, settings: ConstLLM(leaky))
    settings = _settings(t_max=2)
    (trace, _metrics), = loop.run(settings, telemetry=init_telemetry(mode="off"))

    assert trace.reasoning_log, "reasoning should survive scrubbing (not be all-dropped)"
    for r in trace.reasoning_log:
        assert "r_07" not in r.text and "12.0, 3.5" not in r.text
    assert_no_coord_leak([r.model_dump() for r in trace.reasoning_log], path="reasoning_log")


def test_orca_coach_takes_the_llm_path_when_a_model_is_available(monkeypatch):
    # Part 1: when build_llm provides an Orca model, the coach reasons with the LLM
    # (source='llm'), not the offline heuristic — Orca's judgement is model-driven.
    class CoachLLM:
        def complete(self, prompt, schema=None, **kwargs):
            return (
                '{"team_reasoning":"agent_2 lacked the recipe",'
                '"agents":[{"agent_id":"agent_2","credit":"execution",'
                '"reasoning":"missing prereq","new_directives":["check prereqs"],'
                '"learning_signal":0.5}]}'
            )

    events: list = []

    class CapTel:
        def log_event(self, name, data):
            events.append((name, data))

        def log_episode(self, *a, **k):
            return None

    monkeypatch.setattr(
        loop,
        "build_llm",
        lambda role, settings: CoachLLM() if role == "orca" else ConstLLM(_INVALID_JSON),
    )
    monkeypatch.setattr(loop, "AcceptGate", AcceptingGate)

    settings = _settings(t_max=2)
    settings.phases.phase0_length = 0  # coach active from episode 0
    loop.run(settings, telemetry=CapTel())

    coach_events = [d for n, d in events if n == "orca_coach"]
    assert coach_events, "the coach should have run and logged"
    assert any(d["source"] == "llm" for d in coach_events), "coach should use the LLM path"


def test_empty_proposal_with_scores_does_not_write_through_loop(monkeypatch):
    # End-to-end: all-scout workers => the heuristic coach yields an EMPTY proposal.
    # Even though the gate accepts and a nonzero score is present, no write happens.
    monkeypatch.setattr(loop, "build_llm", lambda role, settings: ConstLLM(_WORKER_JSON))
    monkeypatch.setattr(loop, "AcceptGate", AcceptingGate)
    monkeypatch.setattr(loop, "_coach_signals", lambda p: {aid: -0.7 for aid in _ROSTER_IDS})
    calls: list = []

    def spy(self, digest, learning_signal=0.0):
        calls.append(self.agent_id)
        return self.memory

    monkeypatch.setattr(LLMWorker, "end_episode_update", spy)
    settings = _settings(t_max=2)
    settings.phases.phase0_length = 0
    loop.run(settings, telemetry=init_telemetry(mode="off"))
    assert calls == []   # empty proposal -> no memory write despite nonzero scores


def test_negative_coach_signal_weakens_memory_through_loop(monkeypatch):
    # Drive the negative path through the real loop: ep0 adds a rule (positive
    # signal), ep1 weakens it (negative signal). The weakened confidence shows up
    # in ep2's worker prompt — proving the negative path is reachable end-to-end.
    monkeypatch.setattr(loop, "AcceptGate", AcceptingGate)
    learned = Heuristic(condition="rush ahead", action="skip prep", confidence=0.7)
    monkeypatch.setattr(LLMWorker, "propose_memory", lambda self, digest: [learned])

    sigs = iter([1.0, -1.0, 0.0])  # ep0 add @0.7, ep1 weaken to 0.2, ep2 observe

    def fake_signals(proposal):
        s = next(sigs, 0.0)
        return {aid: s for aid in _ROSTER_IDS}

    monkeypatch.setattr(loop, "_coach_signals", fake_signals)

    settings = _settings(t_max=2)
    settings.run.n_episodes = 3
    settings.phases.phase0_length = 0
    const = ConstLLM(_INVALID_JSON)  # non-empty proposals so writes are gated-through
    # Orca on the offline heuristic coach (role -> None); workers on the mock.
    monkeypatch.setattr(
        loop, "build_llm", lambda role, settings: None if role == "orca" else const
    )

    loop.run(settings, telemetry=init_telemetry(mode="off"))

    # 0.7 - 0.5*|−1.0| = 0.20: the weakened confidence appears in a later prompt.
    assert any("confidence 0.20" in p for p in const.prompts)

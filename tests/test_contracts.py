"""The seven frozen contracts: importable + sample-validating + round-tripping (§11)."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, ValidationError

from contracts import (
    CONTRACTS,
    Action,
    BehaviorCard,
    EpisodeMetrics,
    EpisodeTrace,
    ExecutionMemory,
    Heuristic,
    Message,
    Observation,
)
from contracts.enums import ActionName, MessageType
from tests.fixtures import SAMPLES


def test_seven_contracts_present_and_importable():
    assert len(CONTRACTS) == 7
    expected = {
        Observation,
        Action,
        Message,
        BehaviorCard,
        ExecutionMemory,
        EpisodeTrace,
        EpisodeMetrics,
    }
    assert set(CONTRACTS) == expected
    assert all(issubclass(c, BaseModel) for c in CONTRACTS)


@pytest.mark.parametrize("name", list(SAMPLES))
def test_sample_validates(name):
    cls, factory = SAMPLES[name]
    inst = factory()
    assert isinstance(inst, cls)


@pytest.mark.parametrize("name", list(SAMPLES))
def test_sample_json_round_trips(name):
    cls, factory = SAMPLES[name]
    inst = factory()
    dumped = inst.model_dump_json(by_alias=True)
    revived = cls.model_validate_json(dumped)
    # Re-serialize and compare structurally (order-independent).
    assert json.loads(revived.model_dump_json(by_alias=True)) == json.loads(dumped)


def test_message_from_alias():
    m = Message(**{"from": "agent_1"}, to="team", type=MessageType.ACK, content="ok", round=3)
    assert m.from_agent == "agent_1"
    assert m.model_dump(by_alias=True)["from"] == "agent_1"
    # constructible by python name too
    m2 = Message(from_agent="agent_2", to="orca", type=MessageType.REPORT, content="x", round=1)
    assert m2.from_agent == "agent_2"


def test_observation_self_alias_and_no_pos():
    o = SAMPLES["Observation"][1]()
    dumped = o.model_dump(by_alias=True)
    assert "self" in dumped and "self_view" not in dumped
    # extra fields (e.g. a smuggled coordinate) are forbidden at construction
    with pytest.raises(ValidationError):
        Observation(
            round=0,
            time_of_day="day",
            self={"role": "miner", "health": 1.0, "hunger": 1.0, "current_biome": "forest", "pos": [1.0, 2.0]},
            here={},
        )


def test_action_rejects_unknown_name():
    with pytest.raises(ValidationError):
        Action(name="teleport")
    assert Action(name=ActionName.WAIT).name == ActionName.WAIT


def test_execution_memory_cap_enforced():
    # 8 is fine; 9 must fail (the bounded list is part of the contract, §4.5).
    ok = ExecutionMemory(
        agent_id="a",
        heuristics=[Heuristic(condition="c", action="a", confidence=0.5) for _ in range(8)],
    )
    assert len(ok.heuristics) == 8
    with pytest.raises(ValidationError):
        ExecutionMemory(
            agent_id="a",
            heuristics=[Heuristic(condition="c", action="a", confidence=0.5) for _ in range(9)],
        )


def test_bounded_floats_rejected():
    with pytest.raises(ValidationError):
        Message(**{"from": "a"}, to="team", type=MessageType.REPORT, content="x", urgency=1.5, round=0)
    with pytest.raises(ValidationError):
        Heuristic(condition="c", action="a", confidence=2.0)

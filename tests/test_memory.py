"""Execution-memory guard filter + learning-signal write (§4.5).

Covers the leak detector/redactor, the guard filter, and ``update_execution_memory``
(the agent-owned episode-end write modulated by Orca's learning_signal).
"""

from __future__ import annotations

from agents.memory import (
    guard_filter,
    looks_seed_specific,
    scrub_seed_specific,
    update_execution_memory,
)
from contracts import ExecutionMemory, Heuristic
from contracts.execution_memory import MEMORY_CAP


def test_looks_seed_specific_detects_coords_and_ids():
    assert looks_seed_specific("go to 12.0, 3.5")
    assert looks_seed_specific("the cave at (4.0, -2.0)")
    assert looks_seed_specific("iron is in r_07")
    assert looks_seed_specific("travel 40 blocks north")
    assert looks_seed_specific("Region.pos has the answer")


def test_transferable_heuristics_are_kept():
    assert not looks_seed_specific("need iron but only wooden pickaxe")
    assert not looks_seed_specific("craft stone pickaxe before mining iron")


def test_integer_coordinate_pairs_detected_ids_preserved():
    # Bare integer coordinate-like pairs the float pattern misses must be caught...
    assert looks_seed_specific("base at 12, 3")
    assert looks_seed_specific("waypoint 12;3")
    assert looks_seed_specific("go to -4, 7")
    # ...without flagging legitimate agent ids or lone counts.
    assert not looks_seed_specific("agent_2")
    assert not looks_seed_specific("hand off to agent_12")
    assert not looks_seed_specific("gather 6 ingots")


def test_scrub_removes_integer_pairs_but_keeps_meaning():
    cleaned = scrub_seed_specific("iron cache at 12, 3 to the north")
    assert not looks_seed_specific(cleaned)
    assert "iron" in cleaned and "north" in cleaned


def test_guard_filter_drops_only_leaky_rules():
    mem = ExecutionMemory(
        agent_id="agent_2",
        heuristics=[
            Heuristic(condition="need iron, only wooden pickaxe", action="craft stone pickaxe first", confidence=0.8),
            Heuristic(condition="iron at r_07", action="walk 40 blocks N to 12.0, 3.5", confidence=0.9),
        ],
    )
    filtered = guard_filter(mem)
    assert len(filtered.heuristics) == 1
    assert filtered.heuristics[0].action == "craft stone pickaxe first"
    assert filtered.agent_id == "agent_2"


# --------------------------------------------------------------------------- #
# scrub: sanitize (redact) coord-like spans rather than drop the whole string
# --------------------------------------------------------------------------- #
def test_scrub_removes_coord_like_spans():
    cleaned = scrub_seed_specific("iron at 12.0, 3.5 near r_07, travel 40 blocks N")
    assert not looks_seed_specific(cleaned)
    assert "iron" in cleaned and "N" in cleaned  # the meaning survives


def test_scrub_leaves_clean_text_untouched():
    assert scrub_seed_specific("craft stone pickaxe before mining iron") == (
        "craft stone pickaxe before mining iron"
    )


# --------------------------------------------------------------------------- #
# update_execution_memory: learning_signal modulates edit strength (§4.5)
# --------------------------------------------------------------------------- #
def _h(cond, act, conf=0.5):
    return Heuristic(condition=cond, action=act, confidence=conf)


def test_neutral_signal_makes_no_change():
    mem = ExecutionMemory(agent_id="a", heuristics=[_h("c1", "a1", 0.5)])
    out = update_execution_memory(mem, [_h("c2", "a2", 0.9)], learning_signal=0.0)
    assert [(_x.condition, _x.action) for _x in out.heuristics] == [("c1", "a1")]


def test_positive_signal_adds_new_heuristics():
    mem = ExecutionMemory(agent_id="a", heuristics=[])
    out = update_execution_memory(mem, [_h("c1", "a1", 0.6), _h("c2", "a2", 0.7)], learning_signal=1.0)
    conds = {h.condition for h in out.heuristics}
    assert conds == {"c1", "c2"}


def test_positive_signal_strengthens_existing():
    mem = ExecutionMemory(agent_id="a", heuristics=[_h("c1", "a1", 0.5)])
    out = update_execution_memory(mem, [_h("c1", "a1", 0.5)], learning_signal=1.0)
    assert out.heuristics[0].confidence > 0.5  # bumped, not duplicated
    assert len(out.heuristics) == 1


def test_negative_signal_removes_flagged_low_confidence():
    mem = ExecutionMemory(agent_id="a", heuristics=[_h("bad", "do bad", 0.3)])
    out = update_execution_memory(mem, [_h("bad", "do bad", 0.3)], learning_signal=-1.0)
    assert out.heuristics == []  # weakened below floor -> removed


def test_negative_signal_weakens_flagged_high_confidence():
    mem = ExecutionMemory(agent_id="a", heuristics=[_h("keep", "do it", 0.9)])
    out = update_execution_memory(mem, [_h("keep", "do it", 0.9)], learning_signal=-0.4)
    assert len(out.heuristics) == 1
    assert out.heuristics[0].confidence < 0.9


def test_coord_laden_proposal_never_persists():
    mem = ExecutionMemory(agent_id="a", heuristics=[])
    proposed = [_h("iron at r_07", "walk 40 blocks N to 12.0, 3.5", 0.9)]
    out = update_execution_memory(mem, proposed, learning_signal=1.0)
    assert out.heuristics == []  # guard-filtered out


def test_cap_enforced_keeps_highest_confidence():
    existing = [_h(f"c{i}", f"a{i}", 0.5) for i in range(MEMORY_CAP)]
    mem = ExecutionMemory(agent_id="a", heuristics=existing)
    proposed = [_h("new1", "na1", 0.95), _h("new2", "na2", 0.95)]
    out = update_execution_memory(mem, proposed, learning_signal=1.0)
    assert len(out.heuristics) == MEMORY_CAP
    conds = {h.condition for h in out.heuristics}
    assert "new1" in conds and "new2" in conds  # high-confidence newcomers retained

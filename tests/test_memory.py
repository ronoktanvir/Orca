"""Execution-memory guard filter strips coord-like / seed-specific content (§4.5)."""

from __future__ import annotations

from agents.memory import guard_filter, looks_seed_specific
from contracts import ExecutionMemory, Heuristic


def test_looks_seed_specific_detects_coords_and_ids():
    assert looks_seed_specific("go to 12.0, 3.5")
    assert looks_seed_specific("the cave at (4.0, -2.0)")
    assert looks_seed_specific("iron is in r_07")
    assert looks_seed_specific("travel 40 blocks north")
    assert looks_seed_specific("Region.pos has the answer")


def test_transferable_heuristics_are_kept():
    assert not looks_seed_specific("need iron but only wooden pickaxe")
    assert not looks_seed_specific("craft stone pickaxe before mining iron")


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

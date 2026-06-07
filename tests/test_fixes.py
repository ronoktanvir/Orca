"""Regression tests for the craft-loop deadlock fix + the named roster.

Covers three changes:
  1. recipe-name aliasing so LLM-correct names (``wooden_planks``) craft (§3.4),
  2. obs.last_action feeding the env's rejection reason back to the worker (§3.3),
  3. the human-named roster + recipient allow-list staying in sync (§4.1/§5.1).
"""

from __future__ import annotations

from bus.messages import _NAMED_AGENTS, normalize_recipient
from contracts import Action
from contracts.enums import ActionName, Role
from env import StubEnv
from env.techtree import canonical_item, craft_check
from orca.cards import NAME_BY_ROLE


# --- 1. recipe aliasing ---------------------------------------------------- #
def test_canonical_item_maps_synonyms():
    assert canonical_item("wooden_planks") == "planks"
    assert canonical_item("Crafting Table") == "crafting_table"
    assert canonical_item("stick") == "sticks"
    assert canonical_item("planks") == "planks"  # already canonical
    assert canonical_item("totally_made_up") == "totally_made_up"  # honest passthrough


def test_craft_check_accepts_alias():
    ok, _ = craft_check("wooden_planks", {"wood": 1})
    assert ok
    ok2, reason = craft_check("totally_made_up", {"wood": 9})
    assert not ok2 and "no recipe" in reason


def test_craft_alias_resolves_in_env():
    env = StubEnv(seed="A", agents=[("agent_1", Role.MINER)])
    env.reset()
    env.world.agents["agent_1"].inventory["wood"] = 1
    res = env.step({"agent_1": Action(name=ActionName.CRAFT, args={"item": "wooden_planks"})})
    assert res.records[0].valid
    assert env.world.agents["agent_1"].inventory.get("planks", 0) == 4


# --- 2. last-action feedback ---------------------------------------------- #
def test_last_action_none_on_first_obs():
    env = StubEnv(seed="A", agents=[("agent_1", Role.MINER)])
    env.reset()
    assert env.observe("agent_1").last_action is None


def test_invalid_action_surfaces_reason_next_obs():
    env = StubEnv(seed="A", agents=[("agent_1", Role.MINER)])
    env.reset()
    # crafting_table needs 4 planks; with none it is rejected (the deadlock seed).
    env.step({"agent_1": Action(name=ActionName.CRAFT, args={"item": "crafting_table"})})
    la = env.observe("agent_1").last_action
    assert la is not None and la.name == "craft" and la.valid is False
    assert "planks" in (la.reason or "")


def test_valid_action_marked_ok_next_obs():
    env = StubEnv(seed="A", agents=[("agent_1", Role.MINER)])
    env.reset()
    env.step({"agent_1": Action(name=ActionName.GATHER, args={"resource": "wood"})})
    la = env.observe("agent_1").last_action
    assert la is not None and la.valid is True


# --- 3. named roster / recipient allow-list ------------------------------- #
def test_named_recipients_not_downgraded():
    for name in NAME_BY_ROLE.values():
        assert normalize_recipient(name) == name
    # back-compat + leak rejection still hold
    assert normalize_recipient("agent_2") == "agent_2"
    assert normalize_recipient("r_07") == "team"


def test_recipient_allowlist_matches_roster():
    # Drift guard: every roster name must be an accepted recipient.
    assert set(NAME_BY_ROLE.values()) <= set(_NAMED_AGENTS)

"""The coordinate-leak invariant (§3.2, §11) — the hard guard.

This module is both:
  * a reusable scanner (:func:`scan_for_leaks` / :func:`assert_no_coord_leak`)
    other code can call, and
  * a pytest module (the ``test_*`` functions below) collected via the
    ``testpaths = ["tests", "obs_guard"]`` setting.

It asserts that **no float pair, no ``pos``, and no internal region id ever
appears in a serialized observation** (§3.2). This is the third leg of the
three-layer invariant (contracts forbid extras; ``serialize_observation`` never
reads ``.pos``; this test scans the output).
"""

from __future__ import annotations

import re
from typing import Any

from contracts.enums import Role
from env import StubEnv
from env.seeds import make_world

# Patterns that constitute a leak.
_REGION_ID = re.compile(r"\br_\d+\b")
_FLOAT_PAIR = re.compile(r"-?\d+\.\d+\s*[,;]\s*-?\d+\.\d+")


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def scan_for_leaks(obj: Any, path: str = "obs") -> list[str]:
    """Recursively scan a serialized structure for coordinate leaks.

    Accepts a pydantic model, dict, list, or scalar. Returns a list of
    human-readable leak descriptions (empty == clean).
    """
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump(mode="json", by_alias=True)

    leaks: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            ks = str(key)
            if ks.lower() == "pos":
                leaks.append(f"{path}: forbidden 'pos' key")
            if _REGION_ID.search(ks):
                leaks.append(f"{path}: internal region id in key '{ks}'")
            leaks += scan_for_leaks(value, f"{path}.{ks}")
    elif isinstance(obj, (list, tuple)):
        # A bare 2-element numeric list/tuple looks exactly like a coordinate.
        if len(obj) == 2 and all(_is_number(x) for x in obj):
            leaks.append(f"{path}: numeric pair {list(obj)!r} looks like coordinates")
        for i, value in enumerate(obj):
            leaks += scan_for_leaks(value, f"{path}[{i}]")
    elif isinstance(obj, str):
        if obj.lower() == "pos":
            leaks.append(f"{path}: string value 'pos'")
        if _REGION_ID.search(obj):
            leaks.append(f"{path}: internal region id in string {obj!r}")
        if _FLOAT_PAIR.search(obj):
            leaks.append(f"{path}: float pair in string {obj!r}")
    return leaks


def assert_no_coord_leak(obj: Any, path: str = "obs") -> None:
    """Raise AssertionError listing any coordinate leaks found."""
    leaks = scan_for_leaks(obj, path)
    assert not leaks, "coordinate leak(s) detected:\n  " + "\n  ".join(leaks)


# --------------------------------------------------------------------------- #
# pytest tests
# --------------------------------------------------------------------------- #
def _run_oracle_episode():
    from agents.scripted import ShallowOracle

    env = StubEnv(seed="A", agents=[("agent_1", Role.MINER)], t_max=200)
    env.reset()
    oracle = ShallowOracle("agent_1")
    snapshots = []
    for _ in range(200):
        if env.done:
            break
        obs = env.observe("agent_1")
        snapshots.append(obs)
        env.step({"agent_1": oracle.act(obs)})
    return env, snapshots


def test_single_observation_no_leak():
    env = StubEnv(seed="A", agents=[("agent_1", Role.MINER)])
    env.reset()
    obs = env.observe("agent_1")
    assert_no_coord_leak(obs)


def test_full_episode_observations_no_leak():
    _env, snapshots = _run_oracle_episode()
    assert len(snapshots) > 0
    for i, obs in enumerate(snapshots):
        assert_no_coord_leak(obs, path=f"obs[{i}]")


def test_region_ids_present_internally_but_never_in_obs():
    # The world *does* use internal region ids (sanity check the test is real)...
    world = make_world("A")
    assert any(_REGION_ID.search(rid) for rid in world.regions)
    # ...but they must never appear in any serialized observation.
    _env, snapshots = _run_oracle_episode()
    for obs in snapshots:
        dumped = obs.model_dump(mode="json", by_alias=True)
        assert not _REGION_ID.search(str(dumped))


def test_scanner_catches_a_planted_leak():
    # Negative control: the scanner must actually fire on a coordinate.
    leaky = {"here": {"pos": [12.0, 3.5]}, "note": "near r_07 at 4.0, 2.0"}
    leaks = scan_for_leaks(leaky)
    assert any("pos" in s for s in leaks)
    assert any("region id" in s for s in leaks)
    assert any("coordinates" in s or "float pair" in s for s in leaks)


# --------------------------------------------------------------------------- #
# E2: the richer observation (landmarks, mobs, layer transitions) must stay clean
# --------------------------------------------------------------------------- #
def test_rich_obs_with_landmarks_and_mobs_no_leak():
    # Reveal a neighbor that carries a lava_pool (populates known_landmarks) and
    # observe at night (populates mobs) — the richer obs must not leak coords.
    from env.observation import serialize_observation
    from env.world import AgentState

    world = make_world("A")
    world.add_agent(AgentState(agent_id="a", role=Role.MINER, region_id="r_00"))
    world.regions["r_07"].discovered = True  # caves w/ lava_pool, neighbor of start
    obs = serialize_observation(world, "a", round_idx=60, day_length=100)  # round 60 -> night
    assert obs.known_landmarks, "landmarks should be populated (test is real)"
    assert obs.here.mobs, "night mobs should be populated (test is real)"
    assert_no_coord_leak(obs)


def test_obs_in_nether_after_portal_no_leak():
    # Build+light+enter a nether portal, then observe in the Nether (new layer
    # state + nether mob path) — still coordinate-clean.
    from random import Random

    from contracts import Action
    from contracts.enums import ActionName
    from env.actions import resolve_action
    from env.observation import serialize_observation
    from env.world import AgentState

    world = make_world("A")
    agent = AgentState(agent_id="a", role=Role.TINKERER, region_id="r_00", inventory={"nether_portal": 1})
    world.add_agent(agent)
    resolve_action(world, agent, Action(name=ActionName.PLACE, args={"item": "nether_portal"}), Random(0), 0)
    resolve_action(world, agent, Action(name=ActionName.MOVE, args={"to": "nether"}), Random(0), 1)
    obs = serialize_observation(world, "a", round_idx=1, day_length=100)
    assert obs.self_view.layer.value == "nether"
    assert_no_coord_leak(obs)


__all__ = ["scan_for_leaks", "assert_no_coord_leak"]

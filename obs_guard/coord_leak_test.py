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
# Bare integer coordinate-like pairs, e.g. "12, 3" / "12;3" — kept in sync with
# ``agents.memory.looks_seed_specific`` so the hard scanner has no blind spot for
# integer coordinates (legitimate ids like "agent_2" / lone counts are untouched).
_INT_PAIR = re.compile(r"-?\d+\s*[,;]\s*-?\d+")


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
        elif _INT_PAIR.search(obj):  # elif: a float pair already covers its int spans
            leaks.append(f"{path}: integer pair in string {obj!r}")
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


def test_scanner_catches_bare_integer_coordinate_pair():
    # The hard scanner must catch integer coordinate strings ("12, 3"), in sync
    # with agents.memory.looks_seed_specific — no blind spot for integer coords.
    assert any("integer pair" in s for s in scan_for_leaks({"content": "iron at 12, 3"}))
    assert any("integer pair" in s for s in scan_for_leaks({"note": "go to -4;7"}))
    # ...but legitimate agent ids and lone counts are NOT flagged.
    assert scan_for_leaks({"a": "agent_2", "b": "gather 6 logs", "c": "head N"}) == []


# --------------------------------------------------------------------------- #
# Deep-env coverage (Stream 1 E2–E6). The ShallowOracle episode above only ever
# reaches IRON in the Overworld, so the scanner never sees the NEW deep obs
# surfaces — Nether/End mobs (piglin/blaze/ender_dragon) and the perceived
# landmarks (fortress/stronghold/portal/lava_pool). These drive the full-DAG
# oracle to build a real dragon-defeated world, reveal it, and scan an
# observation from every region (day + night) so Law 4 (§3.2) is actually
# enforced on the deepened observation — not just the shallow slice.
# --------------------------------------------------------------------------- #
def _deep_snapshots(seed: str = "A"):
    """``(won, snapshots, surfaces)`` from a fully-explored, dragon-defeated world.

    Plays the full-DAG oracle (so portals are lit and structures/layers are real),
    marks every region discovered to maximize the perception surface, then walks
    the agent through every region and serializes via the real ``env.observe``
    path at a day round and a night round. ``surfaces`` is the set of deep
    mob/landmark tokens actually produced (used to prove the scan isn't vacuous).
    """
    from contracts.enums import TimeOfDay
    from env.observation import time_of_day
    from env.oracle import FullDagOracle

    env = StubEnv(seed=seed, agents=[("oracle", Role.TINKERER)], t_max=8000, stop_at_milestone=None)
    env.reset()
    won = FullDagOracle("oracle").solve(env)
    for region in env.world.regions.values():
        region.discovered = True  # reveal everything so all landmarks/mobs surface
    agent = env.world.agents["oracle"]

    dl = env.day_length
    day_r = next(r for r in range(dl) if time_of_day(r, dl) == TimeOfDay.DAY)
    night_r = next((r for r in range(dl) if time_of_day(r, dl) == TimeOfDay.NIGHT), day_r)

    snapshots, surfaces = [], set()
    for rid in env.world.regions:
        agent.region_id = rid
        for round_idx in (day_r, night_r):
            env.round_idx = round_idx
            obs = env.observe("oracle")
            snapshots.append((rid, round_idx, obs))
            dumped = obs.model_dump(mode="json", by_alias=True)
            surfaces.update(("mob", m) for m in dumped["here"]["mobs"])
            surfaces.update(("landmark", lm["type"]) for lm in dumped["known_landmarks"])
    return won, snapshots, surfaces


def test_deep_world_observations_no_leak():
    # The deep counterpart to test_full_episode_observations_no_leak: every seed,
    # every region of a dragon-defeated world, day + night — none may leak (§3.2).
    from env.seeds import ALL_SEEDS

    scanned = 0
    for seed in ALL_SEEDS:
        won, snapshots, _surfaces = _deep_snapshots(seed)
        assert won, f"oracle did not reach the dragon on seed {seed!r} — cannot scan deep obs"
        for rid, round_idx, obs in snapshots:
            assert_no_coord_leak(obs, path=f"{seed}:{rid}@r{round_idx}")
            scanned += 1
    assert scanned > 0


def test_deep_observation_surfaces_are_actually_exercised():
    # Negative-space guard: the deep mobs + landmarks the shallow episode never
    # produces must actually appear in what we scanned — otherwise the no-leak
    # test above could pass vacuously on empty surfaces.
    _won, _snapshots, surfaces = _deep_snapshots("A")
    for surface in (
        ("mob", "blaze"),          # fortress (Nether)
        ("mob", "ender_dragon"),   # the End
        ("landmark", "fortress"),  # structure landmark from a discovered neighbor
        ("landmark", "portal"),    # a lit nether/end portal seen from a neighbor
    ):
        assert surface in surfaces, f"deep surface {surface} not exercised; saw {sorted(surfaces)}"


__all__ = ["scan_for_leaks", "assert_no_coord_leak"]

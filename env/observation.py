"""The ONE and ONLY observation path: ``serialize_observation`` (§3.2).

This function is the single place an :class:`Observation` is ever built. It has
**no access to ``Region.pos``** — it reaches the world solely through the
coordinate-free perception helpers on :class:`~env.world.World`. Do not add a
``pos`` read here; the coord-leak invariant (§3.2) depends on it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from contracts import (
    Exit,
    HereView,
    Landmark,
    Message,
    Observation,
    SelfView,
    TeammateView,
)
from contracts.enums import Layer, Milestone, Structure, TimeOfDay

if TYPE_CHECKING:  # avoid import cycle at runtime; type-only
    from .world import Region, World


def time_of_day(round_idx: int, day_length: int) -> TimeOfDay:
    """Map a round into a coarse time-of-day (§3.2)."""
    phase = (round_idx % max(1, day_length)) / max(1, day_length)
    if phase < 0.45:
        return TimeOfDay.DAY
    if phase < 0.55:
        return TimeOfDay.DUSK
    if phase < 0.95:
        return TimeOfDay.NIGHT
    return TimeOfDay.DAWN


def perceived_mobs(region: "Region", tod: TimeOfDay) -> list[str]:
    """Mobs visible in a region this round (§3.2) — perception only; combat is
    E4/E5. Deterministic from layer/structure/time (no RNG)."""
    if region.layer == Layer.END:
        return ["ender_dragon"]
    if region.structure == Structure.FORTRESS:
        return ["blaze"]
    if region.layer == Layer.NETHER:
        return ["piglin"]
    # Overworld: hostile mobs only at night (riskier — wired into combat in E4).
    if tod == TimeOfDay.NIGHT:
        return ["zombie"]
    return []


def serialize_observation(
    world: "World",
    agent_id: str,
    *,
    round_idx: int,
    day_length: int,
    assignment: str = "",
    frontier: Milestone = Milestone.START,
    recent_messages: list[Message] | None = None,
    mobs: list[str] | None = None,
    landmarks: list[Landmark] | None = None,
) -> Observation:
    """Build the coordinate-free observation for one agent.

    Reaches the world only via coord-free helpers (``exits_of``,
    ``frontier_dirs_of``, ``teammates_view``). Never reads ``Region.pos``.
    """
    agent = world.agents[agent_id]
    region = world.region_of(agent_id)
    tod = time_of_day(round_idx, day_length)

    self_view = SelfView(
        role=agent.role,
        health=agent.health,
        hunger=agent.hunger,
        inventory=dict(agent.inventory),
        status=agent.status,
        current_biome=region.biome,
        layer=region.layer,
    )

    # mobs / landmarks default to env-derived perception (coord-free); callers may
    # still inject overrides (e.g. tests). Landmarks come from the world's
    # bearing/band helpers — never ``Region.pos`` — so the coord-leak guard holds.
    mob_list = mobs if mobs is not None else perceived_mobs(region, tod)
    landmark_list = (
        landmarks
        if landmarks is not None
        else [
            Landmark(type=ltype, rel_dir=bearing, distance_band=band)
            for (ltype, bearing, band) in world.perceived_landmarks(region.id)
        ]
    )

    here = HereView(
        resources_visible=sorted(region.resources.keys()) if region.discovered else [],
        structure=region.structure if region.discovered else None,
        mobs=list(mob_list),
        exits=[
            Exit(dir=bearing, distance_band=band, biome_hint=hint)
            for (bearing, band, hint) in world.exits_of(region.id)
        ],
        frontier_dirs=world.frontier_dirs_of(region.id),
    )

    teammates = [
        TeammateView(agent=other_id, distance_band=band, bearing=bearing, role=role)
        for (other_id, band, bearing, role) in world.teammates_view(agent_id)
    ]

    return Observation(
        round=round_idx,
        time_of_day=tod,
        self=self_view,
        here=here,
        teammates=teammates,
        known_landmarks=list(landmark_list),
        recent_messages=list(recent_messages or []),
        assignment=assignment,
        dag_frontier_reached=frontier.value,
    )


__all__ = ["serialize_observation", "time_of_day", "perceived_mobs"]

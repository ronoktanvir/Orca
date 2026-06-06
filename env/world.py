"""Graph-on-a-hidden-plane world model (§3.1).

Each :class:`Region` has a HIDDEN ``pos`` used **only by the env** to compute
bearings and distance bands. Agents never receive ``pos`` (§3.1). All public
:class:`World` perception methods return coordinate-free primitives (bearings,
distance bands, biome hints) — the geometry never escapes this module.

The coordinate-leak invariant has three layers:
  1. the contracts forbid extra fields (no ``pos`` can enter an Observation),
  2. ``observation.serialize_observation`` never references ``.pos`` and only
     calls the coord-free helpers below,
  3. ``obs_guard/coord_leak_test.py`` scans serialized output for leaks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from contracts.enums import Bearing, Biome, DistanceBand, Layer, Role, Structure

# Geometry thresholds (env-internal; never emitted).
_BAND_ADJACENT = 1.5
_BAND_NEAR = 2.6
_EDGE_MAX = 3.6  # two regions are graph-adjacent if within this distance
_MOVE_ALIGN_MAX_DEG = 67.5  # a move heading must align to within this angle

_BEARING_ANGLES: dict[Bearing, float] = {
    Bearing.E: 0.0,
    Bearing.NE: 45.0,
    Bearing.N: 90.0,
    Bearing.NW: 135.0,
    Bearing.W: 180.0,
    Bearing.SW: 225.0,
    Bearing.S: 270.0,
    Bearing.SE: 315.0,
}


@dataclass
class Region:
    """A world node. ``pos`` is HIDDEN — env-only, never emitted (§3.1)."""

    id: str  # internal only, e.g. "r_07" — never shown as a landmark
    biome: Biome
    pos: tuple[float, float]  # HIDDEN. env-only. never serialized.
    resources: dict[str, float] = field(default_factory=dict)  # name -> abundance
    structure: Optional[Structure] = None
    layer: Layer = Layer.OVERWORLD
    discovered: bool = False


@dataclass
class AgentState:
    """Per-agent mutable state. Inventory is per-agent — no global stash (§3.5)."""

    agent_id: str
    role: Role
    region_id: str
    inventory: dict[str, int] = field(default_factory=dict)
    health: float = 1.0
    hunger: float = 1.0
    status: str = "free"  # "free" | "busy(action,rounds_left)"
    busy_rounds: int = 0
    alive: bool = True
    deaths: int = 0


# --------------------------------------------------------------------------- #
# Geometry helpers — internal only; all return coord-free results.
# --------------------------------------------------------------------------- #
def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _band(dist: float) -> DistanceBand:
    if dist <= _BAND_ADJACENT:
        return DistanceBand.ADJACENT
    if dist <= _BAND_NEAR:
        return DistanceBand.NEAR
    return DistanceBand.FAR


def _bearing(src: tuple[float, float], dst: tuple[float, float]) -> Bearing:
    """8-way compass heading from src to dst (0°=E, 90°=N), nearest octant."""
    deg = math.degrees(math.atan2(dst[1] - src[1], dst[0] - src[0])) % 360.0
    best: Bearing = Bearing.E
    best_diff = 360.0
    for bearing, angle in _BEARING_ANGLES.items():
        diff = abs((deg - angle + 180.0) % 360.0 - 180.0)
        if diff < best_diff:
            best_diff = diff
            best = bearing
    return best


def _angular_diff(deg: float, bearing: Bearing) -> float:
    return abs((deg - _BEARING_ANGLES[bearing] + 180.0) % 360.0 - 180.0)


class World:
    """Holds all regions + agents; exposes only coordinate-free perception."""

    def __init__(self, regions: dict[str, Region], start_region_id: str) -> None:
        self.regions = regions
        self.start_region_id = start_region_id
        self.agents: dict[str, AgentState] = {}

    # -- agent management --------------------------------------------------- #
    def add_agent(self, agent: AgentState) -> None:
        self.agents[agent.agent_id] = agent

    def region_of(self, agent_id: str) -> Region:
        return self.regions[self.agents[agent_id].region_id]

    # -- coord-free perception (the only data that may reach an Observation) - #
    def neighbors(self, region_id: str) -> list[tuple[str, float]]:
        """(region_id, distance) for graph-adjacent regions. Internal helper."""
        here = self.regions[region_id]
        out = []
        for rid, region in self.regions.items():
            if rid == region_id or region.layer != here.layer:
                continue
            dist = _distance(here.pos, region.pos)
            if dist <= _EDGE_MAX:
                out.append((rid, dist))
        out.sort(key=lambda t: t[1])
        return out

    def exits_of(self, region_id: str) -> list[tuple[Bearing, DistanceBand, Biome]]:
        """Discovered/adjacent exits as (dir, distance_band, biome_hint). Coord-free."""
        here = self.regions[region_id]
        exits = []
        for rid, dist in self.neighbors(region_id):
            region = self.regions[rid]
            bearing = _bearing(here.pos, region.pos)
            hint = region.biome if region.discovered else Biome.UNKNOWN
            exits.append((bearing, _band(dist), hint))
        return exits

    def frontier_dirs_of(self, region_id: str) -> list[Bearing]:
        """Headings toward undiscovered reachable regions. Coord-free."""
        here = self.regions[region_id]
        dirs: list[Bearing] = []
        for rid, _dist in self.neighbors(region_id):
            region = self.regions[rid]
            if not region.discovered:
                bearing = _bearing(here.pos, region.pos)
                if bearing not in dirs:
                    dirs.append(bearing)
        return dirs

    def teammates_view(
        self, agent_id: str
    ) -> list[tuple[str, DistanceBand, Optional[Bearing], Role]]:
        """Relative views of other agents — never coordinates (§3.2)."""
        me = self.agents[agent_id]
        my_region = self.regions[me.region_id]
        out = []
        for other_id, other in self.agents.items():
            if other_id == agent_id or not other.alive:
                continue
            if other.region_id == me.region_id:
                out.append((other_id, DistanceBand.SAME_REGION, None, other.role))
            else:
                other_region = self.regions[other.region_id]
                dist = _distance(my_region.pos, other_region.pos)
                bearing = _bearing(my_region.pos, other_region.pos)
                out.append((other_id, _band(dist), bearing, other.role))
        return out

    def resolve_move(self, region_id: str, direction: Bearing) -> Optional[str]:
        """Region best aligned to ``direction`` and reachable, else None (§3.1).

        Picks the reachable region whose bearing is closest to the requested
        heading (within ``_MOVE_ALIGN_MAX_DEG``), tie-broken by nearer distance.
        Repeatedly moving the same heading walks an axis — a transferable,
        coordinate-free strategy.
        """
        here = self.regions[region_id]
        best_rid: Optional[str] = None
        best_key: tuple[float, float] = (1e9, 1e9)
        for rid, dist in self.neighbors(region_id):
            region = self.regions[rid]
            deg = math.degrees(math.atan2(region.pos[1] - here.pos[1], region.pos[0] - here.pos[0])) % 360.0
            diff = _angular_diff(deg, direction)
            if diff > _MOVE_ALIGN_MAX_DEG:
                continue
            key = (diff, dist)
            if key < best_key:
                best_key = key
                best_rid = rid
        return best_rid

    def pooled_inventory(self) -> dict[str, int]:
        """Team-pooled inventory across all agents (for team-frontier detection)."""
        pooled: dict[str, int] = {}
        for agent in self.agents.values():
            for name, qty in agent.inventory.items():
                pooled[name] = pooled.get(name, 0) + qty
        return pooled


__all__ = ["Region", "AgentState", "World"]

"""Contract 1/7 — ``Observation``: exactly what a worker sees each turn (§3.2).

Egocentric + relative only. **No coordinates, no region count, no global map,
no internal region ids.** Every sub-model sets ``extra="forbid"`` so a stray
``pos`` (or any coordinate-shaped field) cannot be smuggled into an observation
at construction time — this is the first line of the coordinate-leak invariant
(§3.2). The second line is ``env/observation.serialize_observation`` being the
only obs path; the third is ``obs_guard/coord_leak_test.py`` scanning output.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from .enums import Bearing, Biome, DistanceBand, Layer, Role, Structure, TimeOfDay
from .message import Message

_STRICT = ConfigDict(extra="forbid")


class Exit(BaseModel):
    """A discovered/adjacent direction out of the current region (§3.2)."""

    model_config = _STRICT

    dir: Bearing
    distance_band: DistanceBand
    biome_hint: Biome = Biome.UNKNOWN


class SelfView(BaseModel):
    """The agent's own perceivable state (§3.2)."""

    model_config = _STRICT

    role: Role
    health: float = Field(ge=0.0, le=1.0)
    hunger: float = Field(ge=0.0, le=1.0)
    inventory: dict[str, int] = Field(default_factory=dict)
    status: str = "free"  # "free" | "busy(action,rounds_left)"
    current_biome: Biome
    layer: Layer = Layer.OVERWORLD


class HereView(BaseModel):
    """What is perceivable in the current region (§3.2)."""

    model_config = _STRICT

    resources_visible: list[str] = Field(default_factory=list)
    structure: Optional[Structure] = None
    mobs: list[str] = Field(default_factory=list)
    exits: list[Exit] = Field(default_factory=list)
    frontier_dirs: list[Bearing] = Field(default_factory=list)


class TeammateView(BaseModel):
    """A teammate, relative only — never a coordinate (§3.2)."""

    model_config = _STRICT

    agent: str
    distance_band: DistanceBand
    bearing: Optional[Bearing] = None  # null when SAME_REGION
    role: Role


class Landmark(BaseModel):
    """An abstract, transferable landmark — never coordinates (§3.2)."""

    model_config = _STRICT

    type: str
    rel_dir: Bearing
    distance_band: DistanceBand


class Observation(BaseModel):
    """The full coordinate-free observation handed to a worker each turn."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    round: int = Field(ge=0)
    time_of_day: TimeOfDay
    # JSON key is "self" (§3.2); the Python attribute is ``self_view`` to avoid
    # shadowing the conventional instance argument. Serialize with by_alias=True.
    self_view: SelfView = Field(alias="self")
    here: HereView
    teammates: list[TeammateView] = Field(default_factory=list)
    known_landmarks: list[Landmark] = Field(default_factory=list)
    recent_messages: list[Message] = Field(default_factory=list)
    assignment: str = ""
    dag_frontier_reached: str = "start"  # team progress so far (shared signal)


__all__ = [
    "Observation",
    "Exit",
    "SelfView",
    "HereView",
    "TeammateView",
    "Landmark",
]

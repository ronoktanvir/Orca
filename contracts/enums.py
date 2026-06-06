"""Shared enumerations used across the seven frozen Orca contracts.

These are part of the frozen interface (ORCA_master_build_spec.md §11). Adding a
*new* member is an additive change (broadcast to the team, per workflow §8); never
remove or renumber existing members after the fork.

Design note — item/resource identifiers are plain ``str`` (not enums) on purpose.
The tech tree deepens through the streams (E1), and forcing every craftable item
into an enum would make every recipe addition a contract change. The canonical
identifier sets live in ``env/techtree.py``; the contracts stay stable.
"""

from __future__ import annotations

from enum import Enum


class Layer(str, Enum):
    """Which sub-world a region/agent is in (§3.1)."""

    OVERWORLD = "overworld"
    NETHER = "nether"
    END = "end"


class TimeOfDay(str, Enum):
    """Derived from ``round % day_length`` (§3.2)."""

    DAY = "day"
    DUSK = "dusk"
    NIGHT = "night"
    DAWN = "dawn"


class DistanceBand(str, Enum):
    """Coarse, coordinate-free distance (§3.1). ``SAME_REGION`` is used for
    teammates / co-location; exits use ADJACENT/NEAR/FAR."""

    SAME_REGION = "SAME_REGION"
    ADJACENT = "ADJACENT"
    NEAR = "NEAR"
    FAR = "FAR"


class Bearing(str, Enum):
    """8-way compass heading (§3.1). Never a number."""

    N = "N"
    NE = "NE"
    E = "E"
    SE = "SE"
    S = "S"
    SW = "SW"
    W = "W"
    NW = "NW"


class Biome(str, Enum):
    """Biome *types* are shared across all seeds (§3.4). ``UNKNOWN`` is what an
    undiscovered exit hint resolves to."""

    FOREST = "forest"
    JUNGLE = "jungle"
    TAIGA = "taiga"
    PLAINS = "plains"
    MOUNTAINS = "mountains"
    CAVES = "caves"
    DESERT = "desert"
    SWAMP = "swamp"
    OCEAN = "ocean"
    NETHER_WASTES = "nether_wastes"
    SOUL_SAND_VALLEY = "soul_sand_valley"
    BASALT_DELTA = "basalt_delta"
    WARPED_FOREST = "warped_forest"
    STRONGHOLD = "stronghold"
    END = "end"
    UNKNOWN = "unknown"


class Structure(str, Enum):
    """Special discoverable nodes (§3.1)."""

    FORTRESS = "fortress"
    STRONGHOLD = "stronghold"


class Role(str, Enum):
    """Soft role priors — never hard action masks (§4.1)."""

    EXPLORER = "explorer"
    MINER = "miner"
    TINKERER = "tinkerer"
    SUPPORT = "support"


class ActionName(str, Enum):
    """The macro-action menu (§3.3). The env is the source of truth for whether
    a chosen action is *valid* given world state."""

    MOVE = "move"
    SCOUT = "scout"
    GATHER = "gather"
    CRAFT = "craft"
    SMELT = "smelt"
    PLACE = "place"
    FIGHT = "fight"
    EAT = "eat"
    SLEEP = "sleep"
    GIVE_ITEM = "give_item"
    REQUEST_HELP = "request_help"
    REGROUP = "regroup"
    REPORT = "report"
    WAIT = "wait"


class MessageType(str, Enum):
    """Structured bus message kinds (§5.1) — no free chat."""

    REPORT = "report"
    REQUEST_HELP = "request_help"
    SHARE_FINDING = "share_finding"
    PROPOSE_RENDEZVOUS = "propose_rendezvous"
    ACK = "ack"
    HANDOFF = "handoff"


class Milestone(str, Enum):
    """The DAG progression ladder (§3.4, §7.1), ordered from shallow to deep.

    Phase 0's stub env only reaches up to :attr:`IRON`. The full ladder lives
    here so ``EpisodeMetrics.frontier_milestone`` is forward-compatible with the
    Stream-1 env depth that goes all the way to :attr:`DRAGON_DEFEATED`.
    The milestone -> reward-value mapping lives in ``reward/dag.py``.
    """

    START = "start"
    WOOD = "wood"
    WOODEN_TOOLS = "wooden_tools"
    STONE_TOOLS = "stone_tools"
    STABLE_FOOD = "stable_food"
    SHELTER = "shelter"
    IRON = "iron"
    SHIELD_BUCKET = "shield_bucket"
    OBSIDIAN = "obsidian"
    PORTAL_BUILT = "portal_built"
    NETHER_ENTERED = "nether_entered"
    FORTRESS_FOUND = "fortress_found"
    BLAZE_RODS = "blaze_rods"
    ENDER_PEARLS = "ender_pearls"
    EYES_OF_ENDER = "eyes_of_ender"
    STRONGHOLD_FOUND = "stronghold_found"
    END_PORTAL_ACTIVE = "end_portal_active"
    END_ENTERED = "end_entered"
    DRAGON_DEFEATED = "dragon_defeated"


__all__ = [
    "Layer",
    "TimeOfDay",
    "DistanceBand",
    "Bearing",
    "Biome",
    "Structure",
    "Role",
    "ActionName",
    "MessageType",
    "Milestone",
]

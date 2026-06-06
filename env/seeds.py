"""Stub seed layouts (§3.7).

A *seed* is layout only — biome graph, hidden positions, resource abundances —
while rules/recipes/probabilities are identical across seeds. Phase 0 ships one
hand-placed 5-region layout that the scripted oracle can solve to IRON. The seed
name applies a small deterministic abundance jitter so the {A,T2,T3,B,C} pool
*exists* nominally (§3.7 / D7), but only seed ``A`` is guaranteed oracle-solvable
in Phase 0. Stream 1 (E3) replaces this with a real generator + full-DAG oracle.
"""

from __future__ import annotations

from contracts.enums import Biome, Layer

from .rng import make_rng
from .world import Region, World

# Train pool {A, T2, T3} + held-out {B, C} (§3.7, D7). Held-out are never trained on.
TRAIN_SEEDS = ("A", "T2", "T3")
HELDOUT_SEEDS = ("B", "C")
ALL_SEEDS = TRAIN_SEEDS + HELDOUT_SEEDS

# Hand-placed 5-region layout. Positions are HIDDEN (env-only). Chosen so that:
#   r_00 (forest, start) --move N--> r_02 (mountains) is the oracle's path to iron.
_LAYOUT: list[dict] = [
    {"id": "r_00", "biome": Biome.FOREST, "pos": (0.0, 0.0),
     "resources": {"wood": 0.9, "food": 0.5}},
    {"id": "r_01", "biome": Biome.PLAINS, "pos": (1.2, -0.3),
     "resources": {"food": 0.8, "wood": 0.3}},
    {"id": "r_02", "biome": Biome.MOUNTAINS, "pos": (0.2, 2.0),
     "resources": {"cobblestone": 0.9, "coal": 0.7, "iron_ore": 0.6}},
    {"id": "r_03", "biome": Biome.CAVES, "pos": (0.4, 3.2),
     "resources": {"cobblestone": 0.8, "coal": 0.6, "iron_ore": 0.7, "diamond": 0.3}},
    {"id": "r_04", "biome": Biome.DESERT, "pos": (2.4, 0.1),
     "resources": {"sand": 0.9}},
]

_START_REGION = "r_00"


def make_world(seed: str = "A") -> World:
    """Build the stub world for ``seed``. Deterministic; layout is fixed, abundances
    get a tiny per-seed jitter so distinct seeds are distinguishable."""
    rng = make_rng(seed, episode_idx=0, round_idx=0, agent_id="__worldgen__")
    regions: dict[str, Region] = {}
    for spec in _LAYOUT:
        jittered = {
            res: max(0.05, min(1.0, ab * (0.9 + 0.2 * rng.random())))
            for res, ab in spec["resources"].items()
        }
        regions[spec["id"]] = Region(
            id=spec["id"],
            biome=spec["biome"],
            pos=spec["pos"],
            resources=jittered,
            structure=None,
            layer=Layer.OVERWORLD,
            discovered=(spec["id"] == _START_REGION),  # start region known
        )
    return World(regions=regions, start_region_id=_START_REGION)


__all__ = ["make_world", "TRAIN_SEEDS", "HELDOUT_SEEDS", "ALL_SEEDS"]

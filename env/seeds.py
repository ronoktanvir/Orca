"""Seed layouts (§3.7) — multi-layer region graph on a hidden plane.

A *seed* is layout only — biome graph, hidden positions, resource abundances,
structure/portal placement — while rules/recipes/probabilities are identical
across seeds (§3.7). Stream 1 E2 grows the Phase-0 5-node stub into a full
multi-layer world: an Overworld grid + a separate Nether sub-graph + a single
End node, with a fixed-per-seed Nether entry and a stronghold that hosts the End
portal. Cross-layer travel is via explicit portal transitions (``env/actions.py``)
— ``World.neighbors`` never crosses layers.

Determinism: everything derives from ``make_rng(seed, …)`` (sha256-seeded,
process-stable), so a seed reproduces its world byte-for-byte across processes;
distinct seeds get distinct biome arrangements, jitter and abundances. Seed ``A``
is kept oracle-friendly (the FOREST start has a MOUNTAINS region exactly to its
N); ``B``/``C`` merely need to exist as distinct layouts — full reachability
validation is E3, not E2.
"""

from __future__ import annotations

from contracts.enums import Biome, Layer, Structure

from . import techtree
from .rng import make_rng
from .world import Region, World

# Train pool {A, T2, T3} + held-out {B, C} (§3.7, D7). Held-out are never trained on.
TRAIN_SEEDS = ("A", "T2", "T3")
HELDOUT_SEEDS = ("B", "C")
ALL_SEEDS = TRAIN_SEEDS + HELDOUT_SEEDS

# --------------------------------------------------------------------------- #
# Graph dimensions (within §3.7 ranges: 20–40 Overworld + 8–15 Nether + 1 End).
# --------------------------------------------------------------------------- #
_OW_COLS, _OW_ROWS = 6, 4  # 24 Overworld regions
_NE_COLS, _NE_ROWS = 5, 2  # 10 Nether regions
_SPACING = 2.0  # grid step (< _EDGE_MAX=3.6 so orthogonal+diagonal cells are edges)
_JITTER = 0.3  # per-axis position jitter for interior cells (keeps bands varied)
_OW_ORIGIN = (0.0, 0.0)
_NE_ORIGIN = (100.0, 100.0)  # far cluster; the layer filter isolates it regardless

_START_REGION = "r_00"  # FOREST, exact (0,0) — oracle starts here

# Forced "spine" so seed A stays oracle-winnable (start -> N mountains -> caves)
# and the Phase-0 geometry tests still hold. Cell index = row * _OW_COLS + col.
_OW_START_IDX = 0  # (col0,row0) FOREST start
_OW_MOUNTAINS_IDX = _OW_COLS  # (col0,row1) MOUNTAINS — exactly N of the start
_OW_CAVES_IDX = _OW_COLS + 1  # (col1,row1) CAVES — adjacent to mountains
_OW_STRONGHOLD_IDX = _OW_COLS * _OW_ROWS - 1  # far corner — hosts the End portal

# Biomes for the 20 non-spine, non-stronghold Overworld cells (permuted per seed).
_OW_FILL = [
    Biome.PLAINS, Biome.PLAINS, Biome.PLAINS, Biome.DESERT, Biome.DESERT,
    Biome.FOREST, Biome.FOREST, Biome.JUNGLE, Biome.JUNGLE, Biome.TAIGA,
    Biome.TAIGA, Biome.SWAMP, Biome.SWAMP, Biome.OCEAN, Biome.OCEAN,
    Biome.MOUNTAINS, Biome.MOUNTAINS, Biome.CAVES, Biome.CAVES, Biome.DESERT,
]

_NE_ENTRY_IDX = 0  # (col0,row0) — fixed per-seed Nether entry
_NE_FORTRESS_IDX = _NE_COLS  # (col0,row1) — hosts the FORTRESS
# Biomes for the 8 non-entry, non-fortress Nether cells (permuted per seed).
_NE_FILL = [
    Biome.SOUL_SAND_VALLEY, Biome.SOUL_SAND_VALLEY, Biome.BASALT_DELTA,
    Biome.BASALT_DELTA, Biome.WARPED_FOREST, Biome.WARPED_FOREST,
    Biome.NETHER_WASTES, Biome.BASALT_DELTA,
]

# Hand-tuned spine abundances (override the random ones) so the oracle always has
# wood at the start and cobblestone/coal/iron in the mountains.
_SPINE_RESOURCES: dict[int, dict[str, float]] = {
    _OW_START_IDX: {"wood": 0.9, "food": 0.5},
    _OW_MOUNTAINS_IDX: {"cobblestone": 0.9, "coal": 0.7, "iron_ore": 0.6},
    _OW_CAVES_IDX: {
        "cobblestone": 0.8, "coal": 0.6, "iron_ore": 0.7,
        "diamond": 0.4, "lava_pool": 0.5, "flint": 0.4,
    },
}


def _abundance(rng) -> float:
    return round(max(0.1, min(0.95, 0.45 + 0.4 * rng.random())), 3)


def _resources_for(biome: Biome, rng) -> dict[str, float]:
    """Jittered abundances for a biome's resource *types* (§3.4). Sorted for
    deterministic rng consumption."""
    return {name: _abundance(rng) for name in sorted(techtree.BIOME_RESOURCES.get(biome, set()))}


def _pos(col: int, row: int, origin: tuple[float, float], rng, frame: bool) -> tuple[float, float]:
    """Hidden grid position. Frame cells (row 0 / col 0) are exact — they anchor
    the start's geometry (no region to the S; mountains exactly N). Interior cells
    jitter so bearings/bands vary per seed."""
    if frame:
        return (origin[0] + col * _SPACING, origin[1] + row * _SPACING)
    jx = (rng.random() - 0.5) * 2 * _JITTER
    jy = (rng.random() - 0.5) * 2 * _JITTER
    return (origin[0] + col * _SPACING + jx, origin[1] + row * _SPACING + jy)


def _build_layer(
    *,
    n_cols: int,
    n_rows: int,
    origin: tuple[float, float],
    layer: Layer,
    id_offset: int,
    biomes: dict[int, Biome],
    structures: dict[int, Structure],
    rng,
    discovered_idx: int | None,
) -> dict[str, Region]:
    """Build one grid layer of regions with deterministic pos + resources."""
    regions: dict[str, Region] = {}
    for idx in range(n_cols * n_rows):
        col, row = idx % n_cols, idx // n_cols
        rid = f"r_{id_offset + idx:02d}"
        biome = biomes[idx]
        frame = row == 0 or col == 0
        pos = _pos(col, row, origin, rng, frame)
        if id_offset == 0 and idx in _SPINE_RESOURCES:
            resources = dict(_SPINE_RESOURCES[idx])
        else:
            resources = _resources_for(biome, rng)
        regions[rid] = Region(
            id=rid,
            biome=biome,
            pos=pos,
            resources=resources,
            structure=structures.get(idx),
            layer=layer,
            discovered=(idx == discovered_idx),
        )
    return regions


def _overworld_biomes(rng) -> dict[int, Biome]:
    fill = list(_OW_FILL)
    rng.shuffle(fill)  # distinct arrangement per seed; deterministic per seed
    biomes: dict[int, Biome] = {
        _OW_START_IDX: Biome.FOREST,
        _OW_MOUNTAINS_IDX: Biome.MOUNTAINS,
        _OW_CAVES_IDX: Biome.CAVES,
        _OW_STRONGHOLD_IDX: Biome.STRONGHOLD,
    }
    for idx in range(_OW_COLS * _OW_ROWS):
        if idx not in biomes:
            biomes[idx] = fill.pop()
    return biomes


def _nether_biomes(rng) -> dict[int, Biome]:
    fill = list(_NE_FILL)
    rng.shuffle(fill)
    biomes: dict[int, Biome] = {
        _NE_ENTRY_IDX: Biome.NETHER_WASTES,
        _NE_FORTRESS_IDX: Biome.NETHER_WASTES,
    }
    for idx in range(_NE_COLS * _NE_ROWS):
        if idx not in biomes:
            biomes[idx] = fill.pop()
    return biomes


def make_world(seed: str = "A") -> World:
    """Build the full multi-layer world for ``seed`` (§3.1/§3.7). Deterministic."""
    rng = make_rng(seed, episode_idx=0, round_idx=0, agent_id="__worldgen__")

    # --- Overworld ---------------------------------------------------------- #
    ow = _build_layer(
        n_cols=_OW_COLS, n_rows=_OW_ROWS, origin=_OW_ORIGIN, layer=Layer.OVERWORLD,
        id_offset=0, biomes=_overworld_biomes(rng),
        structures={_OW_STRONGHOLD_IDX: Structure.STRONGHOLD},
        rng=rng, discovered_idx=_OW_START_IDX,  # only the start is known at reset
    )
    stronghold_id = f"r_{_OW_STRONGHOLD_IDX:02d}"

    # --- Nether (separate sub-graph) --------------------------------------- #
    ne_offset = _OW_COLS * _OW_ROWS  # 24
    ne = _build_layer(
        n_cols=_NE_COLS, n_rows=_NE_ROWS, origin=_NE_ORIGIN, layer=Layer.NETHER,
        id_offset=ne_offset, biomes=_nether_biomes(rng),
        structures={_NE_FORTRESS_IDX: Structure.FORTRESS},
        rng=rng, discovered_idx=None,
    )
    nether_entry_id = f"r_{ne_offset + _NE_ENTRY_IDX:02d}"

    # --- End (single node) -------------------------------------------------- #
    end_idx = ne_offset + _NE_COLS * _NE_ROWS  # 34
    end_id = f"r_{end_idx:02d}"
    end = {
        end_id: Region(
            id=end_id, biome=Biome.END, pos=(200.0, 200.0),
            resources={}, structure=None, layer=Layer.END, discovered=False,
        )
    }

    regions = {**ow, **ne, **end}
    return World(
        regions=regions,
        start_region_id=_START_REGION,
        nether_entry_id=nether_entry_id,
        end_region_id=end_id,
        stronghold_id=stronghold_id,
    )


__all__ = ["make_world", "TRAIN_SEEDS", "HELDOUT_SEEDS", "ALL_SEEDS"]

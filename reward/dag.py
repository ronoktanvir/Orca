"""The DAG frontier ladder (§7.1).

``team_reward`` is **max-frontier, not cumulative** (§7.1): the base reward is the
value of the *deepest* milestone reached, so agents can't farm easy subtasks. The
anchor values below are exactly §7.1; the finer stub-only milestones
(wooden/stone tools, shelter, etc.) get interpolated values so the curve is
monotonic non-decreasing in milestone depth.
"""

from __future__ import annotations

from contracts.enums import Milestone

# Milestone -> base frontier value. Monotonic non-decreasing in depth.
# §7.1 anchors marked; the rest are interpolated stub/finer milestones.
MILESTONE_VALUE: dict[Milestone, float] = {
    Milestone.START: 0.00,
    Milestone.WOOD: 0.05,  # §7.1 wood / basic tools
    Milestone.WOODEN_TOOLS: 0.07,
    Milestone.STONE_TOOLS: 0.10,
    Milestone.STABLE_FOOD: 0.12,  # §7.1 stable food + shelter/bed
    Milestone.SHELTER: 0.15,
    Milestone.IRON: 0.20,  # §7.1 iron tooling
    Milestone.SHIELD_BUCKET: 0.24,
    Milestone.OBSIDIAN: 0.27,
    Milestone.PORTAL_BUILT: 0.30,  # §7.1 nether portal built
    Milestone.NETHER_ENTERED: 0.40,  # §7.1 Nether entered
    Milestone.FORTRESS_FOUND: 0.55,  # §7.1 fortress found
    Milestone.BLAZE_RODS: 0.65,  # §7.1 blaze rods acquired
    Milestone.ENDER_PEARLS: 0.72,
    Milestone.EYES_OF_ENDER: 0.75,  # §7.1 ender pearls + eyes of ender
    Milestone.STRONGHOLD_FOUND: 0.80,  # §7.1 stronghold found
    Milestone.END_PORTAL_ACTIVE: 0.83,
    Milestone.END_ENTERED: 0.85,  # §7.1 End entered
    Milestone.DRAGON_DEFEATED: 1.00,  # §7.1 dragon defeated (WIN)
}

# Ordered ladder, shallow -> deep.
FRONTIER_LADDER: list[Milestone] = list(Milestone)


def frontier_value(milestone: Milestone) -> float:
    """Base reward for the deepest milestone reached (§7.1)."""
    return MILESTONE_VALUE[milestone]


def is_win(milestone: Milestone) -> bool:
    return milestone == Milestone.DRAGON_DEFEATED


__all__ = ["MILESTONE_VALUE", "FRONTIER_LADDER", "frontier_value", "is_win"]

"""Delegation situations & arms (§6.3) — Stream 3 (O2).

The delegation bandit acts **once per episode** over a tiny, fixed set of
recurring strategic forks (the "situations"), each with 2–4 discrete options
(the "arms"). This is a contextual bandit by construction — no PPO, no per-step
rewards (§6.3). Keeping the space tiny is what lets it learn within dozens of
episodes.

This module is pure config + deterministic mapping:
  * ``SITUATION_ARMS``      — situation -> list of arm names (fed to the bandit).
  * ``roster_for_arm``      — S1 arm -> the per-agent role roster (the *who*).
  * ``strategic_directives``— S2/S3/S4 arms -> readable directive strings folded
                              into every behavior-card (the *how*, team-level).

The deep situations (S2 nether / S3 fortress / S4 end) only bite once the env
reaches that depth (Stream 1); in the shallow iron slice they are still chosen
and still credited with the episode's ``team_reward`` each episode, so the bandit
accumulates honest (near-flat) statistics there rather than fabricated ones.
"""

from __future__ import annotations

from contracts.enums import Role

# --------------------------------------------------------------------------- #
# The situation / arm menu (§6.3). Order within each list is stable.
# --------------------------------------------------------------------------- #
S1 = "S1_role_assignment"  # early-game: who is explorer/miner/tinkerer/support
S2 = "S2_nether_entry"  # enter when geared vs immediately
S3 = "S3_fortress_search"  # solo / two-pairs / all-together
S4 = "S4_end_approach"  # regroup-all vs split

SITUATION_ARMS: dict[str, list[str]] = {
    S1: ["balanced", "mining_heavy", "explore_heavy", "self_sufficient"],
    S2: ["gear_gated", "iron_secured", "immediate"],
    S3: ["solo", "two_pairs", "all_together"],
    S4: ["regroup_all", "split_roles"],
}

# S1 arm -> ordered role vector (mapped onto the agent ids in roster order).
_S1_ROSTERS: dict[str, list[Role]] = {
    "balanced": [Role.EXPLORER, Role.MINER, Role.TINKERER, Role.SUPPORT],
    "mining_heavy": [Role.MINER, Role.MINER, Role.TINKERER, Role.SUPPORT],
    "explore_heavy": [Role.EXPLORER, Role.EXPLORER, Role.MINER, Role.SUPPORT],
    "self_sufficient": [Role.EXPLORER, Role.MINER, Role.MINER, Role.TINKERER],
}

# S2/S3/S4 arm -> the strategic directive it injects into the team's cards.
_DIRECTIVES: dict[str, dict[str, str]] = {
    S2: {
        "gear_gated": "Enter the Nether only with a full iron kit (sword, shield, bucket).",
        "iron_secured": "Enter the Nether once iron tools are secured.",
        "immediate": "Enter the Nether as soon as the portal is built; gear up inside.",
    },
    S3: {
        "solo": "Search for the fortress solo, fanning out to cover ground fast.",
        "two_pairs": "Search for the fortress in two pairs for safety.",
        "all_together": "Search for the fortress as one group.",
    },
    S4: {
        "regroup_all": "Regroup the full team before activating the End portal.",
        "split_roles": "Split into fighters and support for the End approach.",
    },
}


def roster_for_arm(agent_ids: list[str], s1_arm: str) -> list[tuple[str, Role]]:
    """Map the S1 arm onto ``agent_ids`` (in order) -> the episode roster (§4.1).

    Robust to team sizes other than 4: the role vector is cycled so a 1- or
    2-agent roster (e.g. the offline oracle smoke) still gets valid roles.
    """
    roles = _S1_ROSTERS.get(s1_arm, _S1_ROSTERS["balanced"])
    return [(aid, roles[i % len(roles)]) for i, aid in enumerate(agent_ids)]


def strategic_directives(arms: dict[str, str]) -> list[str]:
    """Readable directive strings for the chosen S2/S3/S4 arms (team-level)."""
    out: list[str] = []
    for sit in (S2, S3, S4):
        arm = arms.get(sit)
        if arm and arm in _DIRECTIVES[sit]:
            out.append(_DIRECTIVES[sit][arm])
    return out


def default_arms() -> dict[str, str]:
    """The first (index-0) arm of each situation — a sensible neutral default."""
    return {sit: arm_list[0] for sit, arm_list in SITUATION_ARMS.items()}


__all__ = [
    "SITUATION_ARMS",
    "S1",
    "S2",
    "S3",
    "S4",
    "roster_for_arm",
    "strategic_directives",
    "default_arms",
]

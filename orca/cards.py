"""Default behavior-cards + roster (§4.1, §6.6).

Phase 0 freezes worker behavior-cards to sensible defaults (§6.6 Phase 0): Orca
only learns *delegation* (the bandit) while cards stay fixed. Stream 3 (O4) makes
the coach edit these between episodes. The default roster is the 4 role-biased
workers (§4.1); Phase 0's run uses a single miner oracle by default but the
roster helper supports the full team.
"""

from __future__ import annotations

from contracts import BehaviorCard
from contracts.enums import Role

# Default soft-role roster (§4.1).
DEFAULT_ROSTER: list[tuple[str, Role]] = [
    ("agent_1", Role.EXPLORER),
    ("agent_2", Role.MINER),
    ("agent_3", Role.TINKERER),
    ("agent_4", Role.SUPPORT),
]

_ROLE_ASSIGNMENT: dict[Role, str] = {
    Role.EXPLORER: "Scout outward along a heading; reveal regions, biomes and structures.",
    Role.MINER: "Gather wood, craft tools, then mine cobblestone and iron ore.",
    Role.TINKERER: "Craft and smelt: tools, gear, and portal materials.",
    Role.SUPPORT: "Manage food and combat; escort and revive teammates.",
}


def make_default_card(agent_id: str, role: Role) -> BehaviorCard:
    """A frozen, sensible default behavior-card for one agent (§6.6)."""
    return BehaviorCard(
        agent_id=agent_id,
        role=role,
        assignment=_ROLE_ASSIGNMENT[role],
        directives=["Prefer valid actions; the env rejects illegal ones."],
        priorities=["progress the team DAG frontier"],
        donts=["don't act without the required tool"],
        version=0,
    )


def default_cards(roster: list[tuple[str, Role]]) -> dict[str, BehaviorCard]:
    return {aid: make_default_card(aid, role) for aid, role in roster}


__all__ = ["DEFAULT_ROSTER", "make_default_card", "default_cards"]

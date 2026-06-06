"""Contract 4/7 — ``BehaviorCard``: Orca-owned, per-agent (§4.3, §6.2).

A behavior-card holds WHO does what + coaching: the current role assignment,
directives, priorities, and do/don'ts. Orca authors and edits these between
episodes (frozen in Phase 0, §6.6). ``version`` increments on each edit so the
card diff is auditable in Weave (§10).

Distinction to keep crisp (§4.5): behavior-card = WHO + coaching (Orca owns);
execution-memory = HOW to do tasks (the agent owns).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .enums import Role


class BehaviorCard(BaseModel):
    """The persistent, Orca-authored card read by a worker at episode start."""

    agent_id: str
    role: Role
    assignment: str = ""  # e.g. "Mine iron until you have 6 ingots, then regroup."
    directives: list[str] = Field(default_factory=list)
    priorities: list[str] = Field(default_factory=list)
    donts: list[str] = Field(default_factory=list)
    version: int = Field(default=0, ge=0)  # bumps on each Orca edit (for diffing)


__all__ = ["BehaviorCard"]

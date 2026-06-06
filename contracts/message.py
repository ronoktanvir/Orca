"""Contract 3/7 — ``Message``: a structured bus message (§5.1).

No free chat: every message is typed and carries an urgency for attention
prioritization. Content length is intentionally *not* capped (spec §5.1).
Delivered turn-based at round t+1 (§5.2) — that scheduling lives in ``bus/``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .enums import MessageType


class Message(BaseModel):
    """A single structured message on the comm bus."""

    model_config = ConfigDict(populate_by_name=True)

    # ``from`` is a Python keyword, so the field is ``from_agent`` with the
    # JSON alias "from". Construct with either name; serialize with by_alias=True.
    from_agent: str = Field(alias="from", description="sender agent id, or 'orca'")
    to: str = Field(description='"team" | "agent_k" | "orca"')
    type: MessageType
    content: str
    urgency: float = Field(default=0.0, ge=0.0, le=1.0)
    round: int = Field(ge=0)


__all__ = ["Message"]

"""Message helpers (§5.1) — Stream 2 (A4).

The ``Message`` schema itself is frozen in contracts. This module re-exports it
and offers small constructors; Stream 2 deepens it (bandwidth realism, etc.).
"""

from __future__ import annotations

from contracts import Message
from contracts.enums import MessageType


def make_message(
    from_agent: str,
    to: str,
    type: MessageType,
    content: str,
    round: int,
    urgency: float = 0.3,
) -> Message:
    return Message(
        **{"from": from_agent},
        to=to,
        type=type,
        content=content,
        urgency=urgency,
        round=round,
    )


__all__ = ["Message", "MessageType", "make_message"]

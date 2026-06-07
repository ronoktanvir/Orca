"""Message helpers (§5.1) — Stream 2 (A4).

The ``Message`` schema itself is frozen in contracts. This module re-exports it
and offers small constructors plus the recipient-validation helper the worker and
the run loop share so no internal/seed-specific string can ride the ``to`` field
onto the bus or into the trace (§3.2, §5.1).
"""

from __future__ import annotations

import re

from contracts import Message
from contracts.enums import MessageType

# A message recipient is the team broadcast, the manager, or an agent id matching
# the project convention (``agent_1`` / ``agent_2`` / ...). Anything else — a
# leaky region id like ``r_07``, a coordinate-like string, free text — is invalid.
_AGENT_ID = re.compile(r"^agent_\d+$")
_FIXED_RECIPIENTS = ("team", "orca")


def normalize_recipient(to: object) -> str:
    """Coerce a message recipient to a safe, leak-free value (§5.1).

    Valid recipients are exactly ``"team"``, ``"orca"``, or an ``agent_<n>`` id.
    Anything else (an internal region id like ``r_07``, a coordinate-like string,
    or arbitrary free text) is downgraded to ``"team"`` — chosen over *dropping*
    so the message still reaches the team, but the leaky/invalid recipient string
    never reaches ``Message.to``, ``pending_messages``, the bus, or the trace.
    """
    t = (to if isinstance(to, str) else str(to or "")).strip()
    if t in _FIXED_RECIPIENTS or _AGENT_ID.match(t):
        return t
    return "team"


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
        to=normalize_recipient(to),
        type=type,
        content=content,
        urgency=urgency,
        round=round,
    )


__all__ = ["Message", "MessageType", "make_message", "normalize_recipient"]

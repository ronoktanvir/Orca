"""Communication bus with t+1 delivery (§5.2) — Stream 2 (A4).

Messages posted in round t are delivered starting round t+1 (avoids within-round
causality loops, §5.2). Each agent sees the **last K** messages addressed to it
or to ``team`` across delivered history — a sliding window, not just the current
round (Stream 2 brief: "each agent sees the last K addressed to it or team"). The
full verbatim log is kept for Weave + the Orca trace.
"""

from __future__ import annotations

from contracts import Message


def _msg_key(m: Message) -> tuple:
    """Identity of a delivered message (for de-duplication). Includes urgency so
    two same-content reports sent with different urgency aren't collapsed."""
    return (m.from_agent, m.to, m.type.value, m.content, m.round, m.urgency)


class CommBus:
    """Minimal turn-based bus: post now, deliver from next round on (§5.2)."""

    def __init__(self, window: int = 8) -> None:
        self.window = window
        self._posted: list[Message] = []  # this round -> deliverable next round
        self._delivered: list[Message] = []  # cumulative delivered history
        self.log: list[Message] = []  # full verbatim log (for Weave + Orca trace)

    def post(self, message: Message) -> None:
        self._posted.append(message)
        self.log.append(message)

    def tick(self) -> None:
        """Advance one round: last round's posts join the deliverable history."""
        self._delivered.extend(self._posted)
        self._posted = []

    def recent_for(self, agent_id: str) -> list[Message]:
        """The last ``window`` messages addressed to ``agent_id`` or ``team`` (§5.2).

        Spans the whole delivered history (a sliding window), excludes the agent's
        own messages, and de-duplicates so an identical message delivered twice is
        only surfaced once. Message content is returned in full (never truncated).
        """
        relevant = [
            m for m in self._delivered if m.to in ("team", agent_id) and m.from_agent != agent_id
        ]
        deduped: list[Message] = []
        seen: set = set()
        for m in relevant:
            k = _msg_key(m)
            if k in seen:
                continue
            seen.add(k)
            deduped.append(m)
        return deduped[-self.window :]


__all__ = ["CommBus"]

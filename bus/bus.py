"""Communication bus with t+1 delivery (§5.2) — Stream 2 (A4).

Messages posted in round t are delivered in round t+1 (avoids within-round
causality loops, §5.2). Each agent sees the last K messages addressed to it or
to ``team``. Phase 0's env handles message delivery inline; this is the extracted
bus Stream 2 wires in (plus history summarization).
"""

from __future__ import annotations

from contracts import Message


class CommBus:
    """Minimal turn-based bus: post now, deliver next round (§5.2)."""

    def __init__(self, window: int = 8) -> None:
        self.window = window
        self._posted: list[Message] = []  # this round -> delivered next round
        self._delivered: list[Message] = []  # available to read this round
        self.log: list[Message] = []  # full verbatim log (for Weave + Orca trace)

    def post(self, message: Message) -> None:
        self._posted.append(message)
        self.log.append(message)

    def tick(self) -> None:
        """Advance one round: last round's posts become deliverable."""
        self._delivered = self._posted
        self._posted = []

    def recent_for(self, agent_id: str) -> list[Message]:
        relevant = [
            m for m in self._delivered if m.to in ("team", agent_id) and m.from_agent != agent_id
        ]
        return relevant[-self.window :]


__all__ = ["CommBus"]

"""Agent interface (Stream 2 territory).

Every agent — scripted oracle, no-op, or (Stream 2) LLM worker — implements
``act(observation) -> Action``. The run loop only ever talks to agents through
this method, so swapping the scripted placeholder for the real LLM worker is a
drop-in change inside ``agents/`` (workflow §2).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from contracts import Action, Observation


@runtime_checkable
class Agent(Protocol):
    """Minimal agent contract used by the run loop."""

    agent_id: str

    def act(self, obs: Observation) -> Action:  # pragma: no cover - protocol
        ...


__all__ = ["Agent"]

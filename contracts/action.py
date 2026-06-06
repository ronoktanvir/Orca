"""Contract 2/7 — ``Action``: one macro-action chosen by a worker per free turn (§3.3).

The env (not the contract, not the LLM) is the source of truth for *validity*.
``args`` is an open dict so the action menu can deepen (smelt/place/fight/...)
without a contract change; per-action arg requirements are enforced in
``env/actions.py``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .enums import ActionName


class Action(BaseModel):
    """A macro-action. e.g. ``Action(name="gather", args={"resource": "iron_ore"})``."""

    model_config = ConfigDict(extra="forbid")

    name: ActionName
    args: dict[str, Any] = Field(default_factory=dict)


__all__ = ["Action"]

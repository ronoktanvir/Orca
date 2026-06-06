"""Stream 2 territory — the comm bus (`bus/`). §5."""

from __future__ import annotations

from .bus import CommBus
from .messages import make_message

__all__ = ["CommBus", "make_message"]

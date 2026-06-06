"""LLM client interface (§11): ``llm.complete(prompt, schema)``.

One client behind an interface so the model is swappable (hosted <-> local vLLM)
via config (§11). Phase 0 needs no real model (the oracle is scripted), so this
ships the interface + a deterministic stub. Stream 2 implements the hosted
strong-model client (low temperature, full prompt/output logging, §3.6).
"""

from __future__ import annotations

import json
from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """A model behind one method. ``schema`` (a pydantic model class) requests
    schema-constrained JSON output (§4.3 OUTPUT_SCHEMA)."""

    def complete(self, prompt: str, schema: Optional[type] = None, **kwargs: Any) -> str:
        ...


class StubLLM:
    """Deterministic, offline stub. Returns empty JSON; never called in Phase 0."""

    def __init__(self, temperature: float = 0.2) -> None:
        self.temperature = temperature

    def complete(self, prompt: str, schema: Optional[type] = None, **kwargs: Any) -> str:
        return json.dumps({})


__all__ = ["LLMClient", "StubLLM"]

"""LLM worker turn loop (§4.2) — Stream 2 (A1, A2).

The real worker: build the prompt (prompts.build_worker_prompt), call the LLM
through ``llm.complete(prompt, schema)``, parse + validate strict JSON, repair
once or default to ``wait`` on malformed output (§4.4). Phase 0 ships only the
scripted oracle (``agents.scripted.ShallowOracle``); this is the seam Stream 2
implements.
"""

from __future__ import annotations

from contracts import Action, Observation
from contracts.enums import ActionName


class LLMWorker:
    """Placeholder for the Stream 2 LLM-backed worker."""

    def __init__(self, agent_id: str, llm=None) -> None:
        self.agent_id = agent_id
        self.llm = llm

    def act(self, obs: Observation) -> Action:  # pragma: no cover - Stream 2
        raise NotImplementedError(
            "LLMWorker is implemented by Stream 2 (A1). Phase 0 uses ShallowOracle."
        )

    @staticmethod
    def safe_default() -> Action:
        """The §4.4 fallback action on unrecoverable parse failure."""
        return Action(name=ActionName.WAIT)


__all__ = ["LLMWorker"]

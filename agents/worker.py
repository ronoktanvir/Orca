"""LLM worker turn loop (§4.2-4.5) — Stream 2 (A1, A2, A5, A6).

The real worker:
  * build the prompt (``prompts.build_worker_prompt``),
  * call the LLM through ``llm.complete(prompt, schema=WorkerOutput)``,
  * parse + validate strict JSON into one ``Action`` + zero-or-more ``Message``,
  * on malformed output do exactly one repair retry, else default to ``wait`` and
    record ``parse_failure`` (§4.4) — never crash,
  * sanitize messages: fill ``from``/``round``, default ``to`` to ``team``, and
    scrub any coordinate-like / seed-specific content (§3.2, §5.1),
  * at episode end, propose transferable HOW-TO heuristics and fold them into
    execution-memory, modulated by Orca's ``learning_signal`` (§4.5).

The worker exposes the synchronous ``act(obs) -> Action`` the run loop expects;
emitted messages are stashed on ``self.pending_messages`` for the loop to post to
the comm bus (kept off the protocol so the env's ``step(actions)`` is unchanged).
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Optional

from contracts import Action, BehaviorCard, ExecutionMemory, Heuristic, Message, Observation
from contracts.enums import ActionName, MessageType, Role

from .memory import (
    NEUTRAL_BAND,
    looks_seed_specific,
    scrub_seed_specific,
    update_execution_memory,
)
from .prompts import (
    MemoryProposal,
    WorkerOutput,
    build_memory_prompt,
    build_repair_prompt,
    build_worker_prompt,
    compact_history,
)

def _extract_json(raw: Any) -> dict:
    """Best-effort extraction of a single JSON object from raw model text (§4.4).

    Handles plain JSON, ```json fenced blocks, and a JSON object embedded in
    surrounding prose. Raises ``ValueError``/``JSONDecodeError`` on failure so the
    caller can trigger the one-shot repair.
    """
    if isinstance(raw, dict):
        return raw
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in model output")
    return json.loads(text[start : end + 1])


class LLMWorker:
    """A real LLM-backed worker (§4.2). Implements the ``Agent`` protocol."""

    def __init__(
        self,
        agent_id: str,
        llm: Any = None,
        card: Optional[BehaviorCard] = None,
        memory: Optional[ExecutionMemory] = None,
        history_summary: str = "",
        *,
        logger: Optional[Callable[[str, dict], None]] = None,
        max_messages_per_round: int = 2,
        history_keep: int = 6,
    ) -> None:
        self.agent_id = agent_id
        self.llm = llm
        self.card = card or BehaviorCard(agent_id=agent_id, role=Role.MINER)
        self.memory = memory or ExecutionMemory(agent_id=agent_id)
        self.history_summary = history_summary
        self._logger = logger
        self.max_messages_per_round = max_messages_per_round
        self.history_keep = history_keep

        # Per-turn / per-episode state surfaced to the run loop.
        self.pending_messages: list[Message] = []
        self.last_reasoning: str = ""
        self.parse_failures: int = 0
        self.messages_dropped: int = 0
        self._history_lines: list[str] = []

    # ------------------------------------------------------------------ #
    @property
    def role(self) -> Role:
        return self.card.role

    def _emit(self, event: str, data: dict) -> None:
        if self._logger is not None:
            try:
                self._logger(event, {"agent": self.agent_id, **data})
            except Exception:  # telemetry must never break the turn
                pass

    def _complete(self, prompt: str, schema: type) -> str:
        if self.llm is None:
            raise RuntimeError("LLMWorker has no llm client configured")
        return self.llm.complete(prompt, schema=schema)

    # ------------------------------------------------------------------ #
    def act(self, obs: Observation) -> Action:
        """One worker turn: obs -> Action (+ stashed messages) (§4.2, §4.4)."""
        self.pending_messages = []
        self.last_reasoning = ""

        if self.llm is None:
            self._record_parse_failure(obs, "", "no llm client", "")
            return self._finish(obs, self.safe_default())

        prompt = build_worker_prompt(obs, self.card, self.memory, self.history_summary)

        raw = ""
        try:
            raw = self._complete(prompt, WorkerOutput)
        except Exception as exc:  # network/SDK error — degrade, don't crash
            self._record_parse_failure(obs, raw, f"llm call failed: {exc}", "")
            return self._finish(obs, self.safe_default())

        parsed, err = self._try_parse(raw)
        if parsed is None:
            # Exactly one repair retry with the bad output + the validation error.
            repair_prompt = build_repair_prompt(prompt, raw, err or "")
            raw2 = ""
            try:
                raw2 = self._complete(repair_prompt, WorkerOutput)
            except Exception as exc:
                self._record_parse_failure(obs, raw, f"repair call failed: {exc}", raw2)
                return self._finish(obs, self.safe_default())
            parsed, err2 = self._try_parse(raw2)
            if parsed is None:
                self._record_parse_failure(obs, raw, err2 or err or "unparseable", raw2)
                return self._finish(obs, self.safe_default())

        self.last_reasoning = parsed.reasoning or ""
        action = self._sanitize_action(parsed.action)
        self.pending_messages = self._sanitize_messages(parsed.messages, obs)
        return self._finish(obs, action)

    # ------------------------------------------------------------------ #
    def _try_parse(self, raw: str) -> tuple[Optional[WorkerOutput], Optional[str]]:
        try:
            data = _extract_json(raw)
            return WorkerOutput.model_validate(data), None
        except Exception as exc:
            return None, str(exc)

    @staticmethod
    def _is_number(x: Any) -> bool:
        """A real number (not bool) — matches obs_guard's coordinate-pair test."""
        return isinstance(x, (int, float)) and not isinstance(x, bool)

    @staticmethod
    def _scrub_arg_value(value: Any) -> tuple[Any, bool]:
        """Recursively sanitize an action-arg value so it can never leak (§3.2).

        Returns ``(scrubbed_value, unusable)``. ``unusable`` is True when the
        value cannot be salvaged without leaking — i.e. a string that was
        *entirely* a coordinate/seed leak (scrubs to empty) or a 2-element numeric
        list/tuple (a coordinate pair, per ``obs_guard.scan_for_leaks``); the
        caller then turns the whole action into ``wait``. Leaky dict KEYS (region
        ids / "pos" / coordinate-like text) are dropped. Enum-ish values ("N",
        "iron_ore", "agent_2") and clean scalars (e.g. ``"n": 3``) pass through.
        """
        unusable = False
        if isinstance(value, str):
            if looks_seed_specific(value):
                cleaned = scrub_seed_specific(value)
                if value.strip() and not cleaned:
                    return cleaned, True  # entirely a leak -> unusable
                return cleaned, False
            return value, False
        if isinstance(value, dict):
            out: dict = {}
            for k, v in value.items():
                # Drop keys that themselves leak (e.g. "r_07", "pos").
                if isinstance(k, str) and looks_seed_specific(k):
                    continue
                nv, u = LLMWorker._scrub_arg_value(v)
                out[k] = nv
                unusable = unusable or u
            return out, unusable
        if isinstance(value, (list, tuple)):
            # A 2-element all-numeric list/tuple is a coordinate pair -> unusable.
            if len(value) == 2 and all(LLMWorker._is_number(x) for x in value):
                return value, True
            items = []
            for v in value:
                nv, u = LLMWorker._scrub_arg_value(v)
                items.append(nv)
                unusable = unusable or u
            return (tuple(items) if isinstance(value, tuple) else items), unusable
        return value, False

    def _sanitize_action(self, action: Action) -> Action:
        """Scrub coordinate-like/seed-specific content from ALL action args (§3.2).

        Walks every value AND key in ``action.args`` (through nested
        dict/list/tuple) so nothing ``obs_guard.scan_for_leaks`` would flag — a
        scrubbed string, a coordinate-shaped numeric pair, or a leaky key — ever
        reaches an ``ActionRecord``. Leaky keys are dropped; a numeric coordinate
        pair or an arg that scrubs to empty makes the action fall back to ``wait``.
        """
        if not action.args:
            return action
        sanitized, unusable = self._scrub_arg_value(action.args)
        if unusable:
            self._emit(
                "invalid_action_args",
                {"action": action.name.value, "reason": "coordinate-like or seed-specific arg"},
            )
            return self.safe_default()
        if sanitized == action.args:
            return action
        return Action(name=action.name, args=sanitized)

    def _sanitize_messages(self, drafts: list, obs: Observation) -> list[Message]:
        """Coerce model drafts into validated ``Message`` objects (§4.4, §5.1).

        Fills ``from`` (this worker), ``round`` (current obs round), defaults
        ``to`` to ``team``, coerces ``type``, clamps ``urgency``, scrubs
        coordinate-like content, and drops anything empty after scrubbing.
        Caps the count at ``max_messages_per_round`` (bandwidth realism, §5.1).
        """
        out: list[Message] = []
        for d in drafts:
            content = (getattr(d, "content", "") or "").strip()
            if not content:
                continue
            if looks_seed_specific(content):
                content = scrub_seed_specific(content)
                self.messages_dropped += 1  # count the sanitization for the trace
                if not content:
                    continue
            raw_type = getattr(d, "type", "report")
            try:
                mtype = MessageType(raw_type)
            except ValueError:
                mtype = MessageType.REPORT
            to = (getattr(d, "to", "") or "team").strip() or "team"
            try:
                urgency = max(0.0, min(1.0, float(getattr(d, "urgency", 0.3))))
            except (TypeError, ValueError):
                urgency = 0.3
            out.append(
                Message(
                    **{"from": self.agent_id},
                    to=to,
                    type=mtype,
                    content=content,
                    urgency=urgency,
                    round=obs.round,
                )
            )
            if len(out) >= self.max_messages_per_round:
                break
        return out

    def _finish(self, obs: Observation, action: Action) -> Action:
        """Update the bounded running history summary, then return the action."""
        self._history_lines.append(f"r{obs.round}:{action.name.value}")
        self.history_summary = compact_history(self._history_lines, self.history_keep)
        self._emit(
            "worker_decision",
            {
                "round": obs.round,
                "action": action.name.value,
                "n_messages": len(self.pending_messages),
                "reasoning": self.last_reasoning,
            },
        )
        return action

    def _record_parse_failure(self, obs: Observation, raw: str, error: str, raw2: str) -> None:
        self.parse_failures += 1
        self.pending_messages = []
        self._emit(
            "parse_failure",
            {
                "round": getattr(obs, "round", -1),
                "error": error,
                "raw": raw,
                "raw_repair": raw2,
            },
        )

    # ------------------------------------------------------------------ #
    # Episode-end execution-memory write (§4.5)
    # ------------------------------------------------------------------ #
    def propose_memory(self, episode_digest: str) -> list[Heuristic]:
        """Ask the LLM for ONLY transferable HOW-TO heuristics (§4.5).

        Robust like ``act``: one repair retry, else return ``[]``. Confidence is
        clamped and empty drafts dropped; coordinate scrubbing happens in
        :func:`update_execution_memory`.
        """
        if self.llm is None:
            return []
        prompt = build_memory_prompt(self.card, self.memory, episode_digest)
        try:
            raw = self._complete(prompt, MemoryProposal)
        except Exception:
            return []
        proposal, err = self._try_parse_memory(raw)
        if proposal is None:
            repair = build_repair_prompt(prompt, raw, err or "")
            try:
                raw2 = self._complete(repair, MemoryProposal)
            except Exception:
                return []
            proposal, _ = self._try_parse_memory(raw2)
            if proposal is None:
                return []
        heuristics: list[Heuristic] = []
        for h in proposal.heuristics:
            cond = (h.condition or "").strip()
            act = (h.action or "").strip()
            if not cond or not act:
                continue
            heuristics.append(
                Heuristic(condition=cond, action=act, confidence=max(0.0, min(1.0, float(h.confidence))))
            )
        return heuristics

    def _try_parse_memory(self, raw: str) -> tuple[Optional[MemoryProposal], Optional[str]]:
        try:
            data = _extract_json(raw)
            return MemoryProposal.model_validate(data), None
        except Exception as exc:
            return None, str(exc)

    def end_episode_update(
        self, episode_digest: str, learning_signal: float = 0.0
    ) -> ExecutionMemory:
        """Fold episode-end heuristics into memory, scaled by ``learning_signal`` (§4.5).

        With ``|learning_signal|`` ~ 0 (Phase 0 / no-op Orca) this makes no LLM
        call and leaves memory unchanged (beyond a guard-filter pass), keeping the
        offline path cheap. A positive signal proposes + bakes in heuristics; a
        negative signal weakens/removes flagged rules.
        """
        proposed: list[Heuristic] = []
        if abs(learning_signal) > NEUTRAL_BAND:
            proposed = self.propose_memory(episode_digest)
        self.memory = update_execution_memory(self.memory, proposed, learning_signal)
        self._emit(
            "memory_update",
            {
                "learning_signal": learning_signal,
                "n_heuristics": len(self.memory.heuristics),
            },
        )
        return self.memory

    # ------------------------------------------------------------------ #
    @staticmethod
    def safe_default() -> Action:
        """The §4.4 fallback action on unrecoverable parse failure."""
        return Action(name=ActionName.WAIT)


__all__ = ["LLMWorker"]

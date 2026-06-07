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

from bus.messages import normalize_recipient

from contracts import Action, BehaviorCard, ExecutionMemory, Heuristic, Message, Observation
from contracts.enums import ActionName, MessageType, Role

from .memory import (
    NEUTRAL_BAND,
    looks_seed_specific,
    sanitize_action_args,
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
        role: Optional[Role] = None,
        logger: Optional[Callable[[str, dict], None]] = None,
        max_messages_per_round: int = 2,
        history_keep: int = 6,
    ) -> None:
        self.agent_id = agent_id
        self.llm = llm
        # When no card is supplied, keep the roster ``role`` (the advertised seam
        # must not silently fall back to miner, §4.1). ``role`` is an optional
        # kwarg so existing positional constructor calls keep working.
        self.card = card or BehaviorCard(agent_id=agent_id, role=role or Role.MINER)
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
        self._history_lines: list[str] = []  # compact per-round action notes
        # Bounded running history of *older* message events (those that have
        # scrolled out of the live recent window). ``_prev_recent`` lets us detect
        # the scroll-out; ``_folded`` holds (key, note) deduped by key; ``_live_keys``
        # is the current window so a message that is live again is excluded from the
        # summary — never both shown live and summarized at once (§4.3/§5.2).
        self._folded: list[tuple] = []
        self._folded_keys: set = set()
        self._prev_recent: list[Message] = []
        self._live_keys: set = set()

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
        # Fold any messages that scrolled out of the live window into the bounded
        # running history BEFORE building the prompt (§4.3/§5.2).
        self._observe_messages(obs)

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

    def _sanitize_action(self, action: Action) -> Action:
        """Scrub coordinate-like/seed-specific content from ALL action args (§3.2).

        Delegates to the shared :func:`agents.memory.sanitize_action_args` (the
        same enforcement the run loop applies at the env boundary) and adds the
        worker's telemetry when an unsalvageable arg forces a ``wait`` fallback."""
        result = sanitize_action_args(action)
        if result.name == ActionName.WAIT and action.name != ActionName.WAIT:
            self._emit(
                "invalid_action_args",
                {"action": action.name.value, "reason": "coordinate-like or seed-specific arg"},
            )
        return result

    def _sanitize_messages(self, drafts: list, obs: Observation) -> list[Message]:
        """Coerce model drafts into validated ``Message`` objects (§4.4, §5.1).

        Fills ``from`` (this worker), ``round`` (current obs round), **validates
        the recipient** (``to`` may only be ``team`` / ``orca`` / an ``agent_<n>``
        id — anything leaky like ``r_07`` is downgraded to ``team``, §3.2), coerces
        ``type``, clamps ``urgency``, scrubs coordinate-like content, and drops
        anything empty after scrubbing. Caps the count at
        ``max_messages_per_round`` (bandwidth realism, §5.1).
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
            # The model cannot be trusted with the recipient: validate it so an
            # internal region id / coordinate-like string never reaches Message.to.
            to = normalize_recipient(getattr(d, "to", ""))
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

    @staticmethod
    def _action_hint(action: Action) -> str:
        """A compact, leak-free arg hint for the history line (e.g. the crafted
        item / gathered resource / move direction). Makes a repeated invalid
        attempt visible in the compacted history, not just the live obs (§4.3)."""
        args = action.args or {}
        for key in ("item", "resource", "direction", "target", "agent"):
            val = args.get(key)
            if isinstance(val, str) and val:
                return val
        return ""

    def _finish(self, obs: Observation, action: Action) -> Action:
        """Update the bounded running history summary, then return the action."""
        hint = self._action_hint(action)
        line = f"r{obs.round}:{action.name.value}" + (f"({hint})" if hint else "")
        self._history_lines.append(line)
        self.history_summary = self._compose_history()
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

    # ------------------------------------------------------------------ #
    # Bounded running history (actions + older message events) (§4.3/§5.2)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _msg_key(m: Message) -> tuple:
        mtype = getattr(getattr(m, "type", None), "value", str(getattr(m, "type", "")))
        return (m.from_agent, m.to, m.round, mtype, m.content)

    @staticmethod
    def _compact_msg_note(m: Message) -> str:
        """A compact, leak-free note for a message that has left the live window.

        Carries sender / recipient / type / round — meaningful context for older
        turns — but not the raw content (which was shown in full while live, and
        whose omission keeps the summary bounded and out of the leak surface)."""
        mtype = getattr(getattr(m, "type", None), "value", str(getattr(m, "type", "")))
        return f"r{m.round} {m.from_agent}->{m.to} [{mtype}]"

    def _observe_messages(self, obs: Observation) -> None:
        """Fold messages that scrolled out of the live recent window into history.

        The live window (``obs.recent_messages``) is shown in full in the prompt;
        here we record the *older* messages — those seen previously but no longer
        live — exactly once (deduped by key). The summary then excludes anything
        currently live, so recent and summarized message context stay disjoint even
        if a message re-enters the window (no double-count), bounded + leak-free.
        """
        current = list(getattr(obs, "recent_messages", []) or [])
        self._live_keys = {self._msg_key(m) for m in current}
        for m in self._prev_recent:
            k = self._msg_key(m)
            if k in self._live_keys or k in self._folded_keys:
                continue
            self._folded_keys.add(k)
            self._folded.append((k, self._compact_msg_note(m)))
        self._prev_recent = current
        self.history_summary = self._compose_history()

    def _compose_history(self) -> str:
        """Combine the action history and older-message history into one bounded,
        leak-free summary string (§4.3)."""
        parts: list[str] = []
        if self._history_lines:
            parts.append("actions: " + compact_history(self._history_lines, self.history_keep))
        # Only messages NOT currently live (so a re-entered message is never both
        # shown live in the obs and summarized here).
        older = [note for k, note in self._folded if k not in self._live_keys]
        if older:
            parts.append("messages: " + compact_history(older, self.history_keep))
        composed = "\n".join(parts)
        # Defense in depth: never let a coordinate-like span survive into the prompt.
        return scrub_seed_specific(composed) if looks_seed_specific(composed) else composed

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

        The candidate rules depend on the sign of the coach's signal:
          * ``~0`` (|signal| <= NEUTRAL_BAND): no LLM call, memory unchanged beyond a
            guard-filter pass (keeps the offline path cheap).
          * ``> 0``: ask the LLM for NEW transferable how-to to add/strengthen.
          * ``< 0``: the coach judged this agent's lessons net-harmful — the
            weaken/remove candidates are its OWN current heuristics. We feed those
            in directly rather than calling the "propose good how-to" LLM (which
            would never surface a rule to drop), so a negative signal actually
            weakens/removes memory instead of relying on a coincidental re-proposal.
        """
        proposed: list[Heuristic] = []
        if learning_signal > NEUTRAL_BAND:
            proposed = self.propose_memory(episode_digest)
        elif learning_signal < -NEUTRAL_BAND:
            proposed = list(self.memory.heuristics)
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

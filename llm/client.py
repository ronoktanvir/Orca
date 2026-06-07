"""LLM client interface (§11): ``llm.complete(prompt, schema)``.

One client behind an interface so the model is swappable (OpenAI <-> W&B
Inference/GLM <-> local) via config (§11). Phase 0 needs no real model (the
oracle is scripted), but Streams 2 (workers) and 3 (Orca) both do — so this
ships the real **OpenAI-backed** client plus a ``build_llm`` factory that reads
the model per role from config.

Design notes:
  * ``complete`` returns the raw text. When a ``schema`` is passed it requests
    JSON mode and nudges the system prompt — but **validation + the one-shot
    repair retry is the caller's job** (§4.4 / Stream 2 A2), kept out of here so
    the client stays a thin seam.
  * The SDK client is created lazily, so importing/constructing this never needs
    a key — only an actual ``complete`` call does (keeps pytest offline).
  * Reasoning models (gpt-5) reject ``temperature``; we retry without it.
  * Decorated with ``@op`` so calls nest in the Weave trace (§10); Weave also
    auto-instruments the underlying OpenAI call.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, Protocol, runtime_checkable

from telemetry import op

_log = logging.getLogger(__name__)


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
        return "{}"


class OpenAIClient:
    """OpenAI-compatible chat client (also serves W&B Inference via ``base_url``)."""

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.2,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: Optional[str] = None,
        default_headers: Optional[dict[str, str]] = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self._api_key_env = api_key_env
        self._base_url = base_url
        self._default_headers = default_headers
        self._client: Any = None  # lazy

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("openai SDK not installed (uv pip install openai)") from exc
            key = os.environ.get(self._api_key_env)
            if not key:
                raise RuntimeError(
                    f"{self._api_key_env} not set — put it in .env (config.load_dotenv) or export it"
                )
            self._client = OpenAI(
                api_key=key, base_url=self._base_url, default_headers=self._default_headers
            )
        return self._client

    @op
    def complete(
        self,
        prompt: str,
        schema: Optional[type] = None,
        *,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        client = self._ensure_client()
        mdl = model or self.model
        sys_msg = system or "You are a careful agent. Follow the requested output format exactly."
        if schema is not None:
            sys_msg += " Respond with a single valid JSON object and nothing else."
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt},
        ]
        kwargs: dict[str, Any] = {"model": mdl, "messages": messages}
        if schema is not None:
            kwargs["response_format"] = {"type": "json_object"}
        temp = self.temperature if temperature is None else temperature

        try:
            resp = client.chat.completions.create(temperature=temp, **kwargs)
        except Exception as exc:  # reasoning models reject temperature -> retry without
            if "temperature" in str(exc).lower():
                resp = client.chat.completions.create(**kwargs)
            else:
                raise
        return resp.choices[0].message.content or ""


class FallbackLLM:
    """A primary client with an automatic backup (§11).

    Tries ``primary.complete`` first; on **any** exception (rate limit, 5xx,
    endpoint down, ...) logs a warning and retries once on ``backup``. JSON
    validation is the caller's job (§4.4), so an exception reaching here is always
    a transport/API failure — exactly when a backup helps. If the backup also
    fails, a single error chaining both is raised.
    """

    def __init__(
        self,
        primary: LLMClient,
        backup: LLMClient,
        *,
        primary_name: str = "primary",
        backup_name: str = "backup",
    ) -> None:
        self.primary = primary
        self.backup = backup
        self.primary_name = primary_name
        self.backup_name = backup_name

    @op
    def complete(self, prompt: str, schema: Optional[type] = None, **kwargs: Any) -> str:
        try:
            return self.primary.complete(prompt, schema, **kwargs)
        except Exception as primary_exc:
            _log.warning(
                "LLM primary (%s) failed: %s — falling back to %s",
                self.primary_name,
                primary_exc,
                self.backup_name,
            )
            try:
                return self.backup.complete(prompt, schema, **kwargs)
            except Exception as backup_exc:
                raise RuntimeError(
                    f"both LLM providers failed — primary={self.primary_name} "
                    f"({primary_exc!r}); backup={self.backup_name} ({backup_exc!r})"
                ) from backup_exc


def _provider_for(role: str, cfg: Any) -> str:
    """Resolve the provider for ``role`` (per-role override -> global default)."""
    per_role = cfg.orca_provider if role == "orca" else cfg.worker_provider
    return per_role or cfg.provider


def _build_single(role: str, settings: Any, provider: str) -> LLMClient:
    """Build one (unwrapped) client for ``role`` on the given ``provider``.
      * 'openai'          -> worker_model / orca_model on the OpenAI API.
      * 'wandb_inference' -> GLM on the W&B Inference OpenAI-compatible endpoint
        (worker uses ``wandb_inference_model``; Orca uses
        ``wandb_inference_orca_model`` if set, else the same), billed to W&B credits.
    """
    cfg = settings.llm
    if provider == "wandb_inference":
        headers = None
        # Prefer the configured telemetry identifiers (single source of truth),
        # falling back to env vars — so usage attribution matches the Weave run.
        tel = getattr(settings, "telemetry", None)
        entity = (getattr(tel, "entity", None) if tel else None) or os.environ.get("WANDB_ENTITY")
        project = (getattr(tel, "project", None) if tel else None) or os.environ.get("WANDB_PROJECT", "orca")
        if entity:
            # W&B Inference attributes usage to a project via this header.
            headers = {"OpenAI-Project": f"{entity}/{project}"}
        model = cfg.wandb_inference_model
        if role == "orca" and cfg.wandb_inference_orca_model:
            model = cfg.wandb_inference_orca_model
        return OpenAIClient(
            model=model,
            temperature=cfg.temperature,
            api_key_env="WANDB_INFERENCE_API_KEY",
            base_url=cfg.wandb_inference_base_url,
            default_headers=headers,
        )

    model = cfg.orca_model if role == "orca" else cfg.worker_model
    return OpenAIClient(model=model, temperature=cfg.temperature, base_url=cfg.openai_base_url)


def build_llm(role: str, settings: Any) -> LLMClient:
    """Construct the configured LLM for a role ('worker' | 'orca') (§11).

    Provider is resolved per role (``worker_provider`` / ``orca_provider`` override
    the global ``provider``). When ``fallback_provider`` is set and differs from the
    resolved primary, the client is wrapped in :class:`FallbackLLM` so a failed
    primary call (e.g. GLM/W&B unavailable) automatically retries on the backup
    (e.g. OpenAI). Default config: GLM-5.1 primary + OpenAI backup.
    """
    cfg = settings.llm
    provider = _provider_for(role, cfg)
    primary = _build_single(role, settings, provider)

    fallback = getattr(cfg, "fallback_provider", None)
    if fallback and fallback != provider:
        backup = _build_single(role, settings, fallback)
        return FallbackLLM(primary, backup, primary_name=provider, backup_name=fallback)
    return primary


__all__ = ["LLMClient", "StubLLM", "OpenAIClient", "FallbackLLM", "build_llm"]

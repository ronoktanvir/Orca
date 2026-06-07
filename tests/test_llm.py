"""LLM client seam (§11): factory + offline behavior. No network in pytest."""

from __future__ import annotations

import os

import pytest

from config import load_config
from llm import FallbackLLM, LLMClient, OpenAIClient, StubLLM, build_llm


def test_stub_llm_is_offline():
    s = StubLLM()
    assert isinstance(s, LLMClient)
    assert s.complete("anything") == "{}"


def test_build_llm_openai_roles_pick_models():
    cfg = load_config()
    cfg.llm.provider = "openai"
    worker = build_llm("worker", cfg)
    orca = build_llm("orca", cfg)
    assert isinstance(worker, OpenAIClient) and isinstance(orca, OpenAIClient)
    assert worker.model == cfg.llm.worker_model
    assert orca.model == cfg.llm.orca_model


def test_build_llm_wandb_inference_provider():
    cfg = load_config()
    cfg.llm.provider = "wandb_inference"
    cfg.llm.fallback_provider = None  # isolate the bare wandb client (no wrapper)
    client = build_llm("worker", cfg)
    assert isinstance(client, OpenAIClient)
    assert client.model == cfg.llm.wandb_inference_model
    assert client._base_url == cfg.llm.wandb_inference_base_url
    assert client._api_key_env == "WANDB_INFERENCE_API_KEY"


def test_build_llm_per_role_hybrid():
    # GLM workers (W&B Inference) + gpt-5 Orca (OpenAI) via per-role providers.
    cfg = load_config()
    cfg.llm.provider = "openai"
    cfg.llm.worker_provider = "wandb_inference"  # workers -> GLM
    cfg.llm.fallback_provider = None  # isolate provider resolution from the backup wrapper
    # orca_provider unset -> falls back to provider "openai"
    worker = build_llm("worker", cfg)
    orca = build_llm("orca", cfg)
    assert worker.model == cfg.llm.wandb_inference_model
    assert worker._base_url == cfg.llm.wandb_inference_base_url
    assert worker._api_key_env == "WANDB_INFERENCE_API_KEY"
    assert orca.model == cfg.llm.orca_model
    assert orca._api_key_env == "OPENAI_API_KEY"


def test_build_llm_wandb_inference_distinct_orca_model():
    cfg = load_config()
    cfg.llm.provider = "wandb_inference"
    cfg.llm.fallback_provider = None  # isolate model resolution from the backup wrapper
    cfg.llm.wandb_inference_orca_model = "zai-org/GLM-5.1-pro"
    worker = build_llm("worker", cfg)
    orca = build_llm("orca", cfg)
    assert worker.model == cfg.llm.wandb_inference_model
    assert orca.model == "zai-org/GLM-5.1-pro"


def test_build_llm_default_wraps_glm_primary_with_openai_backup():
    # The default config: GLM-5.1 (W&B Inference) primary + OpenAI fallback.
    cfg = load_config()
    assert cfg.llm.provider == "wandb_inference"
    assert cfg.llm.fallback_provider == "openai"
    worker = build_llm("worker", cfg)
    assert isinstance(worker, FallbackLLM)
    # primary -> GLM on the W&B endpoint; backup -> OpenAI worker model
    assert worker.primary.model == cfg.llm.wandb_inference_model
    assert worker.primary._api_key_env == "WANDB_INFERENCE_API_KEY"
    assert worker.backup.model == cfg.llm.worker_model
    assert worker.backup._api_key_env == "OPENAI_API_KEY"
    # Orca: GLM-5.1 primary, gpt-5 backup
    orca = build_llm("orca", cfg)
    assert isinstance(orca, FallbackLLM)
    assert orca.primary.model == cfg.llm.wandb_inference_model
    assert orca.backup.model == cfg.llm.orca_model


def test_build_llm_no_wrap_when_fallback_equals_primary():
    cfg = load_config()
    cfg.llm.provider = "openai"
    cfg.llm.fallback_provider = "openai"  # same as primary -> no point wrapping
    client = build_llm("worker", cfg)
    assert isinstance(client, OpenAIClient)


class _RaisingLLM:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def complete(self, prompt, schema=None, **kwargs):
        raise self.exc


class _EchoLLM:
    def __init__(self, out: str) -> None:
        self.out = out
        self.calls: list[str] = []

    def complete(self, prompt, schema=None, **kwargs):
        self.calls.append(prompt)
        return self.out


def test_fallback_llm_uses_backup_on_primary_error():
    backup = _EchoLLM('{"from": "backup"}')
    fb = FallbackLLM(_RaisingLLM(RuntimeError("GLM down")), backup, primary_name="wandb_inference", backup_name="openai")
    out = fb.complete("hi", schema=dict)
    assert out == '{"from": "backup"}'
    assert backup.calls == ["hi"]  # backup actually received the prompt


def test_fallback_llm_prefers_primary_when_it_succeeds():
    backup = _EchoLLM('{"from": "backup"}')
    fb = FallbackLLM(_EchoLLM('{"from": "primary"}'), backup)
    assert fb.complete("hi") == '{"from": "primary"}'
    assert backup.calls == []  # backup never touched on a primary success


def test_fallback_llm_raises_when_both_fail():
    fb = FallbackLLM(
        _RaisingLLM(RuntimeError("GLM down")),
        _RaisingLLM(RuntimeError("OpenAI down")),
    )
    with pytest.raises(RuntimeError, match="both LLM providers failed"):
        fb.complete("hi")


def test_openai_client_constructs_without_key():
    # Lazy: construction must not require a key (keeps pytest offline).
    c = OpenAIClient(model="gpt-5-mini")
    assert c._client is None


def test_openai_client_complete_errors_clearly_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    c = OpenAIClient(model="gpt-5-mini")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        c.complete("hi")


@pytest.mark.skipif(
    not (os.environ.get("ORCA_LIVE_LLM") and os.environ.get("OPENAI_API_KEY")),
    reason="live LLM test — set ORCA_LIVE_LLM=1 and OPENAI_API_KEY to run",
)
def test_live_openai_smoke():
    cfg = load_config()
    cfg.llm.provider = "openai"  # default is now GLM; pin OpenAI for this live test
    cfg.llm.fallback_provider = None
    client = build_llm("worker", cfg)
    out = client.complete('Reply with JSON {"ok": true} only.', schema=dict)
    assert "ok" in out.lower()

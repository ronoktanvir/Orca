"""LLM client seam (§11): factory + offline behavior. No network in pytest."""

from __future__ import annotations

import os

import pytest

from config import load_config
from llm import LLMClient, OpenAIClient, StubLLM, build_llm


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
    cfg.llm.wandb_inference_orca_model = "zai-org/GLM-4.6-bigger"
    worker = build_llm("worker", cfg)
    orca = build_llm("orca", cfg)
    assert worker.model == cfg.llm.wandb_inference_model
    assert orca.model == "zai-org/GLM-4.6-bigger"


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
    client = build_llm("worker", cfg)
    out = client.complete('Reply with JSON {"ok": true} only.', schema=dict)
    assert "ok" in out.lower()

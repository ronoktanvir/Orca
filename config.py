"""Config loading + validation (§15).

``configs/default.yaml`` holds every tuning knob. The structure here is part of
the frozen handoff (workflow §2.4): streams read knobs from these typed sections.
Adding a *new* knob is additive; renaming/removing is a broadcast change.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
DEFAULT_ENV_PATH = REPO_ROOT / ".env"


def load_dotenv(path: str | Path | None = None) -> list[str]:
    """Load KEY=VALUE pairs from a gitignored ``.env`` into ``os.environ``.

    Minimal, dependency-free. Never overrides a variable already set in the
    shell. Returns the list of keys it set (values never logged). Secrets stay
    out of the repo: the real ``.env`` is gitignored; ``.env.example`` documents
    the keys (OPENAI_API_KEY, WANDB_API_KEY, WANDB_INFERENCE_API_KEY, ...).
    """
    path = Path(path) if path else DEFAULT_ENV_PATH
    if not path.exists():
        return []
    set_keys: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            set_keys.append(key)
    return set_keys


class RunConfig(BaseModel):
    seed: str = "A"
    n_episodes: int = 1
    t_max: int = 120
    day_length: int = 100
    stop_at_milestone: Optional[str] = "iron"  # Phase 0 shallow target
    message_window: int = 8
    single_agent_oracle: bool = True  # Phase 0 uses the scripted oracle
    # 0 => AUTO: one worker call per agent each round, so the full 4-agent team
    # runs its calls concurrently (asyncio) while the single-agent oracle stays
    # sequential. A positive int pins the concurrency explicitly (tests/eval).
    worker_concurrency: int = 0


class AgentsConfig(BaseModel):
    temperature: float = 0.2  # §15 worker determinism (LLM workers, Stream 2)


class LLMConfig(BaseModel):
    """Swappable model config (§11). provider: openai | wandb_inference.

    Default: **GLM-5.1 on W&B Inference** for both roles, with **OpenAI as the
    automatic backup** (``fallback_provider``) when a primary call fails. The
    per-role ``worker_provider`` / ``orca_provider`` still override ``provider``
    (e.g. the GLM-workers + gpt-5-Orca hybrid). ``worker_model`` / ``orca_model``
    double as the OpenAI fallback models."""

    provider: str = "wandb_inference"  # primary: GLM-5.1 via W&B Inference ($50 credits)
    worker_provider: Optional[str] = None  # overrides provider for the 4 workers
    orca_provider: Optional[str] = None  # overrides provider for Orca
    # Backup provider used when the primary ``complete()`` raises (rate limit, 5xx,
    # endpoint down). None disables fallback; skipped when it equals the primary.
    fallback_provider: Optional[str] = "openai"
    worker_model: str = "gpt-5-mini"  # workers on OpenAI (also the worker fallback)
    orca_model: str = "gpt-5"  # Orca on OpenAI (also the Orca fallback)
    temperature: float = 0.2  # §15
    openai_base_url: Optional[str] = None  # None => api.openai.com
    # W&B Inference (GLM) — billed to W&B credits.
    wandb_inference_base_url: str = "https://api.inference.wandb.ai/v1"
    wandb_inference_model: str = "zai-org/GLM-5.1"  # primary worker+Orca model (confirmed live)
    wandb_inference_orca_model: Optional[str] = None  # None => same as wandb_inference_model


class RewardConfig(BaseModel):
    weights: dict[str, float] = Field(
        default_factory=lambda: {"deaths": 0.02, "invalid": 0.05, "idle": 0.05}
    )


class BanditConfig(BaseModel):
    epsilon: float = 0.2


class PhasesConfig(BaseModel):
    phase0_length: int = 15
    memory_cap: int = 8


class SeedsConfig(BaseModel):
    train: list[str] = Field(default_factory=lambda: ["A", "T2", "T3"])
    heldout: list[str] = Field(default_factory=lambda: ["B", "C"])


class EvalConfig(BaseModel):
    """Eval-campaign knobs (§9), read by ``eval.run_eval`` / the harness."""

    n_train: int = 40  # Full C2 training episodes
    eval_reps: int = 8  # eval repetitions per seed (variance)
    gate_batch: int = 2  # train-pool episodes the accept-gate re-runs per coach episode


class TelemetryConfig(BaseModel):
    mode: str = "auto"  # auto | weave | local | off
    entity: Optional[str] = None  # W&B entity (team); None => W&B default entity
    project: str = "orca"
    run_dir: str = "runs"


class OrcaSettings(BaseModel):
    run: RunConfig = Field(default_factory=RunConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    bandit: BanditConfig = Field(default_factory=BanditConfig)
    phases: PhasesConfig = Field(default_factory=PhasesConfig)
    seeds: SeedsConfig = Field(default_factory=SeedsConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)


def load_config(path: str | Path | None = None) -> OrcaSettings:
    """Load + validate config from YAML (defaults if file/keys absent)."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        return OrcaSettings()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return OrcaSettings(**data)


__all__ = [
    "OrcaSettings",
    "load_config",
    "load_dotenv",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_ENV_PATH",
    "REPO_ROOT",
]

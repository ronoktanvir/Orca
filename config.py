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


class AgentsConfig(BaseModel):
    temperature: float = 0.2  # §15 worker determinism (LLM workers, Stream 2)


class LLMConfig(BaseModel):
    """Swappable model config (§11). provider: openai | wandb_inference."""

    provider: str = "openai"
    worker_model: str = "gpt-5-mini"  # 4 workers: cheap/fast, high call volume
    orca_model: str = "gpt-5"  # Orca: strong reasoning, 1 call/episode
    temperature: float = 0.2  # §15
    openai_base_url: Optional[str] = None  # None => api.openai.com
    # W&B Inference (GLM-5.1) — billed to W&B credits; an alternate/ablation provider.
    wandb_inference_base_url: str = "https://api.inference.wandb.ai/v1"
    wandb_inference_model: str = "zai-org/GLM-4.6"  # set to the GLM-5.1 id when confirmed


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

"""Config loading + validation (§15).

``configs/default.yaml`` holds every tuning knob. The structure here is part of
the frozen handoff (workflow §2.4): streams read knobs from these typed sections.
Adding a *new* knob is additive; renaming/removing is a broadcast change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


class RunConfig(BaseModel):
    seed: str = "A"
    n_episodes: int = 1
    t_max: int = 600
    day_length: int = 100
    stop_at_milestone: Optional[str] = "iron"  # Phase 0 shallow target
    message_window: int = 8
    single_agent_oracle: bool = True  # Phase 0 uses the scripted oracle


class AgentsConfig(BaseModel):
    temperature: float = 0.2  # §15 worker determinism (LLM workers, Stream 2)


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
    project: str = "orca"
    run_dir: str = "runs"


class OrcaSettings(BaseModel):
    run: RunConfig = Field(default_factory=RunConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
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


__all__ = ["OrcaSettings", "load_config", "DEFAULT_CONFIG_PATH", "REPO_ROOT"]

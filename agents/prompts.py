"""Worker prompt construction (§4.3) — Stream 2 (A1, A6).

System prompt = ROLE_PRIMER[role] + behavior_card + execution_memory +
ACTION_SPEC + OUTPUT_SCHEMA. User prompt = the JSON observation + a compacted
history summary + the team DAG frontier. Phase 0 provides the role primers and a
placeholder builder; Stream 2 implements the full prompt + strict-JSON schema.
"""

from __future__ import annotations

from contracts import BehaviorCard, ExecutionMemory, Observation
from contracts.enums import Role

ROLE_PRIMERS: dict[Role, str] = {
    Role.EXPLORER: "You are the Explorer: scout outward, reveal regions and structures, find biomes.",
    Role.MINER: "You are the Miner: gather cobblestone/coal/iron, manage lava/obsidian logistics.",
    Role.TINKERER: "You are the Tinkerer: craft, smelt, build gear and the nether portal.",
    Role.SUPPORT: "You are Support: food/hunger, combat, shelter, escorting and reviving.",
}


def build_worker_prompt(
    obs: Observation,
    card: BehaviorCard,
    memory: ExecutionMemory,
    history_summary: str = "",
) -> str:  # pragma: no cover - Stream 2 fills this in
    """Placeholder for the Stream 2 worker prompt builder (§4.3)."""
    raise NotImplementedError("build_worker_prompt is implemented by Stream 2 (A1)")


__all__ = ["ROLE_PRIMERS", "build_worker_prompt"]

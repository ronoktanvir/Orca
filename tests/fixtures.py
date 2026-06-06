"""Contract validation fixtures — one valid sample per frozen contract (§11).

Used by the contract tests and available for any stream to build against. Each
``sample_*`` returns a fully-valid instance of the corresponding contract.
"""

from __future__ import annotations

from contracts import (
    Action,
    AgentStats,
    BehaviorCard,
    EpisodeMetrics,
    EpisodeTrace,
    ExecutionMemory,
    Heuristic,
    Message,
    Observation,
)
from contracts.enums import (
    ActionName,
    Bearing,
    Biome,
    DistanceBand,
    Layer,
    MessageType,
    Milestone,
    Role,
    TimeOfDay,
)
from contracts.episode import ActionRecord, MilestoneEvent


def sample_message() -> Message:
    return Message(
        **{"from": "agent_1"},
        to="team",
        type=MessageType.REPORT,
        content="Found caves with iron to the N",
        urgency=0.4,
        round=42,
    )


def sample_observation() -> Observation:
    return Observation(
        round=42,
        time_of_day=TimeOfDay.NIGHT,
        self={
            "role": Role.MINER,
            "health": 0.6,
            "hunger": 0.4,
            "inventory": {"cobblestone": 12, "iron_ore": 3, "wooden_pickaxe": 1},
            "status": "free",
            "current_biome": Biome.MOUNTAINS,
            "layer": Layer.OVERWORLD,
        },
        here={
            "resources_visible": ["coal", "iron_ore"],
            "structure": None,
            "mobs": ["zombie"],
            "exits": [
                {"dir": Bearing.N, "distance_band": DistanceBand.NEAR, "biome_hint": Biome.CAVES},
                {"dir": Bearing.SE, "distance_band": DistanceBand.FAR, "biome_hint": Biome.UNKNOWN},
            ],
            "frontier_dirs": [Bearing.E, Bearing.W],
        },
        teammates=[
            {
                "agent": "agent_3",
                "distance_band": DistanceBand.SAME_REGION,
                "bearing": None,
                "role": Role.TINKERER,
            }
        ],
        known_landmarks=[{"type": "lava_pool", "rel_dir": Bearing.S, "distance_band": DistanceBand.FAR}],
        recent_messages=[sample_message()],
        assignment="Mine iron until you have 6 ingots, then regroup with tinkerer.",
        dag_frontier_reached="iron",
    )


def sample_action() -> Action:
    return Action(name=ActionName.GATHER, args={"resource": "iron_ore"})


def sample_behavior_card() -> BehaviorCard:
    return BehaviorCard(
        agent_id="agent_2",
        role=Role.MINER,
        assignment="Mine iron until you have 6 ingots, then regroup.",
        directives=["craft a stone pickaxe before mining iron"],
        priorities=["iron tooling"],
        donts=["don't mine without the right pickaxe"],
        version=1,
    )


def sample_execution_memory() -> ExecutionMemory:
    return ExecutionMemory(
        agent_id="agent_2",
        heuristics=[
            Heuristic(
                condition="need iron but only wooden pickaxe",
                action="craft stone pickaxe first",
                confidence=0.8,
            )
        ],
    )


def sample_episode_trace() -> EpisodeTrace:
    return EpisodeTrace(
        episode_idx=0,
        seed="A",
        n_rounds=14,
        agent_ids=["agent_1"],
        config={"arms": {}, "roster": [("agent_1", "miner")]},
        behavior_cards=[sample_behavior_card()],
        action_records=[
            ActionRecord(
                round=0,
                agent_id="agent_1",
                action=Action(name=ActionName.GATHER, args={"resource": "wood"}),
                valid=True,
                result={"gathered": {"wood": 3}},
            ),
            ActionRecord(
                round=1,
                agent_id="agent_1",
                action=Action(name=ActionName.CRAFT, args={"item": "diamond_armor"}),
                valid=False,
                reason="need 24 diamond (have 0)",
            ),
        ],
        messages=[sample_message()],
        milestone_timeline=[
            MilestoneEvent(milestone=Milestone.WOOD, round=0),
            MilestoneEvent(milestone=Milestone.IRON, round=13),
        ],
        frontier_reached=Milestone.IRON,
        terminated_reason="frontier_target",
        observations=[],
    )


def sample_episode_metrics() -> EpisodeMetrics:
    return EpisodeMetrics(
        episode_idx=0,
        seed="A",
        frontier_milestone=Milestone.IRON,
        frontier_value=0.20,
        team_reward=0.19,
        penalties={"deaths": 0.0, "invalid": 0.01, "idle": 0.0},
        invalid_rate=0.2,
        idle_fraction=0.0,
        deaths=0,
        n_rounds=14,
        won=False,
        milestone_timeline={"wood": 0, "iron": 13},
        agent_stats=[
            AgentStats(
                agent_id="agent_1",
                role=Role.MINER,
                actions_taken=14,
                invalid_actions=0,
                idle_rounds=0,
                items_gathered={"wood": 3, "iron_ore": 2},
                items_crafted={"wooden_pickaxe": 1, "stone_pickaxe": 1},
            )
        ],
        speed_bonus=0.0,
    )


# name -> (contract class, sample factory)
SAMPLES = {
    "Observation": (Observation, sample_observation),
    "Action": (Action, sample_action),
    "Message": (Message, sample_message),
    "BehaviorCard": (BehaviorCard, sample_behavior_card),
    "ExecutionMemory": (ExecutionMemory, sample_execution_memory),
    "EpisodeTrace": (EpisodeTrace, sample_episode_trace),
    "EpisodeMetrics": (EpisodeMetrics, sample_episode_metrics),
}

__all__ = ["SAMPLES"] + [f"sample_{k.lower()}" for k in SAMPLES]

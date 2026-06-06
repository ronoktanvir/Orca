"""The run loop (F5 / §8).

The thin, faithful version of the §8 training loop with a no-op Orca:

    config  = orca.choose_config(history)     # frozen cards (Phase 0)
    trace   = run_episode(seed, config)        # workers act over rounds, env enforces validity
    metrics = reward_computer(trace)           # §7 team frontier + stats
    orca.observe_outcome(config, metrics)      # bandit update (no-op in Phase 0)
    proposal = orca.coach(trace, metrics)      # coaching (no-op in Phase 0)
    if accept_gate(proposal): orca.commit(proposal)
    telemetry.log_episode(...)

Everything load-bearing is a ``@op`` so the Weave trace tree nests (§10).
"""

from __future__ import annotations

from typing import Optional

from agents.scripted import ShallowOracle
from config import OrcaSettings, load_config
from contracts import EpisodeMetrics, EpisodeTrace, MilestoneEvent
from contracts.enums import Milestone, Role
from env import StubEnv
from orca import DEFAULT_ROSTER, NoOpOrca, accept_gate
from orca.orca import OrcaConfig
from reward import reward_computer
from telemetry import Telemetry, init_telemetry, op


@op
def worker_turn(agent, obs):
    """One worker's turn: obs in, action out (§4.2). Inputs/outputs logged (§10)."""
    return agent.act(obs)


@op
def env_step(env: StubEnv, actions: dict):
    """One synchronous round of the environment (§3.6)."""
    return env.step(actions)


@op
def run_episode(
    env: StubEnv,
    agents: list,
    orca_config: OrcaConfig,
    *,
    episode_idx: int,
    telemetry: Telemetry,
    settings: OrcaSettings,
) -> tuple[EpisodeTrace, EpisodeMetrics]:
    """Run one full episode end-to-end; emit EpisodeTrace + EpisodeMetrics (§8)."""
    env.reset()
    obs_snapshots: list[dict] = []

    while not env.done:
        actions = {}
        for agent in agents:
            obs = env.observe(agent.agent_id)
            obs_snapshots.append(obs.model_dump(mode="json", by_alias=True))
            action = worker_turn(agent, obs)
            actions[agent.agent_id] = action
            telemetry.log_event(
                "worker_turn",
                {
                    "round": env.round_idx,
                    "agent": agent.agent_id,
                    "action": action.name.value,
                    "args": action.args,
                },
            )
        env_step(env, actions)

    trace = EpisodeTrace(
        episode_idx=episode_idx,
        seed=env.seed,
        n_rounds=env.round_idx,
        agent_ids=env.agent_ids,
        config={
            "arms": orca_config.arms,
            "roster": [(aid, role.value) for aid, role in orca_config.roster],
        },
        behavior_cards=list(orca_config.behavior_cards.values()),
        action_records=list(env.all_records),
        messages=list(env.all_messages),
        milestone_timeline=list(env.milestone_timeline),
        frontier_reached=env.frontier,
        terminated_reason=env.terminated_reason,
        observations=obs_snapshots,
    )

    metrics = op(reward_computer)(
        trace,
        agent_roles=orca_config.roles(),
        weights=settings.reward.weights,
    )
    return trace, metrics


def _parse_milestone(value: Optional[str]) -> Optional[Milestone]:
    if not value:
        return None
    return Milestone(value)


def run(
    settings: Optional[OrcaSettings] = None,
    *,
    config_path: Optional[str] = None,
    telemetry: Optional[Telemetry] = None,
) -> list[tuple[EpisodeTrace, EpisodeMetrics]]:
    """Drive ``n_episodes`` episodes; return their (trace, metrics) pairs (§8)."""
    settings = settings or load_config(config_path)
    telemetry = telemetry or init_telemetry(
        mode=settings.telemetry.mode,
        project=settings.telemetry.project,
        run_dir=settings.telemetry.run_dir,
    )

    if settings.run.single_agent_oracle:
        roster: list[tuple[str, Role]] = [("agent_1", Role.MINER)]
    else:
        roster = list(DEFAULT_ROSTER)
    # Phase 0 placeholder: the scripted oracle stands in for every worker.
    agents = [ShallowOracle(aid) for aid, _role in roster]

    orca = NoOpOrca(roster)
    stop_at = _parse_milestone(settings.run.stop_at_milestone)
    train_seeds = settings.seeds.train or [settings.run.seed]

    history: list = []
    results: list[tuple[EpisodeTrace, EpisodeMetrics]] = []

    for ep in range(settings.run.n_episodes):
        seed = settings.run.seed if settings.run.n_episodes == 1 else train_seeds[ep % len(train_seeds)]
        config = op(orca.choose_config)(history)
        env = StubEnv(
            seed=seed,
            episode_idx=ep,
            agents=config.roster,
            t_max=settings.run.t_max,
            day_length=settings.run.day_length,
            message_window=settings.run.message_window,
            stop_at_milestone=stop_at,
            behavior_cards=config.behavior_cards,
        )
        trace, metrics = run_episode(
            env, agents, config, episode_idx=ep, telemetry=telemetry, settings=settings
        )

        orca.observe_outcome(config, metrics)  # bandit update (no-op)
        proposal = op(orca.coach)(trace, metrics)  # coaching (no-op)
        if op(accept_gate)(proposal):  # gate (no-op keep)
            orca.commit(proposal)

        telemetry.log_episode(trace, metrics)
        history.append((seed, config, metrics))
        results.append((trace, metrics))

    return results


__all__ = ["run", "run_episode", "worker_turn", "env_step"]

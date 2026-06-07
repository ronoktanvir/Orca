"""The run loop (F5 / §8) — Stream 3 owns the Orca integration.

The faithful §8 training loop, with the real Architecture-C2 manager wired in:

    config   = orca.choose_config(history)     # bandit picks delegation arms (§6.3)
    trace    = run_episode(seed, config)        # workers act; env enforces validity
    metrics  = reward_computer(trace)           # §7 objective team frontier + stats
    metrics  = orca.objective_scores(metrics)   # advisory dials, never the reward (§7.3)
    orca.observe_outcome(config, metrics)       # bandit update, once/episode (§6.3)
    if phase >= 1:                              # phasing gates Orca's authority (§6.6)
        proposal = orca.coach(trace, metrics)   # verbal credit assignment (§6.4)
        gate.evaluate(orca, proposal, eval_fn)  # accept iff non-regressing (§6.5)
    telemetry.log_episode(...)

The offline fallback is preserved (Green-main law): with ``single_agent_oracle``
(the default) the loop uses the scripted oracle + :class:`NoOpOrca`, so ``python
run.py`` and ``pytest`` run with no LLM and no network. The full team path uses
the real :class:`Orca` and the same scripted oracle as a stand-in until Stream 2's
``LLMWorker`` is swapped in via ``worker_factory``.
"""

from __future__ import annotations

from statistics import median
from typing import Callable, Optional

from agents.scripted import ShallowOracle
from config import OrcaSettings, load_config
from contracts import EpisodeMetrics, EpisodeTrace
from contracts.enums import Milestone, Role
from env import StubEnv
from orca import DEFAULT_ROSTER, AcceptGate, NoOpOrca, Orca
from orca.orca import OrcaConfig
from reward import reward_computer
from telemetry import Telemetry, init_telemetry, op
from train.phases import Phase, current_phase

# Train-pool episodes the accept-gate re-runs to score a proposal (§6.5). Small so
# coaching stays cheap; held-out seeds are NEVER used here (anti-leakage, §9).
GATE_BATCH = 2
GATE_EPSILON = 0.02


@op
def worker_turn(agent, obs):
    """One worker's turn: obs in, action out (§4.2). Inputs/outputs logged (§10)."""
    return agent.act(obs)


@op
def env_step(env: StubEnv, actions: dict):
    """One synchronous round of the environment (§3.6)."""
    return env.step(actions)


def _round_actions(
    agents: list, env: StubEnv, obs_snapshots: list[dict], telemetry: Telemetry, concurrency: int
) -> dict:
    """Collect one round's actions. Observe is sequential (reads world state);
    the worker LLM calls run concurrently across agents when ``concurrency > 1``.

    The actions/snapshots/logs are assembled in fixed agent order regardless of
    thread completion, so results stay deterministic. Note: under threading the
    per-call ``@op`` Weave spans don't nest under the round (a known thread-context
    limitation); the default ``concurrency=1`` keeps the trace tree intact."""
    obs_by = []
    for agent in agents:
        obs = env.observe(agent.agent_id)
        obs_snapshots.append(obs.model_dump(mode="json", by_alias=True))
        obs_by.append((agent, obs))

    if concurrency and concurrency > 1 and len(obs_by) > 1:
        from concurrent.futures import ThreadPoolExecutor

        # Call agent.act directly (not the @op-wrapped worker_turn): Weave's op
        # context-vars don't cross threads, so wrapping here could raise in weave
        # mode and the spans wouldn't nest anyway. The sequential path keeps @op.
        with ThreadPoolExecutor(max_workers=min(concurrency, len(obs_by))) as ex:
            results = list(ex.map(lambda ao: ao[0].act(ao[1]), obs_by))
    else:
        results = [worker_turn(agent, obs) for agent, obs in obs_by]

    actions = {}
    for (agent, _obs), action in zip(obs_by, results):
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
    return actions


@op
def run_episode(
    env: StubEnv,
    agents: list,
    orca_config: OrcaConfig,
    *,
    episode_idx: int,
    telemetry: Telemetry,
    settings: OrcaSettings,
    baseline_steps: Optional[int] = None,
) -> tuple[EpisodeTrace, EpisodeMetrics]:
    """Run one full episode end-to-end; emit EpisodeTrace + EpisodeMetrics (§8)."""
    env.reset()
    obs_snapshots: list[dict] = []
    concurrency = getattr(settings.run, "worker_concurrency", 1)

    while not env.done:
        actions = _round_actions(agents, env, obs_snapshots, telemetry, concurrency)
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
        baseline_steps=baseline_steps,
    )
    return trace, metrics


def _parse_milestone(value: Optional[str]) -> Optional[Milestone]:
    if not value:
        return None
    return Milestone(value)


# --------------------------------------------------------------------------- #
# Building blocks shared by the loop and the eval harness (O7).
# --------------------------------------------------------------------------- #
def build_agents(
    roster: list[tuple[str, Role]],
    *,
    llm=None,
    worker_factory: Optional[Callable] = None,
) -> list:
    """Construct the worker objects for a roster.

    Default: the scripted :class:`ShallowOracle` (offline, role-independent).
    ``worker_factory(agent_id, role, llm)`` is the Stream 2 seam to swap in the
    real ``LLMWorker`` without the loop changing.
    """
    if worker_factory is not None:
        return [worker_factory(aid, role, llm) for aid, role in roster]
    return [ShallowOracle(aid) for aid, _role in roster]


def make_env(
    seed: str, config: OrcaConfig, settings: OrcaSettings, stop_at: Optional[Milestone]
) -> StubEnv:
    return StubEnv(
        seed=seed,
        episode_idx=0,
        agents=config.roster,
        t_max=settings.run.t_max,
        day_length=settings.run.day_length,
        message_window=settings.run.message_window,
        stop_at_milestone=stop_at,
        behavior_cards=config.behavior_cards,
    )


def play_episode(
    orca,
    seed: str,
    settings: OrcaSettings,
    *,
    episode_idx: int,
    telemetry: Telemetry,
    stop_at: Optional[Milestone] = None,
    llm=None,
    worker_factory: Optional[Callable] = None,
    greedy: bool = False,
    baseline_steps: Optional[int] = None,
) -> tuple[OrcaConfig, EpisodeTrace, EpisodeMetrics]:
    """Choose a config, run one episode, and fill the advisory dials (§7.3).

    ``greedy`` (Orca only) selects the best learned arms with no exploration — for
    held-out eval and gate batches. Works for both ``Orca`` and ``NoOpOrca``.
    """
    try:
        config = orca.choose_config(None, greedy=greedy)
    except TypeError:
        config = orca.choose_config(None)
    agents = build_agents(config.roster, llm=llm, worker_factory=worker_factory)
    env = make_env(seed, config, settings, stop_at)
    trace, metrics = run_episode(
        env,
        agents,
        config,
        episode_idx=episode_idx,
        telemetry=telemetry,
        settings=settings,
        baseline_steps=baseline_steps,
    )
    if isinstance(orca, Orca):
        metrics = orca.objective_scores(metrics)
    return config, trace, metrics


def _gate_eval_batch(
    orca: Orca,
    settings: OrcaSettings,
    seeds: list[str],
    *,
    stop_at: Optional[Milestone],
    telemetry: Telemetry,
    llm=None,
    worker_factory: Optional[Callable] = None,
) -> list[EpisodeMetrics]:
    """Re-run a small train-pool batch with greedy arms (no bandit update) (§6.5)."""
    out: list[EpisodeMetrics] = []
    for s in seeds:
        _cfg, _trace, metrics = play_episode(
            orca,
            s,
            settings,
            episode_idx=0,
            telemetry=telemetry,
            stop_at=stop_at,
            llm=llm,
            worker_factory=worker_factory,
            greedy=True,
        )
        out.append(metrics)
    return out


# --------------------------------------------------------------------------- #
def run(
    settings: Optional[OrcaSettings] = None,
    *,
    config_path: Optional[str] = None,
    telemetry: Optional[Telemetry] = None,
    orca=None,
    llm=None,
    worker_factory: Optional[Callable] = None,
) -> list[tuple[EpisodeTrace, EpisodeMetrics]]:
    """Drive ``n_episodes`` episodes; return their (trace, metrics) pairs (§8).

    Default (``single_agent_oracle``): the offline smoke — scripted oracle +
    :class:`NoOpOrca`. Full team: the real :class:`Orca` with bandit + (phased)
    coach + accept-gate and rotated train seeds.
    """
    settings = settings or load_config(config_path)
    telemetry = telemetry or init_telemetry(
        mode=settings.telemetry.mode,
        entity=settings.telemetry.entity,
        project=settings.telemetry.project,
        run_dir=settings.telemetry.run_dir,
    )
    stop_at = _parse_milestone(settings.run.stop_at_milestone)

    if settings.run.single_agent_oracle:
        roster: list[tuple[str, Role]] = [("agent_1", Role.MINER)]
        orca = orca or NoOpOrca(roster)
    else:
        roster = list(DEFAULT_ROSTER)
        orca = orca or Orca(
            roster,
            llm=llm,
            epsilon=settings.bandit.epsilon,
            seed=0,
            telemetry=telemetry,
        )
    real = isinstance(orca, Orca)

    train_seeds = settings.seeds.train or [settings.run.seed]
    gate_seeds = train_seeds[:GATE_BATCH]
    phase0_length = settings.phases.phase0_length

    first_win_seen = False
    win_rounds: list[int] = []
    baseline_steps: Optional[int] = None
    gate: Optional[AcceptGate] = None

    history: list = []
    results: list[tuple[EpisodeTrace, EpisodeMetrics]] = []

    for ep in range(settings.run.n_episodes):
        seed = (
            settings.run.seed
            if settings.run.n_episodes == 1
            else train_seeds[ep % len(train_seeds)]
        )
        phase = current_phase(ep, phase0_length, first_win_seen)
        if real:
            orca.enable_coach = phase >= Phase.PHASE_1

        config, trace, metrics = play_episode(
            orca,
            seed,
            settings,
            episode_idx=ep,
            telemetry=telemetry,
            stop_at=stop_at,
            llm=llm,
            worker_factory=worker_factory,
            baseline_steps=baseline_steps if phase >= Phase.PHASE_2 else None,
        )

        # Phase 2 (§6.6/§7.4): activate the speed-reward baseline only after a win.
        if metrics.won:
            win_rounds.append(metrics.n_rounds)
            if not first_win_seen:
                first_win_seen = True
                baseline_steps = int(median(win_rounds))

        orca.observe_outcome(config, metrics)  # bandit update (no-op for NoOpOrca)

        if real and orca.enable_coach:
            if gate is None:  # bar to beat = what bandit-only achieved in Phase 0
                prior = [m.team_reward for _s, _c, m in history]
                base = sum(prior) / len(prior) if prior else 0.0
                gate = AcceptGate(epsilon=GATE_EPSILON, baseline=base)
            proposal = op(orca.coach)(trace, metrics)
            gate.evaluate(
                orca,
                proposal,
                lambda: _gate_eval_batch(
                    orca,
                    settings,
                    gate_seeds,
                    stop_at=stop_at,
                    telemetry=telemetry,
                    llm=llm,
                    worker_factory=worker_factory,
                ),
                telemetry=telemetry,
            )

        telemetry.log_episode(trace, metrics)
        history.append((seed, config, metrics))
        results.append((trace, metrics))

    return results


__all__ = [
    "run",
    "run_episode",
    "play_episode",
    "build_agents",
    "make_env",
    "worker_turn",
    "env_step",
]

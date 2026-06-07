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
the real :class:`Orca`; after Stream 2, its default worker implementation is
``LLMWorker`` when a worker LLM is provided, while the explicit ``worker_factory``
seam remains available for tests, RealRunner swaps, and offline mocks.
"""

from __future__ import annotations

from statistics import median
from typing import Callable, Optional

from agents.memory import looks_seed_specific, sanitize_action_args, scrub_seed_specific
from agents.scripted import ShallowOracle
from agents.worker import LLMWorker
from bus import CommBus
from bus.messages import normalize_recipient
from config import OrcaSettings, load_config
from contracts import (
    EpisodeMetrics,
    EpisodeTrace,
    ExecutionMemory,
    Message,
    Observation,
    ReasoningRecord,
)
from contracts.enums import Milestone, Role
from env import StubEnv
from llm import build_llm
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


def _merge_bus_messages(
    obs: Observation, bus: CommBus, agent_id: str, window: int
) -> Observation:
    """Fold bus deliveries for ``agent_id`` into ``obs.recent_messages`` (§5.2)."""
    extra = bus.recent_for(agent_id)
    if not extra:
        return obs
    combined = (list(obs.recent_messages) + list(extra))[-window:]
    return obs.model_copy(update={"recent_messages": combined})


def _effective_concurrency(settings: OrcaSettings, n_agents: int) -> int:
    """Resolve this round's worker-call concurrency (§5 'async parallel').

    An explicit ``run.worker_concurrency > 0`` wins. Otherwise AUTO: one worker
    call per agent — so the single-agent oracle (``n_agents == 1``) stays
    sequential/Weave-safe and the full 4-agent team runs its calls concurrently.
    """
    c = getattr(settings.run, "worker_concurrency", 0) or 0
    if c > 0:
        return c
    return max(1, n_agents)


def _act_async(obs_by: list, concurrency: int) -> list:
    """Run each worker's sync ``act`` concurrently via ``asyncio.to_thread`` and
    return results in roster order (§4.2/§5).

    ``asyncio.gather`` preserves input order, so action assembly stays
    deterministic regardless of which worker finishes first. A semaphore bounds
    the in-flight calls to ``concurrency``."""
    import asyncio

    async def _run() -> list:
        sem = asyncio.Semaphore(max(1, concurrency))

        async def _one(agent, obs):
            async with sem:
                return await asyncio.to_thread(agent.act, obs)

        return await asyncio.gather(*[_one(a, o) for a, o in obs_by])

    return asyncio.run(_run())


def _act_round(obs_by: list, concurrency: int) -> list:
    """Collect one round's actions, concurrently when ``concurrency > 1``.

    Sequential path keeps the ``@op``-wrapped ``worker_turn`` so Weave spans nest
    (§10). The concurrent path calls ``agent.act`` directly off the main thread
    (Weave op context-vars don't cross threads). We detect an already-running event
    loop *up front* and use a thread pool then — rather than catching a RuntimeError
    from ``asyncio.run``, which would also swallow a RuntimeError raised inside a
    worker's ``act`` and re-run every worker (double side effects)."""
    if concurrency > 1 and len(obs_by) > 1:
        import asyncio

        try:
            asyncio.get_running_loop()
            in_loop = True
        except RuntimeError:
            in_loop = False
        if in_loop:  # asyncio.run would raise here -> thread-pool bridge instead
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=min(concurrency, len(obs_by))) as ex:
                return list(ex.map(lambda ao: ao[0].act(ao[1]), obs_by))
        return _act_async(obs_by, concurrency)
    return [worker_turn(agent, obs) for agent, obs in obs_by]


def _sanitize_bus_message(msg: Message) -> Optional[Message]:
    """Recipient-validate + content-scrub a message before it rides the bus (§3.2).

    Defense in depth for ANY message reaching the bus — worker-emitted drafts AND
    env-emitted ``report`` / ``request_help`` messages — so a custom worker_factory
    worker cannot leak through a raw pending message. The recipient is normalized
    (``team`` / ``orca`` / ``agent_<n>`` only) and coordinate-like content scrubbed.
    Returns ``None`` if nothing is left to say after scrubbing (the message is then
    dropped)."""
    content = msg.content
    if looks_seed_specific(content):
        content = scrub_seed_specific(content)
    if not content.strip():
        return None
    to = normalize_recipient(msg.to)
    if to == msg.to and content == msg.content:
        return msg
    return msg.model_copy(update={"to": to, "content": content})


def _drain_env_messages(env: StubEnv, bus: CommBus, telemetry: Telemetry) -> None:
    """Move this round's env-created messages onto the one authoritative bus for
    t+1 delivery, clearing the env's internal queue so it never *also* delivers
    them (single bus path; fixes the report/request_help t+2 delay, §5.2)."""
    for msg in env.drain_posted():
        safe = _sanitize_bus_message(msg)
        if safe is None:
            continue
        bus.post(safe)
        telemetry.log_event("message", safe.model_dump(by_alias=True))


def _round_actions(
    agents: list,
    env: StubEnv,
    obs_snapshots: list[dict],
    telemetry: Telemetry,
    concurrency: int,
    *,
    bus: Optional[CommBus] = None,
    reasoning_log: Optional[list] = None,
) -> dict:
    """Collect one round's actions.

    Observation is sequential (reads world state); worker calls can run
    concurrently across agents when ``concurrency > 1``. If a comm bus is present,
    messages posted last round are delivered before observation, and this round's
    worker-emitted (draft) messages are posted for t+1 delivery. Action-level
    ``report`` / ``request_help`` messages are drained onto the same bus after the
    env step (see :func:`_drain_env_messages`) — one authoritative path.
    """
    if bus is not None:
        bus.tick()

    obs_by = []
    for agent in agents:
        obs = env.observe(agent.agent_id)
        if bus is not None:
            obs = _merge_bus_messages(obs, bus, agent.agent_id, env.message_window)
        obs_snapshots.append(obs.model_dump(mode="json", by_alias=True))
        obs_by.append((agent, obs))

    results = _act_round(obs_by, concurrency)

    actions = {}
    for (agent, _obs), action in zip(obs_by, results):
        # Enforce the leak invariant at the env boundary for EVERY worker (not just
        # LLMWorker): a custom worker_factory worker could emit raw, leaky args that
        # would otherwise land in the trace's ActionRecords. Idempotent for already-
        # sanitized LLMWorker actions (§3.2).
        action = sanitize_action_args(action)
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
        # Thread the worker LLM's own reasoning onto the trace so Orca's coach can
        # read *how* it reasoned (§6.4) — scrubbed at this boundary for EVERY worker
        # (the leak wall applies to reasoning as much as to messages/args, §3.2).
        if reasoning_log is not None:
            raw_reason = (getattr(agent, "last_reasoning", "") or "").strip()
            if raw_reason:
                safe = (
                    scrub_seed_specific(raw_reason)
                    if looks_seed_specific(raw_reason)
                    else raw_reason
                ).strip()
                if safe:
                    reasoning_log.append(
                        ReasoningRecord(round=_obs.round, agent_id=agent.agent_id, text=safe)
                    )

    if bus is not None:
        for agent, _obs in obs_by:
            for msg in getattr(agent, "pending_messages", []) or []:
                # Sanitize at the bus boundary regardless of source: a custom
                # worker's raw pending_messages must not leak (recipient/content)
                # into the trace or, via team-addressing, into observations (§3.2).
                safe = _sanitize_bus_message(msg)
                if safe is None:
                    continue
                bus.post(safe)
                telemetry.log_event("message", safe.model_dump(by_alias=True))

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
    bus: Optional[CommBus] = None,
) -> tuple[EpisodeTrace, EpisodeMetrics]:
    """Run one full episode end-to-end; emit EpisodeTrace + EpisodeMetrics (§8)."""
    env.reset()
    obs_snapshots: list[dict] = []
    reasoning_log: list[ReasoningRecord] = []
    concurrency = _effective_concurrency(settings, len(agents))

    while not env.done:
        actions = _round_actions(
            agents,
            env,
            obs_snapshots,
            telemetry,
            concurrency,
            bus=bus,
            reasoning_log=reasoning_log,
        )
        env_step(env, actions)
        if bus is not None:
            # Drain the env's report/request_help messages onto the bus right away
            # so the bus is the single authoritative delivery path (§5.2).
            _drain_env_messages(env, bus, telemetry)

    # With a bus, its verbatim log is the single source of trace messages (worker
    # drafts + drained env messages) — no duplicates. Offline (no bus) the env's
    # own record stands. ``env.all_messages`` is preserved for compatibility.
    trace_messages = list(bus.log) if bus is not None else list(env.all_messages)
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
        messages=trace_messages,
        milestone_timeline=list(env.milestone_timeline),
        frontier_reached=env.frontier,
        terminated_reason=env.terminated_reason,
        observations=obs_snapshots,
        reasoning_log=reasoning_log,
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


def _episode_digest(trace: EpisodeTrace, metrics: EpisodeMetrics) -> str:
    """A compact, coordinate-free episode summary for memory writes (§4.5)."""
    lines = [
        f"frontier={metrics.frontier_milestone.value} reward={metrics.team_reward:.3f} "
        f"rounds={metrics.n_rounds} reason={trace.terminated_reason}"
    ]
    for st in metrics.agent_stats:
        lines.append(
            f"{st.agent_id}({st.role.value}): actions={st.actions_taken} "
            f"invalid={st.invalid_actions} idle={st.idle_rounds} "
            f"gathered={st.items_gathered} crafted={st.items_crafted}"
        )
    return "\n".join(lines)


def _should_write_memory(accepted: bool, proposal) -> bool:
    """Whether an episode-end memory write should run (§4.5/§6.5).

    Memory is written ONLY when a NON-EMPTY coach proposal was accepted after a
    real gate eval. An empty proposal (no card edits) is only *trivially* accepted
    by the gate — it short-circuits and skips the non-regression batch — so its
    scores must NOT drive an ungated memory write (which could persist a negative
    weaken/remove with the safety check bypassed). ``None`` (Phase 0, no coach) and
    rejected proposals also write nothing."""
    return bool(accepted) and proposal is not None and not proposal.is_empty()


def _coach_signals(proposal) -> dict[str, float]:
    """Per-agent ``learning_signal`` from an ACCEPTED coach proposal (§6.4/§4.5).

    These are the dials the verbal coach set (``proposal.scores[aid]
    ["learning_signal"]``), NOT the stale objective default on
    ``metrics.agent_stats`` — so a negative coach signal can actually reach the
    memory write and weaken/remove a heuristic."""
    out: dict[str, float] = {}
    for aid, sc in (getattr(proposal, "scores", None) or {}).items():
        try:
            out[aid] = float(sc.get("learning_signal", 0.0))
        except (TypeError, ValueError, AttributeError):
            out[aid] = 0.0
    return out


def _update_execution_memories(
    agents: list,
    memories: dict[str, ExecutionMemory],
    trace: EpisodeTrace,
    metrics: EpisodeMetrics,
    learning_signals: dict[str, float],
) -> None:
    """Persist agent-owned memory writes after an accepted update (§4.5, §6.5).

    Each agent's edit magnitude is driven by the accepted coach proposal's
    ``learning_signal`` (``learning_signals``); agents the coach did not score get
    ``0.0`` (a no-op — no LLM call, no change). ``> 0`` adds/strengthens, ``~0``
    leaves memory alone, ``< 0`` weakens/removes the flagged rule."""
    if not memories:
        return
    digest = _episode_digest(trace, metrics)
    for agent in agents:
        update = getattr(agent, "end_episode_update", None)
        if update is None:
            continue
        ls = float(learning_signals.get(agent.agent_id, 0.0))
        update(digest, ls)
        memory = getattr(agent, "memory", None)
        if isinstance(memory, ExecutionMemory):
            memories[agent.agent_id] = memory


# --------------------------------------------------------------------------- #
# Building blocks shared by the loop and the eval harness (O7).
# --------------------------------------------------------------------------- #
def build_agents(
    roster: list[tuple[str, Role]],
    *,
    llm=None,
    worker_factory: Optional[Callable] = None,
    behavior_cards: Optional[dict] = None,
    memories: Optional[dict[str, ExecutionMemory]] = None,
    telemetry: Optional[Telemetry] = None,
) -> list:
    """Construct worker objects for a roster.

    Default with no ``llm``: the scripted :class:`ShallowOracle` (offline).
    Default with ``llm``: Stream 2's :class:`LLMWorker`.
    ``worker_factory(agent_id, role, llm)`` remains the explicit seam for tests,
    RealRunner swaps, and custom worker implementations.
    """
    if worker_factory is not None:
        return [worker_factory(aid, role, llm) for aid, role in roster]
    if llm is not None:
        behavior_cards = behavior_cards or {}
        memories = memories or {}
        logger = telemetry.log_event if telemetry is not None else None
        return [
            LLMWorker(
                aid,
                llm,
                behavior_cards.get(aid),
                memories.get(aid, ExecutionMemory(agent_id=aid)),
                logger=logger,
                role=role,  # preserve the roster role when no card is supplied (§4.1)
            )
            for aid, role in roster
        ]
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
    memories: Optional[dict[str, ExecutionMemory]] = None,
    agent_sink: Optional[list] = None,
    enable_bus: bool = False,
) -> tuple[OrcaConfig, EpisodeTrace, EpisodeMetrics]:
    """Choose a config, run one episode, and fill advisory dials (§7.3)."""
    try:
        config = orca.choose_config(None, greedy=greedy)
    except TypeError:
        config = orca.choose_config(None)
    agents = build_agents(
        config.roster,
        llm=llm,
        worker_factory=worker_factory,
        behavior_cards=config.behavior_cards,
        memories=memories,
        telemetry=telemetry,
    )
    if agent_sink is not None:
        agent_sink[:] = agents
    env = make_env(seed, config, settings, stop_at)
    bus = CommBus(window=settings.run.message_window) if enable_bus else None
    trace, metrics = run_episode(
        env,
        agents,
        config,
        episode_idx=episode_idx,
        telemetry=telemetry,
        settings=settings,
        baseline_steps=baseline_steps,
        bus=bus,
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
            enable_bus=llm is not None,
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
    coach + accept-gate and rotated train seeds. When no explicit
    ``worker_factory`` is supplied, the full team uses Stream 2's ``LLMWorker``.
    """
    settings = settings or load_config(config_path)
    telemetry = telemetry or init_telemetry(
        mode=settings.telemetry.mode,
        entity=settings.telemetry.entity,
        project=settings.telemetry.project,
        run_dir=settings.telemetry.run_dir,
    )
    stop_at = _parse_milestone(settings.run.stop_at_milestone)

    use_oracle = settings.run.single_agent_oracle
    worker_llm = llm
    memories: dict[str, ExecutionMemory] = {}
    if use_oracle:
        roster: list[tuple[str, Role]] = [("agent_1", Role.MINER)]
        orca = orca or NoOpOrca(roster)
    else:
        roster = list(DEFAULT_ROSTER)
        if orca is None:
            # Orca's verbal coach runs on its OWN model (§6.4/§11): default GLM-5.1
            # via W&B Inference with the OpenAI fallback (or the explicit ``llm``).
            # The client is lazy, so building it is offline-safe — it only calls out
            # once the coach actually fires (Phase >= 1), and on any failure the coach
            # falls back to the deterministic heuristic path (coach._call_llm).
            orca_llm = llm if llm is not None else build_llm("orca", settings)
            orca = Orca(
                roster,
                llm=orca_llm,
                epsilon=settings.bandit.epsilon,
                seed=0,
                telemetry=telemetry,
            )
        if worker_factory is None:
            worker_llm = build_llm("worker", settings)
            memories = {aid: ExecutionMemory(agent_id=aid) for aid, _role in roster}
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

        episode_agents: list = []
        config, trace, metrics = play_episode(
            orca,
            seed,
            settings,
            episode_idx=ep,
            telemetry=telemetry,
            stop_at=stop_at,
            llm=worker_llm,
            worker_factory=worker_factory,
            baseline_steps=baseline_steps if phase >= Phase.PHASE_2 else None,
            memories=memories,
            agent_sink=episode_agents,
            enable_bus=not use_oracle,
        )

        # Phase 2 (§6.6/§7.4): activate the speed-reward baseline only after a win.
        if metrics.won:
            win_rounds.append(metrics.n_rounds)
            if not first_win_seen:
                first_win_seen = True
                baseline_steps = int(median(win_rounds))

        orca.observe_outcome(config, metrics)  # bandit update (no-op for NoOpOrca)

        memory_update_accepted = True
        proposal = None
        if real and orca.enable_coach:
            if gate is None:  # bar to beat = what bandit-only achieved in Phase 0
                prior = [m.team_reward for _s, _c, m in history]
                base = sum(prior) / len(prior) if prior else 0.0
                gate = AcceptGate(epsilon=GATE_EPSILON, baseline=base)
            proposal = op(orca.coach)(trace, metrics)
            decision = gate.evaluate(
                orca,
                proposal,
                lambda: _gate_eval_batch(
                    orca,
                    settings,
                    gate_seeds,
                    stop_at=stop_at,
                    telemetry=telemetry,
                    llm=worker_llm,
                    worker_factory=worker_factory,
                ),
                telemetry=telemetry,
            )
            memory_update_accepted = decision.accepted

        # Memory writes are driven ONLY by an accepted, NON-EMPTY coach proposal's
        # learning_signal (§4.5/§6.5): no coach (Phase 0), a rejected proposal, or
        # an empty (gate-bypassing) proposal means no write. The signal sign decides
        # add/strengthen vs weaken/remove.
        if _should_write_memory(memory_update_accepted, proposal):
            _update_execution_memories(
                episode_agents, memories, trace, metrics, _coach_signals(proposal)
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

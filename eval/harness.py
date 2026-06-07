"""Eval harness (§9) — Stream 3 (O7). The engine behind the headline plots.

Three conditions (static baseline / comms-no-Orca / Full C2), a **transfer test**
(train Full C2 on {A,T2,T3}, freeze, eval all three on held-out {B,C}), and the
ablations (memory / coaching / accept-gate on-off). The outcome source is a
*pluggable runner*:

  * :class:`SimRunner`  — the calibrated outcome model (offline, CI, free). Default.
  * :class:`RealRunner` — the real env + workers via ``train.loop`` (LLM-backed
    when available). Same record format, so the plots are identical.

Held-out seeds {B,C} are **never** used to train or to feed the bandit — only the
frozen policy is evaluated on them (anti-leakage, Law 4).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from random import Random
from typing import Callable, Optional

from config import OrcaSettings, load_config
from contracts import EpisodeMetrics, EpisodeTrace
from contracts.enums import Role
from env.rng import derive_seed
from orca import DEFAULT_ROSTER, AcceptGate, Orca
from orca.cards import make_default_card
from orca.orca import OrcaConfig, Proposal
from train.phases import Phase, current_phase

from . import outcome_model as om
from .outcome_model import BAD_DIRECTIVE, COMMS, FULL_C2, STATIC, SimParams
from .records import HELDOUT, TRAIN, EpisodeRecord

# Probability a coaching proposal is corrupted by simulated LLM noise (a bad
# directive). The accept-gate rolls these back; without it they're committed.
COACH_NOISE_P = 0.35


# --------------------------------------------------------------------------- #
# Condition specs — how each §9 condition configures Orca + the runner.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ConditionSpec:
    name: str  # plot label
    sim_condition: str  # STATIC | COMMS | FULL_C2 (drives the outcome model)
    bandit: bool
    coach: bool
    memory: bool
    gate: bool


STATIC_SPEC = ConditionSpec("static", STATIC, bandit=False, coach=False, memory=False, gate=False)
COMMS_SPEC = ConditionSpec("comms", COMMS, bandit=False, coach=False, memory=False, gate=False)
FULL_C2_SPEC = ConditionSpec("full_c2", FULL_C2, bandit=True, coach=True, memory=True, gate=True)
# Ablations (Full C2 with one knob removed).
ABL_NO_MEMORY = ConditionSpec("no_memory", FULL_C2, bandit=True, coach=True, memory=False, gate=True)
ABL_NO_COACH = ConditionSpec("no_coaching", FULL_C2, bandit=True, coach=False, memory=True, gate=True)
ABL_NO_GATE = ConditionSpec("no_gate", FULL_C2, bandit=True, coach=True, memory=True, gate=False)


# --------------------------------------------------------------------------- #
# Runners.
# --------------------------------------------------------------------------- #
class SimRunner:
    """Calibrated outcome model runner (offline headline). See ``outcome_model``."""

    def __init__(self, params: SimParams = om.DEFAULT_PARAMS) -> None:
        self.params = params

    def __call__(
        self,
        config: OrcaConfig,
        seed: str,
        *,
        condition: str,
        episode_idx: int,
        coaching_active: bool = False,
        memory: bool = True,
        gate_on: bool = True,
    ) -> tuple[EpisodeTrace, EpisodeMetrics]:
        return om.simulate_episode(
            config,
            seed,
            condition=condition,
            episode_idx=episode_idx,
            coaching_active=coaching_active,
            memory=memory,
            gate_on=gate_on,
            params=self.params,
        )


class RealRunner:
    """Real env + workers via ``train.loop`` (LLM-backed when provided).

    The seam for the live demo: identical record format to :class:`SimRunner`.
    The sim-only flags (condition/coaching/memory/gate) don't change the scripted
    oracle's behaviour — they bite once Stream 2's ``LLMWorker`` is wired in.
    """

    def __init__(self, settings: OrcaSettings, *, telemetry=None, llm=None, worker_factory=None) -> None:
        from telemetry import init_telemetry
        from train.loop import _parse_milestone

        self.settings = settings
        self.telemetry = telemetry or init_telemetry(mode="off")
        self.llm = llm
        self.worker_factory = worker_factory
        self.stop_at = _parse_milestone(settings.run.stop_at_milestone)

    def __call__(
        self,
        config: OrcaConfig,
        seed: str,
        *,
        condition: str,
        episode_idx: int,
        coaching_active: bool = False,
        memory: bool = True,
        gate_on: bool = True,
    ) -> tuple[EpisodeTrace, EpisodeMetrics]:
        from train.loop import build_agents, make_env, run_episode

        agents = build_agents(config.roster, llm=self.llm, worker_factory=self.worker_factory)
        env = make_env(seed, config, self.settings, self.stop_at)
        trace, metrics = run_episode(
            env, agents, config, episode_idx=episode_idx, telemetry=self.telemetry, settings=self.settings
        )
        # Fill the advisory dials objectively too, so the real path mirrors the sim.
        return trace, Orca.objective_scores(metrics)


Runner = Callable[..., "tuple[EpisodeTrace, EpisodeMetrics]"]


def make_orca(spec: ConditionSpec, settings: OrcaSettings, *, llm=None, seed: int = 0) -> Orca:
    """Build an Orca configured for ``spec`` over the default 4-agent roster."""
    return Orca(
        list(DEFAULT_ROSTER),
        llm=llm,
        epsilon=settings.bandit.epsilon,
        seed=seed,
        enable_bandit=spec.bandit,
        enable_coach=False,  # phased on during training
    )


# --------------------------------------------------------------------------- #
# Core run primitives.
# --------------------------------------------------------------------------- #
def eval_batch(
    orca: Orca,
    runner: Runner,
    seeds: list[str],
    spec: ConditionSpec,
    split: str,
    *,
    reps: int = 1,
    greedy: bool = True,
) -> list[EpisodeRecord]:
    """Run ``reps`` frozen episodes per seed; return records (no bandit update)."""
    records: list[EpisodeRecord] = []
    idx = 0
    for rep in range(reps):
        for s in seeds:
            config = orca.choose_config(greedy=greedy)
            _trace, metrics = runner(
                config,
                s,
                condition=spec.sim_condition,
                episode_idx=idx,
                coaching_active=False,
                memory=spec.memory,
                gate_on=spec.gate,
            )
            records.append(
                EpisodeRecord.from_metrics(metrics, condition=spec.name, split=split, arms=config.arms)
            )
            idx += 1
    return records


def _inject_coach_noise(proposal: Proposal, orca: Orca, rng: Random) -> Proposal:
    """Simulate a noisy LLM episode: *replace* the proposal with a bad-only edit.

    This is the noise the accept-gate exists to filter (§6.5). As a standalone bad
    proposal it has no redeeming fix, so with the gate on its eval frontier falls
    below the bar and it is rolled back; with the gate off it is committed and
    persistently degrades the cards (the gate ablation, §9). Replacing (not
    augmenting) is what keeps the gate's judgement clean — a good fix is never
    bundled with the noise.
    """
    if rng.random() >= COACH_NOISE_P:
        return proposal
    aid = orca.agent_ids[rng.randrange(len(orca.agent_ids))]
    base = orca._coached.get(aid) or make_default_card(aid, Role.MINER)
    bad_card = base.model_copy(
        update={"directives": list(base.directives) + [BAD_DIRECTIVE], "version": base.version + 1}
    )
    return Proposal(
        behavior_cards={aid: bad_card},
        notes="coach(noise)",
        rationale="(simulated noisy LLM edit — should be rejected by the gate)",
    )


@dataclass
class TrainResult:
    orca: Orca
    learning: list[EpisodeRecord] = field(default_factory=list)  # per training episode
    value_snapshots: list[dict[str, dict[str, float]]] = field(default_factory=list)
    train_seeds: list[str] = field(default_factory=list)
    gate: Optional[AcceptGate] = None


def train_full_c2(
    spec: ConditionSpec,
    settings: OrcaSettings,
    runner: Runner,
    train_seeds: list[str],
    n_episodes: int,
    *,
    llm=None,
    phase0_length: Optional[int] = None,
    gate_epsilon: float = 0.02,
    gate_batch: int = 2,
    bandit_seed: int = 0,
) -> TrainResult:
    """Train an Orca under ``spec`` on ``train_seeds`` (bandit + phased coach + gate).

    ``gate_batch`` is the number of train-pool episodes the accept-gate re-runs per
    coached episode — the dominant Phase-1 cost lever (see ``eval/cost_model.py``).
    """
    phase0_length = settings.phases.phase0_length if phase0_length is None else phase0_length
    orca = make_orca(spec, settings, llm=llm, seed=bandit_seed)
    gate_seeds = train_seeds[:max(1, gate_batch)]
    gate: Optional[AcceptGate] = None
    first_win = False
    history_reward: list[float] = []
    noise_rng = Random(derive_seed("coachnoise", spec.name, bandit_seed))
    res = TrainResult(orca=orca, train_seeds=list(train_seeds))

    for ep in range(n_episodes):
        seed = train_seeds[ep % len(train_seeds)]
        phase = current_phase(ep, phase0_length, first_win)
        orca.enable_coach = spec.coach and phase >= Phase.PHASE_1

        config = orca.choose_config()
        trace, metrics = runner(
            config,
            seed,
            condition=spec.sim_condition,
            episode_idx=ep,
            coaching_active=orca.enable_coach,
            memory=spec.memory,
            gate_on=spec.gate,
        )
        metrics = orca.objective_scores(metrics)
        if metrics.won and not first_win:
            first_win = True
        orca.observe_outcome(config, metrics)

        if orca.enable_coach:
            if gate is None:
                base = sum(history_reward) / len(history_reward) if history_reward else 0.0
                gate = AcceptGate(epsilon=gate_epsilon, baseline=base)
            proposal = orca.coach(trace, metrics)
            proposal = _inject_coach_noise(proposal, orca, noise_rng)  # simulated LLM noise
            if spec.gate:
                gate.evaluate(
                    orca,
                    proposal,
                    lambda: [
                        runner(
                            orca.choose_config(greedy=True),
                            gs,
                            condition=spec.sim_condition,
                            episode_idx=0,
                            coaching_active=False,
                            memory=spec.memory,
                            gate_on=spec.gate,
                        )[1]
                        for gs in gate_seeds
                    ],
                )
            else:
                orca.commit(proposal)  # no gate -> keep every (noisy) edit

        history_reward.append(metrics.team_reward)
        res.learning.append(
            EpisodeRecord.from_metrics(metrics, condition=spec.name, split=TRAIN, arms=config.arms)
        )
        res.value_snapshots.append(copy.deepcopy(orca.bandit.values()))

    res.gate = gate
    orca.freeze()  # frozen policy for held-out eval
    return res


# --------------------------------------------------------------------------- #
# Experiments.
# --------------------------------------------------------------------------- #
@dataclass
class TransferResult:
    records: list[EpisodeRecord]
    train_result: TrainResult


def run_transfer(
    settings: Optional[OrcaSettings] = None,
    *,
    runner: Optional[Runner] = None,
    n_train: int = 30,
    eval_reps: int = 6,
    gate_batch: int = 2,
    llm=None,
) -> TransferResult:
    """The money plot data: train Full C2 on train seeds, eval all 3 on B/C (§9)."""
    settings = settings or load_config()
    runner = runner or SimRunner()
    train_seeds = list(settings.seeds.train)
    heldout = list(settings.seeds.heldout)

    records: list[EpisodeRecord] = []
    tr = train_full_c2(FULL_C2_SPEC, settings, runner, train_seeds, n_train, llm=llm, gate_batch=gate_batch)
    records += eval_batch(tr.orca, runner, train_seeds, FULL_C2_SPEC, TRAIN, reps=eval_reps)
    records += eval_batch(tr.orca, runner, heldout, FULL_C2_SPEC, HELDOUT, reps=eval_reps)

    for spec in (STATIC_SPEC, COMMS_SPEC):
        orca = make_orca(spec, settings, llm=llm)
        orca.freeze()
        records += eval_batch(orca, runner, train_seeds, spec, TRAIN, reps=eval_reps)
        records += eval_batch(orca, runner, heldout, spec, HELDOUT, reps=eval_reps)

    return TransferResult(records=records, train_result=tr)


def run_ablations(
    settings: Optional[OrcaSettings] = None,
    *,
    runner: Optional[Runner] = None,
    n_train: int = 30,
    eval_reps: int = 6,
    gate_batch: int = 2,
    llm=None,
) -> list[EpisodeRecord]:
    """Full C2 vs. memory-off / coaching-off / gate-off, evaluated on held-out (§9)."""
    settings = settings or load_config()
    runner = runner or SimRunner()
    train_seeds = list(settings.seeds.train)
    heldout = list(settings.seeds.heldout)

    records: list[EpisodeRecord] = []
    for spec in (FULL_C2_SPEC, ABL_NO_MEMORY, ABL_NO_COACH, ABL_NO_GATE):
        tr = train_full_c2(spec, settings, runner, train_seeds, n_train, llm=llm, gate_batch=gate_batch)
        records += eval_batch(tr.orca, runner, heldout, spec, HELDOUT, reps=eval_reps)
    return records


def run_provider_ablation(
    variants: dict[str, OrcaSettings],
    *,
    worker_factory: Optional[Callable] = None,
    n_train: int = 30,
    eval_reps: int = 6,
    gate_batch: int = 2,
) -> dict[str, TransferResult]:
    """Model-swap ablation: run the transfer under each LLM config, tagged by label.

    ``variants`` maps a label (e.g. "gpt-5-mini" / "GLM-5.1") to an ``OrcaSettings``
    whose ``llm`` section selects that provider/model. Each variant runs through a
    :class:`RealRunner` with workers built from *that* settings, so the comparison
    reflects the real model swap.

    REAL-runner only: this is meaningful once Stream 2's ``LLMWorker`` is wired in
    via ``worker_factory`` — with the scripted oracle (``worker_factory=None``) the
    LLM config is ignored and every variant ties (a documented no-op offline).
    """
    from llm import build_llm

    out: dict[str, TransferResult] = {}
    for label, vs in variants.items():
        worker_llm = build_llm("worker", vs) if worker_factory is not None else None
        runner = RealRunner(vs, llm=worker_llm, worker_factory=worker_factory)
        out[label] = run_transfer(
            vs, runner=runner, n_train=n_train, eval_reps=eval_reps, gate_batch=gate_batch
        )
    return out


def run_learning_curve(
    settings: Optional[OrcaSettings] = None,
    *,
    runner: Optional[Runner] = None,
    n_train: int = 40,
    gate_batch: int = 2,
    llm=None,
) -> TrainResult:
    """Train Full C2 and return per-episode learning + bandit value snapshots (§9)."""
    settings = settings or load_config()
    runner = runner or SimRunner()
    return train_full_c2(
        FULL_C2_SPEC, settings, runner, list(settings.seeds.train), n_train, llm=llm, gate_batch=gate_batch
    )


__all__ = [
    "ConditionSpec",
    "STATIC_SPEC",
    "COMMS_SPEC",
    "FULL_C2_SPEC",
    "ABL_NO_MEMORY",
    "ABL_NO_COACH",
    "ABL_NO_GATE",
    "SimRunner",
    "RealRunner",
    "make_orca",
    "eval_batch",
    "train_full_c2",
    "TrainResult",
    "run_transfer",
    "run_ablations",
    "run_provider_ablation",
    "run_learning_curve",
    "TransferResult",
]

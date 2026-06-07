"""Calibrated outcome model (§9 testbed) — Stream 3 (O7).

**What this is, plainly:** a small, transparent, *seeded* generative model of
episode outcomes. The Phase-0 stub env + scripted oracle reach IRON
deterministically regardless of Orca's choices, and the real coupling
(behavior-card → worker behaviour → frontier) only exists once Stream 2's
``LLMWorker`` lands (which costs API calls and is not offline/CI-runnable). So
the headline learning/transfer/ablation plots need an outcome source that
*causally* depends on delegation and coaching — reproducibly, offline, for free.

**What it is NOT:** it never touches ``team_reward`` from Orca's own scores
(anti-circularity holds — the reward here is an objective frontier function of the
world + delegation, never Orca's opinion), and it is a stand-in, not a claim
about real LLM behaviour. The same eval harness runs against the *real* loop
(``eval.harness.RealRunner``) when an API key + LLM workers are present.

The model encodes two honest, separable effects the demo argues for:

  1. **Transferable delegation.** Each S1/S2/S3/S4 arm has a latent quality that
     is *shared across seeds* (the strategy) plus a *per-seed terrain* offset
     (the confound). So a bandit that learns the good arms on {A,T2,T3} is still
     good on held-out {B,C}: "strategy, not terrain."
  2. **Coaching that clears a bottleneck.** A roster's iron-miner repeatedly
     fails to mine iron (missing a stone pickaxe) until its card carries the
     corrective directive the coach writes — which lowers invalid-rate and lifts
     the frontier on subsequent (and held-out) episodes. This is the
     failure→fix→improve loop, made reproducible.

It emits a full ``EpisodeTrace`` + ``EpisodeMetrics`` (with synthetic but
plausible action records) so the unmodified Orca machinery — digest, scoring,
verbal coach, accept-gate, Weave scorers — operates on it exactly as on the real
env.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random

from contracts import (
    Action,
    ActionRecord,
    AgentStats,
    BehaviorCard,
    EpisodeMetrics,
    EpisodeTrace,
    MilestoneEvent,
)
from contracts.enums import ActionName, Milestone, Role
from env.rng import derive_seed
from orca.orca import OrcaConfig
from orca.situations import S1, S2, S3, S4
from reward.dag import MILESTONE_VALUE

# Conditions compared in §9.
STATIC = "static"  # fixed balanced roster, no Orca, no comms
COMMS = "comms"  # agents message, but no delegation/coaching/memory
FULL_C2 = "full_c2"  # Orca bandit + (phased) coaching + memory + gate


@dataclass(frozen=True)
class SimParams:
    """The latent calibration. Tunable; values chosen to give clean separation."""

    # S1 roster quality — explore-heavy reaches the deepest frontier (finds the
    # fortress/stronghold faster); this is the strategy the bandit should learn.
    s1_quality: dict[str, float] = field(
        default_factory=lambda: {
            "balanced": 0.46,
            "mining_heavy": 0.40,
            "explore_heavy": 0.62,
            "self_sufficient": 0.50,
        }
    )
    # Per-situation arm bonuses (smaller). two_pairs / gear_gated / regroup are best.
    policy_bonus: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            S2: {"gear_gated": 0.03, "iron_secured": 0.01, "immediate": -0.02},
            S3: {"solo": -0.02, "two_pairs": 0.05, "all_together": 0.0},
            S4: {"regroup_all": 0.02, "split_roles": 0.0},
        }
    )
    comms_bonus: float = 0.05  # comms-no-Orca beats pure static a little
    bottleneck_penalty: float = 0.15  # frontier lost while the iron-miner stays unfixed
    memory_bonus: float = 0.03
    seed_terrain: float = 0.07  # amplitude of the per-seed (shared) terrain offset
    noise: float = 0.025  # per-episode gaussian noise sd
    base_invalid: float = 0.30  # invalid-rate when the miner is unfixed
    fixed_invalid: float = 0.06  # invalid-rate once the miner's card is fixed
    bad_edit_penalty: float = 0.14  # frontier hit from an ungated bad coaching edit


DEFAULT_PARAMS = SimParams()

# Directive substrings the coach writes that "fix" the miner bottleneck.
_FIX_MARKERS = ("prerequisit", "stone_pickaxe", "stone pickaxe", "initiative")
# Substrings that mark a *bad* (noisy LLM) edit — the accept-gate should reject these.
_BAD_MARKERS = ("avoid mining", "wait for a teammate handoff", "stop gathering")
# A canonical bad directive the gate ablation injects (simulated LLM noise).
BAD_DIRECTIVE = "Avoid mining; wait for a teammate handoff instead."


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _seed_terrain_offset(seed: str, params: SimParams) -> float:
    """A fixed per-seed offset, identical across conditions/arms (the confound)."""
    r = Random(derive_seed("terrain", seed))
    return (r.random() * 2.0 - 1.0) * params.seed_terrain


def _nearest_milestone(value: float) -> Milestone:
    """Deepest ladder milestone whose value ≤ ``value`` (for display + scorers)."""
    best = Milestone.START
    for m, v in MILESTONE_VALUE.items():
        if v <= value + 1e-9 and MILESTONE_VALUE[m] >= MILESTONE_VALUE[best]:
            best = m
    return best


def _card_text(card: BehaviorCard) -> str:
    return (" ".join(card.directives) + " " + card.assignment).lower()


def _card_has_fix(card: BehaviorCard | None) -> bool:
    if card is None:
        return False
    return any(mk in _card_text(card) for mk in _FIX_MARKERS)


def _any_card_bad(cards: dict[str, BehaviorCard]) -> bool:
    """True if any card carries a bad (noisy) directive the gate should have caught."""
    return any(any(mk in _card_text(c) for mk in _BAD_MARKERS) for c in cards.values())


def _bottleneck_agent(roster: list[tuple[str, Role]]) -> str | None:
    """The agent that repeatedly fails to mine iron — a *fixed* id (the 2nd seat)
    so the coach's fix, committed by ``agent_id``, reliably applies at eval time
    regardless of which S1 arm (roster) the bandit later settles on."""
    if not roster:
        return None
    return roster[1][0] if len(roster) > 1 else roster[0][0]


def simulate_episode(
    config: OrcaConfig,
    seed: str,
    *,
    condition: str,
    episode_idx: int,
    coaching_active: bool = False,
    memory: bool = True,
    gate_on: bool = True,
    params: SimParams = DEFAULT_PARAMS,
) -> tuple[EpisodeTrace, EpisodeMetrics]:
    """Generate one synthetic episode for ``condition`` (see module docstring)."""
    rng = Random(
        derive_seed("outcome", seed, condition, episode_idx, tuple(sorted(config.arms.items())))
    )
    arms = config.arms
    roster = config.roster
    miner_id = _bottleneck_agent(roster)
    miner_card = config.behavior_cards.get(miner_id) if miner_id else None
    miner_fixed = _card_has_fix(miner_card)

    # --- frontier (objective) ------------------------------------------------ #
    # (1) delegation/strategy — transferable across seeds; baselines never learn it.
    if condition in (STATIC, COMMS):
        target = params.s1_quality["balanced"] + (params.comms_bonus if condition == COMMS else 0.0)
    else:  # FULL_C2: the learned arms drive the frontier
        target = params.s1_quality.get(arms.get(S1, "balanced"), params.s1_quality["balanced"])
        for sit in (S2, S3, S4):
            target += params.policy_bonus.get(sit, {}).get(arms.get(sit, ""), 0.0)
        if memory:
            target += params.memory_bonus
    # (2) coaching — only Full C2 ever writes the iron-miner's missing lesson. An
    #     unfixed miner bottlenecks the whole team (stuck without iron tooling).
    if not miner_fixed:
        target -= params.bottleneck_penalty
    # (3) a *bad* committed card (ungated LLM noise) drags the frontier down —
    #     this is what the accept-gate exists to prevent (the gate ablation).
    if _any_card_bad(config.behavior_cards):
        target -= params.bad_edit_penalty

    target = _clip(target + _seed_terrain_offset(seed, params) + rng.gauss(0.0, params.noise))
    milestone = _nearest_milestone(target)

    # --- invalid-rate: high until the miner's card is fixed ------------------- #
    if miner_fixed:
        inv_rate = _clip(params.fixed_invalid + rng.gauss(0.0, 0.02), 0.0, 1.0)
        miner_invalids = 0
    else:
        inv_rate = _clip(params.base_invalid + rng.gauss(0.0, 0.03), 0.0, 1.0)
        miner_invalids = 4  # enough for the digest/coach to flag (top_invalid ≥ 2)

    # --- fabricate a plausible trace + stats so the real machinery runs ------- #
    n_rounds = 30
    # Cooperation: comms-enabled conditions (comms / Full C2) actually message;
    # the static baseline does not. Feeds the §10 cooperation-events scorer.
    msgs_per_agent = 0 if condition == STATIC else 3
    handoffs_per_agent = 0 if condition == STATIC else (2 if condition == FULL_C2 else 1)
    records: list[ActionRecord] = []
    stats: list[AgentStats] = []
    total_actions = total_invalid = total_idle = 0
    for aid, role in roster:
        acts = n_rounds
        invalid = miner_invalids if aid == miner_id else 0
        idle = 1 if role == Role.SUPPORT else 0
        gathered = {"wood": 4} if role in (Role.MINER, Role.EXPLORER) else {}
        crafted = {"wooden_pickaxe": 1} if role == Role.MINER else {}
        # a few representative records (incl. repeated invalids for the miner)
        for k in range(invalid):
            records.append(
                ActionRecord(
                    round=k,
                    agent_id=aid,
                    action=Action(name=ActionName.GATHER, args={"resource": "iron_ore"}),
                    valid=False,
                    reason="need stone_pickaxe (have 0)",
                )
            )
        records.append(
            ActionRecord(
                round=n_rounds - 1,
                agent_id=aid,
                action=Action(name=ActionName.GATHER, args={"resource": "wood"}),
                valid=True,
                result={"gathered": gathered} if gathered else {},
            )
        )
        total_actions += acts
        total_invalid += invalid
        total_idle += idle
        stats.append(
            AgentStats(
                agent_id=aid,
                role=role,
                actions_taken=acts,
                invalid_actions=invalid,
                idle_rounds=idle,
                items_gathered=gathered,
                items_crafted=crafted,
                handoffs_given=handoffs_per_agent,
                messages_sent=msgs_per_agent,
            )
        )

    # milestone timeline up to the reached milestone
    ladder = [m for m in Milestone if MILESTONE_VALUE[m] <= MILESTONE_VALUE[milestone]]
    step = max(1, n_rounds // max(1, len(ladder)))
    timeline = [MilestoneEvent(milestone=m, round=min(n_rounds, i * step)) for i, m in enumerate(ladder)]

    trace = EpisodeTrace(
        episode_idx=episode_idx,
        seed=seed,
        n_rounds=n_rounds,
        agent_ids=[aid for aid, _ in roster],
        config={"arms": dict(arms), "roster": [(aid, r.value) for aid, r in roster], "condition": condition},
        behavior_cards=list(config.behavior_cards.values()),
        action_records=records,
        messages=[],
        milestone_timeline=timeline,
        frontier_reached=milestone,
        terminated_reason="win" if milestone == Milestone.DRAGON_DEFEATED else "frontier_target",
        observations=[],
    )

    # objective penalties (same shape as reward.py) on the continuous frontier
    idle_fraction = total_idle / total_actions if total_actions else 0.0
    penalties = {
        "deaths": 0.0,
        "invalid": 0.05 * inv_rate,
        "idle": 0.05 * idle_fraction,
    }
    team_reward = max(0.0, target - sum(penalties.values()))

    metrics = EpisodeMetrics(
        episode_idx=episode_idx,
        seed=seed,
        frontier_milestone=milestone,
        frontier_value=round(target, 4),
        team_reward=round(team_reward, 4),
        penalties=penalties,
        invalid_rate=round(inv_rate, 4),
        idle_fraction=round(idle_fraction, 4),
        deaths=0,
        n_rounds=n_rounds,
        won=milestone == Milestone.DRAGON_DEFEATED,
        milestone_timeline={ev.milestone.value: ev.round for ev in timeline},
        agent_stats=stats,
        speed_bonus=0.0,
    )
    return trace, metrics


__all__ = [
    "SimParams",
    "DEFAULT_PARAMS",
    "simulate_episode",
    "STATIC",
    "COMMS",
    "FULL_C2",
]

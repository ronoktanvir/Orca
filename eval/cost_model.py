#!/usr/bin/env python3
"""Token + $ budget model for a *real* LLM-worker eval run (§9 planning).

The offline calibrated outcome model is free; the real run (Stream 2's
``LLMWorker`` + a deep env via ``RealRunner``) is not. This estimates the token
volume and cost so we can size ``n_train`` / ``T_max`` against the ~$150 OpenAI +
$50 W&B budget *before* spending it.

The token *counts* are derived from the actual loop/harness control flow and are
the rigorous part. The per-token **prices are ILLUSTRATIVE placeholders** — verify
current numbers at the provider before trusting the dollar figures.

Two structural cost drivers this surfaces (both free offline, expensive live):
  1. The accept-gate re-runs ``gate_batch`` *full* episodes per Phase-1 coaching
     episode → multiplies worker calls during Phase 1 (only for specs that have
     BOTH coaching and the gate on; no-coach / no-gate ablations skip it).
  2. ``eval.run_eval`` as written trains Full C2 **8 times**: 3 inside
     ``make_all_plots`` (run_learning_curve + run_transfer + run_ablations' first
     spec) + ``run_transfer`` again + ``evaluate_conditions`` = 5 Full C2, plus 3
     ablation trainings. A real run must compute experiments ONCE and share — the
     ``campaign()`` model below assumes that deduplicated path (4 trainings total).

Run: ``python -m eval.cost_model``
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Prices:
    """Per-1M-token prices. ILLUSTRATIVE — confirm at the provider's pricing page."""

    worker_in: float = 0.25  # gpt-5-mini input  ($/1M tok) — PLACEHOLDER
    worker_out: float = 2.00  # gpt-5-mini output ($/1M tok) — PLACEHOLDER
    orca_in: float = 1.25  # gpt-5 input        ($/1M tok) — PLACEHOLDER
    orca_out: float = 10.00  # gpt-5 output       ($/1M tok) — PLACEHOLDER


@dataclass
class Latency:
    """Wall-clock assumptions. ILLUSTRATIVE — the Orca reasoning model is the big
    unknown (10–60s); double it and the training spine roughly doubles."""

    worker_s: float = 2.0  # one worker (gpt-5-mini) call
    orca_s: float = 20.0  # one Orca coach (gpt-5 reasoning) call
    async_workers: bool = True  # 4 workers concurrent within a round (loop scaffold)
    eval_concurrency: int = 20  # parallel episodes for the embarrassingly-parallel parts


@dataclass
class Assumptions:
    n_workers: int = 4
    rounds_per_episode: int = 14  # shallow=~14 (iron), deep can approach T_max
    # per-call token sizes (prompt incl. card+obs+menu; output is a small JSON)
    worker_in_tok: int = 1200
    worker_out_tok: int = 150
    orca_in_tok: int = 2500  # digest + cards
    orca_out_tok: int = 700
    # loop / harness structure
    n_train: int = 40
    phase0_length: int = 15
    gate_batch: int = 2  # eval episodes the gate re-runs per Phase-1 coach episode
    # eval
    n_train_seeds: int = 3
    n_heldout_seeds: int = 2
    eval_reps: int = 8  # matches run_eval.py's --reps default
    n_conditions: int = 3  # static / comms / full_c2
    prices: Prices = field(default_factory=Prices)


# The Full C2 ablations and which cost terms each actually incurs (mirrors the
# specs in eval/harness.py): (name, has_coach, has_gate). no_coaching has no coach
# call and therefore no gate re-runs; no_gate coaches but skips the gate batch.
_ABLATIONS: list[tuple[str, bool, bool]] = [
    ("no_memory", True, True),
    ("no_coaching", False, False),
    ("no_gate", True, False),
]


@dataclass
class CostLine:
    label: str
    worker_calls: int
    orca_calls: int
    worker_tokens_in: int
    worker_tokens_out: int
    orca_tokens_in: int
    orca_tokens_out: int

    @property
    def dollars(self) -> float:
        return 0.0  # filled by _price


def _worker_calls_training(a: Assumptions, *, coach: bool = True, gate: bool = True) -> tuple[int, int]:
    """(worker_calls, coach_calls) for one training run with the given knobs.

    ``phase1`` counts Phase-1 (coaching) episodes as ``n_train - phase0_length`` —
    a LOWER BOUND: per train/phases.py a first win flips straight to Phase 2 and
    can enable coaching during warmup, which (rarely) makes the true count higher.
    The gate batch only applies when the spec both coaches AND gates.
    """
    per_ep = a.rounds_per_episode * a.n_workers
    phase1 = max(0, a.n_train - a.phase0_length) if coach else 0
    train_calls = a.n_train * per_ep
    gate_calls = (phase1 * a.gate_batch * per_ep) if (coach and gate) else 0
    coach_calls = phase1  # one coach LLM call per Phase-1 episode (gated or not)
    return train_calls + gate_calls, coach_calls


def _worker_calls_eval(a: Assumptions, *, conditions: int, seeds: int) -> int:
    return conditions * seeds * a.eval_reps * a.rounds_per_episode * a.n_workers


def _line(a: Assumptions, label: str, worker_calls: int, coach_calls: int) -> CostLine:
    return CostLine(
        label=label,
        worker_calls=worker_calls,
        orca_calls=coach_calls,
        worker_tokens_in=worker_calls * a.worker_in_tok,
        worker_tokens_out=worker_calls * a.worker_out_tok,
        orca_tokens_in=coach_calls * a.orca_in_tok,
        orca_tokens_out=coach_calls * a.orca_out_tok,
    )


def _price(line: CostLine, p: Prices) -> float:
    return (
        line.worker_tokens_in / 1e6 * p.worker_in
        + line.worker_tokens_out / 1e6 * p.worker_out
        + line.orca_tokens_in / 1e6 * p.orca_in
        + line.orca_tokens_out / 1e6 * p.orca_out
    )


def campaign(a: Assumptions | None = None) -> list[CostLine]:
    """A *deduplicated* real campaign that yields all 5 figures + leaderboard + pitch.

    Train Full C2 ONCE (reused for learning-curve / bandit / invalid-rate plots),
    eval 3 conditions on train+held-out (transfer + leaderboard), then 3 extra
    ablation trainings (no_memory/no_coaching/no_gate) + their held-out evals.
    """
    a = a or Assumptions()
    lines: list[CostLine] = []

    # 1. Full C2 training (shared by learning-curve / bandit-values / invalid-rate)
    w, c = _worker_calls_training(a)
    lines.append(_line(a, "train Full C2 (x1, reused for 3 train-curve figures)", w, c))

    # 2. Transfer + leaderboard eval: 3 conditions x (train+heldout) seeds
    seeds = a.n_train_seeds + a.n_heldout_seeds
    lines.append(_line(a, "eval 3 conditions on train+heldout (transfer+leaderboard)",
                       _worker_calls_eval(a, conditions=a.n_conditions, seeds=seeds), 0))

    # 3. Ablations: each EXTRA spec RE-TRAINS (full_c2 already counted above), but
    #    no_coaching / no_gate skip the gate batch, so they are cheaper.
    extra = len(_ABLATIONS)
    wt = ct = 0
    for _name, coach, gate in _ABLATIONS:
        w2, c2 = _worker_calls_training(a, coach=coach, gate=gate)
        wt += w2
        ct += c2
    lines.append(_line(a, f"ablations: {extra} extra trainings (gate-aware)", wt, ct))
    lines.append(_line(a, f"ablations: {extra} held-out evals",
                       _worker_calls_eval(a, conditions=extra, seeds=a.n_heldout_seeds), 0))

    # 4. Pitch trace (2 episodes + 1 coach)
    lines.append(_line(a, "pitch trace", 2 * a.rounds_per_episode * a.n_workers, 1))
    return lines


def _training_seconds(a: Assumptions, lat: Latency, *, coach: bool, gate: bool) -> float:
    """Critical-path seconds for one training run.

    ALL episodes are sequential — the main episodes (online learning) *and* the
    gate batch, which ``_gate_eval_batch`` runs one after another. The only
    concurrency is the 4 workers WITHIN a round (``round_s``). So async and sync
    have the same round count and differ only in per-round latency."""
    R = a.rounds_per_episode
    p1 = max(0, a.n_train - a.phase0_length) if coach else 0
    round_s = lat.worker_s if lat.async_workers else lat.worker_s * a.n_workers
    gate_eps = p1 * a.gate_batch if (coach and gate) else 0
    rounds = (a.n_train + gate_eps) * R
    return rounds * round_s + p1 * lat.orca_s


def time_estimate(a: Assumptions, lat: Latency) -> dict:
    """Wall-clock hours: the sequential training spine + the parallel remainder.

    Training episodes are inherently sequential (online learning); eval episodes
    and the 3 ablation trainings are independent (parallelizable)."""
    import math

    t_train = _training_seconds(a, lat, coach=True, gate=True)
    t_abl_seq = sum(_training_seconds(a, lat, coach=c, gate=g) for _n, c, g in _ABLATIONS)
    seeds = a.n_train_seeds + a.n_heldout_seeds
    eval_eps = a.n_conditions * seeds * a.eval_reps + len(_ABLATIONS) * a.n_heldout_seeds * a.eval_reps
    t_eval = math.ceil(eval_eps / max(1, lat.eval_concurrency)) * a.rounds_per_episode * lat.worker_s
    return {
        "one_training_h": t_train / 3600,
        "full_seq_ablations_h": (t_train + t_abl_seq + t_eval) / 3600,
        "full_par_ablations_h": (t_train + t_eval) / 3600,  # ablations overlap the spine
    }


def summarize(lines: list[CostLine], p: Prices) -> dict:
    total = sum(_price(ln, p) for ln in lines)
    wcalls = sum(ln.worker_calls for ln in lines)
    ocalls = sum(ln.orca_calls for ln in lines)
    return {"dollars": round(total, 2), "worker_calls": wcalls, "orca_calls": ocalls}


def _print_scenario(name: str, a: Assumptions) -> float:
    lines = campaign(a)
    p = a.prices
    print(f"\n## {name}  (rounds/ep={a.rounds_per_episode}, n_train={a.n_train}, "
          f"workers={a.n_workers}, gate_batch={a.gate_batch})")
    print(f"  {'line':<52} {'wkr calls':>10} {'orca':>6} {'$ (illustr.)':>13}")
    for ln in lines:
        print(f"  {ln.label:<52} {ln.worker_calls:>10,} {ln.orca_calls:>6,} {_price(ln, p):>13.2f}")
    s = summarize(lines, p)
    print(f"  {'TOTAL':<52} {s['worker_calls']:>10,} {s['orca_calls']:>6,} {s['dollars']:>13.2f}")
    return s["dollars"]


def main() -> int:
    print("ORCA — real LLM-worker eval cost model")
    print("NOTE: token counts are derived from the loop; PRICES ARE ILLUSTRATIVE "
          "placeholders — verify before trusting $ figures.")

    shallow = Assumptions(rounds_per_episode=14, n_train=40)
    deep = Assumptions(rounds_per_episode=80, n_train=40)
    deep_lean = Assumptions(rounds_per_episode=80, n_train=24, eval_reps=4, gate_batch=1)

    d1 = _print_scenario("Shallow (iron only — Stream 2 alone)", shallow)
    d2 = _print_scenario("Deep (full DAG — Stream 1+2)", deep)
    d3 = _print_scenario("Deep, lean (n_train=24, reps=4, gate_batch=1)", deep_lean)

    print("\n## Budget check (~$150 OpenAI)")
    for name, d in (("shallow", d1), ("deep", d2), ("deep-lean", d3)):
        verdict = "OK" if d <= 150 else "OVER — reduce n_train / gate_batch / eval_reps"
        print(f"  {name:<10}: ${d:>8.2f}  [{verdict}]")

    print("\n## Wall-clock (ILLUSTRATIVE latency; the training spine is sequential)")
    def _h(x: float) -> str:
        return f"{x*60:.0f} min" if x < 1.5 else f"{x:.1f} h"
    print(f"  {'scenario':<26}{'1 training':>12}{'full (par.abl)':>16}{'full (seq.abl)':>16}")
    rows = [
        ("shallow, async", shallow, Latency(async_workers=True)),
        ("shallow, sync", shallow, Latency(async_workers=False)),
        ("deep, async", deep, Latency(async_workers=True)),
        ("deep, sync", deep, Latency(async_workers=False)),
    ]
    for name, a, lat in rows:
        t = time_estimate(a, lat)
        print(f"  {name:<26}{_h(t['one_training_h']):>12}{_h(t['full_par_ablations_h']):>16}{_h(t['full_seq_ablations_h']):>16}")
    print("  async parallelizes the 4 workers WITHIN a round (~4x on the worker portion "
          "via worker_concurrency); overall speedup is < 4x since the Orca coach calls "
          "and the sequential gate-batch episodes don't shrink. Coach latency is the big "
          "uncertainty. (No across-episode concurrency exists in the loop.)")
    print("\nLevers, by impact: gate_batch (multiplies Phase-1 worker calls; a real "
          "knob via train_full_c2(gate_batch=...)), ablation re-trainings, "
          "rounds/episode (T_max), eval_reps, n_train.")
    print("Also: eval.run_eval as written trains Full C2 8x (3 in make_all_plots + "
          "run_transfer + evaluate_conditions = 5, plus 3 ablation trainings) — for a "
          "real run, compute experiments ONCE (this campaign() model assumes that "
          "deduplicated path: 1 Full C2 + 3 ablation trainings).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

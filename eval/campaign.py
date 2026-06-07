#!/usr/bin/env python3
"""Deduplicated real-LLM eval campaign — the headline trained ONCE (§9, §10).

``eval.run_eval`` is the *offline* demo: it re-trains Full C2 ~8x against the
calibrated ``outcome_model`` (cheap, deterministic, NOT the real system). This
module is the **real** path: env + Stream-2 ``LLMWorker``s via
:class:`eval.harness.RealRunner`, with every experiment computed exactly once so
the cost matches ``eval/cost_model.py``'s campaign estimate (1 Full C2 training +
3 ablation trainings; all eval batches reuse frozen Orcas).

Dedup map (vs. run_eval's 8 trainings):
  * Full C2 is trained **once** -> ``tr``. Its frozen ``tr.orca`` feeds the
    learning-curve, bandit-value and invalid-rate figures directly, AND its
    held-out eval is reused as the transfer plot's Full-C2 bars, the leaderboard's
    full_c2 row, and the ablation plot's Full-C2 baseline bar.
  * STATIC / COMMS need no training (untrained frozen Orcas) — eval only.
  * Only the 3 ablation specs (no-memory / no-coach / no-gate) train again.

Usage:
    python -m eval.run_campaign --config configs/deep.yaml --out figures_real
    python -m eval.run_campaign --smoke          # StubLLM, offline wiring check ($0)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from config import OrcaSettings, load_config
from eval.harness import (
    ABL_NO_COACH,
    ABL_NO_GATE,
    ABL_NO_MEMORY,
    COMMS_SPEC,
    FULL_C2_SPEC,
    STATIC_SPEC,
    RealRunner,
    make_orca,
    train_full_c2,
)
from eval.plots import (
    REAL_SOURCE,
    plot_ablation,
    plot_bandit_values,
    plot_invalid_rate,
    plot_learning_curve,
    plot_transfer,
)
from eval.records import HELDOUT, TRAIN, EpisodeRecord, summarize
from eval.transfer import transfer_verdict
from eval.weave_eval import build_leaderboard, capture_pitch_trace
from llm import build_llm

# Honest provenance tag written into results.json + figure captions.
REAL_RESULT_SOURCE = "real_llm_worker_env"


def campaign(
    settings: OrcaSettings,
    out_dir: str | Path = "figures_real",
    *,
    n_train: int,
    eval_reps: int,
    gate_batch: int,
    telemetry: Any = None,
    worker_llm: Any = None,
    orca_llm: Any = None,
    source: str = REAL_SOURCE,
    result_source: str = REAL_RESULT_SOURCE,
) -> dict[str, Any]:
    """Run the full headline once on the real env + LLM workers; write figures + JSON.

    ``worker_llm`` / ``orca_llm`` default to ``build_llm(...)`` from ``settings``;
    inject ``StubLLM`` instances to exercise the whole pipeline offline ($0).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if worker_llm is None:
        worker_llm = build_llm("worker", settings)
    if orca_llm is None:
        orca_llm = build_llm("orca", settings)

    runner = RealRunner(settings, telemetry=telemetry, llm=worker_llm)
    train_seeds = list(settings.seeds.train)
    heldout = list(settings.seeds.heldout)

    def _collect(orca, seeds, spec, split):
        """One frozen eval pass -> (EpisodeRecords, raw (trace, metrics) tuples)."""
        recs: list[EpisodeRecord] = []
        eps: list[tuple] = []
        idx = 0
        for _rep in range(eval_reps):
            for s in seeds:
                config = orca.choose_config(greedy=True)
                trace, metrics = runner(
                    config,
                    s,
                    condition=spec.sim_condition,
                    episode_idx=idx,
                    coaching_active=False,
                    memory=spec.memory,
                    gate_on=spec.gate,
                )
                recs.append(
                    EpisodeRecord.from_metrics(
                        metrics, condition=spec.name, split=split, arms=config.arms
                    )
                )
                eps.append((trace, metrics))
                idx += 1
        return recs, eps

    # 1) Train Full C2 — the ONLY headline training of the money condition.
    print(f"[campaign] training Full C2 (n_train={n_train}, gate_batch={gate_batch}) ...")
    tr = train_full_c2(
        FULL_C2_SPEC, settings, runner, train_seeds, n_train, llm=orca_llm, gate_batch=gate_batch
    )

    # 2) Eval Full C2 frozen on train + held-out. Held-out reused 3x (below).
    print("[campaign] evaluating Full C2 (train + held-out) ...")
    fc_train_recs, _ = _collect(tr.orca, train_seeds, FULL_C2_SPEC, TRAIN)
    fc_held_recs, fc_held_eps = _collect(tr.orca, heldout, FULL_C2_SPEC, HELDOUT)

    transfer_records: list[EpisodeRecord] = [*fc_train_recs, *fc_held_recs]
    leaderboard_by_cond: dict[str, list[tuple]] = {"full_c2": fc_held_eps}

    # 3) Baselines: untrained frozen Orcas — eval only, no training cost.
    for spec in (STATIC_SPEC, COMMS_SPEC):
        print(f"[campaign] evaluating baseline '{spec.name}' (train + held-out) ...")
        orca = make_orca(spec, settings, llm=orca_llm)
        orca.freeze()
        tr_recs, _ = _collect(orca, train_seeds, spec, TRAIN)
        hd_recs, hd_eps = _collect(orca, heldout, spec, HELDOUT)
        transfer_records += [*tr_recs, *hd_recs]
        leaderboard_by_cond[spec.name] = hd_eps

    # 4) Ablations: reuse Full C2's held-out bar; train only the 3 ablation specs.
    ablation_records: list[EpisodeRecord] = [*fc_held_recs]
    for spec in (ABL_NO_MEMORY, ABL_NO_COACH, ABL_NO_GATE):
        print(f"[campaign] training + evaluating ablation '{spec.name}' ...")
        atr = train_full_c2(
            spec, settings, runner, train_seeds, n_train, llm=orca_llm, gate_batch=gate_batch
        )
        ahd, _ = _collect(atr.orca, heldout, spec, HELDOUT)
        ablation_records += ahd

    # 5) Pitch trace (failure -> fix -> improve) on the real runner.
    print("[campaign] capturing pitch trace ...")
    pitch = capture_pitch_trace(settings, runner=runner, seed=train_seeds[0], telemetry=telemetry)

    # 6) Figures.
    print(f"[campaign] writing figures -> {out}/ ...")
    fig_paths = {
        "learning_curve": plot_learning_curve(tr, out / "learning_curve.png", source=source),
        "bandit_values": plot_bandit_values(tr, out / "bandit_values.png", source=source),
        "transfer": plot_transfer(transfer_records, out / "transfer.png", source=source),
        "ablation": plot_ablation(ablation_records, out / "ablation.png", source=source),
        "invalid_rate": plot_invalid_rate(tr, out / "invalid_rate.png", source=source),
    }

    # 7) results.json + verdict.
    verdict = transfer_verdict(transfer_records)
    transfer_stats = summarize(transfer_records)
    leaderboard = build_leaderboard(leaderboard_by_cond)
    results = {
        "result_source": result_source,
        "n_train": n_train,
        "eval_reps": eval_reps,
        "gate_batch": gate_batch,
        "worker_provider": settings.llm.worker_provider or settings.llm.provider,
        "orca_provider": settings.llm.orca_provider or settings.llm.provider,
        "figures": {k: str(v) for k, v in fig_paths.items()},
        "transfer_verdict": verdict,
        "transfer_table": {f"{c}/{s}": st.as_tuple() for (c, s), st in transfer_stats.items()},
        "leaderboard": leaderboard,
        "pitch_trace": pitch,
        "trained_seeds": sorted(tr.orca.trained_seeds),
        "heldout_seeds": list(settings.seeds.heldout),
    }
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # ---- printed report ---------------------------------------------------- #
    print("\n" + "=" * 72)
    print(f"ORCA EVAL — HEADLINE  [{result_source}]")
    print("=" * 72)
    print("\nTransfer (mean frontier ± std), held-out {B,C}:")
    for cond in ("static", "comms", "full_c2"):
        st = transfer_stats.get((cond, HELDOUT))
        if st:
            print(f"  {cond:>8} / held-out: {st.mean:.3f} ± {st.std:.3f}")
    print(f"  => Full C2 ≥ baselines on held-out: {verdict['full_c2_wins']}")
    print(f"\nHeld-out guard: trained on {results['trained_seeds']}; "
          f"held-out {results['heldout_seeds']} never trained.")
    print(f"\n[campaign] wrote {out}/results.json + {len(fig_paths)} figures")
    return results


__all__ = ["campaign", "REAL_RESULT_SOURCE"]

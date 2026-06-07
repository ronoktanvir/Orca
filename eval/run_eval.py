#!/usr/bin/env python3
"""One-command eval demo (§9, §10) — Stream 3.

Generates the whole headline in one shot:

    python -m eval.run_eval                 # offline, calibrated outcome model
    python -m eval.run_eval --out figures   # choose the figure dir
    python -m eval.run_eval --weave         # also init Weave so traces/leaderboard log live

Outputs: the five figures, a ``results.json`` summary (transfer verdict, leaderboard,
pitch trace), and a printed report. Runs fully offline by default (the calibrated
``outcome_model``); pass ``--weave`` to log the pitch trace + leaderboard to the
live Weave project as well.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from config import load_config, load_dotenv
from eval.plots import make_all_plots
from eval.records import HELDOUT, summarize
from eval.transfer import run_transfer, transfer_verdict
from eval.weave_eval import build_leaderboard, capture_pitch_trace, evaluate_conditions

# These offline numbers come from the calibrated outcome model, NOT the real
# LLM-worker environment. Transfer is demonstrated on that scaffold; it is not yet
# evidence that the real C2 system transfers (that needs Stream 2's LLMWorker
# evaluated via eval.harness.RealRunner).
CAVEAT = (
    "OFFLINE CALIBRATED OUTCOME-MODEL RESULT (eval/outcome_model.py) — a CI/demo "
    "scaffold, NOT the real LLM-worker env. Real-system transfer is unproven until "
    "Stream 2's LLMWorker is evaluated via RealRunner."
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the Orca eval + produce the headline figures.")
    p.add_argument("--out", default="figures", help="output directory for figures + results.json")
    p.add_argument("--episodes", type=int, default=40, help="training episodes for Full C2")
    p.add_argument("--reps", type=int, default=8, help="eval repetitions per seed")
    p.add_argument("--weave", action="store_true", help="init Weave so the pitch trace logs live")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    settings = load_config()
    telemetry = None
    if args.weave:
        load_dotenv()
        from telemetry import init_telemetry

        telemetry = init_telemetry(
            mode="weave",
            entity=settings.telemetry.entity,
            project=settings.telemetry.project,
            run_dir=settings.telemetry.run_dir,
        )
        print(f"[eval] {telemetry.summary()}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[eval] generating figures -> {out}/ ...")
    fig_paths = make_all_plots(out, n_train=args.episodes, eval_reps=args.reps)

    print("[eval] running transfer experiment ...")
    transfer = run_transfer(n_train=args.episodes, eval_reps=args.reps)
    verdict = transfer_verdict(transfer.records)
    transfer_stats = summarize(transfer.records)

    print("[eval] building Weave leaderboard ...")
    by_cond = evaluate_conditions(n_train=args.episodes, reps=args.reps)
    leaderboard = build_leaderboard(by_cond)

    print("[eval] capturing the pitch trace (failure -> fix -> improve) ...")
    pitch = capture_pitch_trace(settings, seed=settings.seeds.train[0], telemetry=telemetry)

    results = {
        "result_source": "calibrated_outcome_model",
        "caveat": CAVEAT,
        "figures": {k: str(v) for k, v in fig_paths.items()},
        "transfer_verdict": verdict,
        "transfer_table": {f"{c}/{s}": st.as_tuple() for (c, s), st in transfer_stats.items()},
        "leaderboard": leaderboard,
        "pitch_trace": pitch,
        "trained_seeds": sorted(transfer.train_result.orca.trained_seeds),
        "heldout_seeds": list(settings.seeds.heldout),
    }
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # ---- printed report ---------------------------------------------------- #
    print("\n" + "=" * 72)
    print("ORCA EVAL — HEADLINE  [offline calibrated outcome-model result]")
    print("=" * 72)
    print("\n!! CAVEAT: " + CAVEAT)
    print("\nTransfer (mean frontier ± std) — calibrated outcome model, not real env:")
    for cond in ("static", "comms", "full_c2"):
        st = transfer_stats.get((cond, HELDOUT))
        if st:
            print(f"  {cond:>8} / held-out {{B,C}}: {st.mean:.3f} ± {st.std:.3f}")
    print(f"  => Full C2 ≥ baselines on held-out: {verdict['full_c2_wins']}")

    print("\nLeaderboard (held-out, by scorer):")
    hdr = "  cond     | " + " | ".join(f"{n[:9]:>9}" for n in leaderboard["full_c2"] if n != "n")
    print(hdr)
    for cond, row in sorted(leaderboard.items(), key=lambda kv: -kv[1]["frontier"]):
        cells = " | ".join(f"{row[n]:>9.3f}" for n in row if n != "n")
        print(f"  {cond:<8} | {cells}")

    print("\nPitch trace (one nested Orca trace):")
    print(f"  bottleneck agent: {pitch['bottleneck_agent']}")
    print(f"  before: frontier={pitch['before']['frontier']:.3f} ({pitch['before']['milestone']}) "
          f"invalid={pitch['before']['invalid_rate']:.2f}")
    print(f"  Orca:   {pitch['coach_rationale'].splitlines()[0]}")
    if pitch["card_diff_added_directives"]:
        print(f"  edit:   + {pitch['card_diff_added_directives'][0]}")
    print(f"  after:  frontier={pitch['after']['frontier']:.3f} ({pitch['after']['milestone']}) "
          f"invalid={pitch['after']['invalid_rate']:.2f}  -> improved={pitch['improved']}")

    print(f"\nHeld-out guard: trained on {results['trained_seeds']}; "
          f"held-out {results['heldout_seeds']} never trained.")
    print(f"\n[eval] wrote {out}/results.json + 5 figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

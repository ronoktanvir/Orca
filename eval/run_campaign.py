#!/usr/bin/env python3
"""CLI for the deduplicated real-LLM eval campaign (see ``eval.campaign``).

    python -m eval.run_campaign --config configs/deep.yaml --out figures_real
    python -m eval.run_campaign --config configs/deep.yaml --weave   # live Weave traces
    python -m eval.run_campaign --smoke                              # $0 StubLLM wiring check

``--smoke`` swaps in offline ``StubLLM`` clients and tiny knobs so the entire
pipeline (train -> eval -> ablate -> plot -> json) runs with zero API calls.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import load_config, load_dotenv
from eval.campaign import campaign
from eval.plots import REAL_SOURCE, SIM_SOURCE


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the deduplicated real-LLM Orca eval campaign.")
    p.add_argument("--config", default=None, help="preset YAML (e.g. configs/deep.yaml)")
    p.add_argument("--out", default="figures_real", help="output dir for figures + results.json")
    p.add_argument("--episodes", type=int, default=None, help="override eval.n_train")
    p.add_argument("--reps", type=int, default=None, help="override eval.eval_reps")
    p.add_argument("--gate-batch", type=int, default=None, help="override eval.gate_batch")
    p.add_argument("--weave", action="store_true", help="init Weave so traces log live")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="offline wiring check: StubLLM clients + tiny knobs, zero API calls",
    )
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    load_dotenv()  # OPENAI / WANDB keys from .env
    settings = load_config(args.config)

    n_train = args.episodes if args.episodes is not None else settings.eval.n_train
    eval_reps = args.reps if args.reps is not None else settings.eval.eval_reps
    gate_batch = args.gate_batch if args.gate_batch is not None else settings.eval.gate_batch

    worker_llm = orca_llm = None
    source = REAL_SOURCE
    if args.smoke:
        from llm import StubLLM

        worker_llm = StubLLM()
        orca_llm = StubLLM()
        source = SIM_SOURCE
        n_train = args.episodes if args.episodes is not None else 2
        eval_reps = args.reps if args.reps is not None else 1
        gate_batch = args.gate_batch if args.gate_batch is not None else 1
        print("[campaign] SMOKE MODE — StubLLM, no API calls")

    telemetry = None
    if args.weave:
        from telemetry import init_telemetry

        telemetry = init_telemetry(
            mode="weave",
            entity=settings.telemetry.entity,
            project=settings.telemetry.project,
            run_dir=settings.telemetry.run_dir,
        )
        print(f"[campaign] {telemetry.summary()}")

    worker_provider = settings.llm.worker_provider or settings.llm.provider
    orca_provider = settings.llm.orca_provider or settings.llm.provider
    print(
        f"[campaign] config={args.config or 'default'} · n_train={n_train} · reps={eval_reps} "
        f"· gate_batch={gate_batch} · workers={worker_provider} · orca={orca_provider} "
        f"· out={Path(args.out)}"
    )

    campaign(
        settings,
        args.out,
        n_train=n_train,
        eval_reps=eval_reps,
        gate_batch=gate_batch,
        telemetry=telemetry,
        worker_llm=worker_llm,
        orca_llm=orca_llm,
        source=source,
        result_source="smoke_stub_llm" if args.smoke else "real_llm_worker_env",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

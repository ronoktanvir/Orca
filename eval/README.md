# Orca — Stream 3: the manager + the eval (the headline)

Orca is the claim that **a manager LLM learns a transferable delegation strategy**
over a team of worker LLMs (Architecture C2). This folder produces the evidence:
the learning curve, the transfer plot, the ablations, the Weave leaderboard, and
the one *failure → fix → improve* pitch trace.

---

## ⚠️ Read this first — where the numbers come from

The offline figures and the transfer verdict are an **offline calibrated
outcome-model result**, produced by [`outcome_model.py`](outcome_model.py) — a
CI/demo scaffold, **not** the real LLM-worker environment.

Why a scaffold is needed: the Phase-0 stub env + scripted oracle reach IRON
deterministically *regardless of Orca's choices*, so they can't move the bandit or
separate conditions offline. The real coupling (behavior-card → worker behaviour →
frontier) only exists once **Stream 2's `LLMWorker`** is wired in, and the full
depth needs **Stream 1's deep env**.

So: **real-system transfer is not yet proven.** What is proven is that the
*machinery* is correct and runner-agnostic — [`RealRunner`](harness.py) is the
drop-in seam, and the harness / plots / verdict are identical, so the same code
yields the real result once that path is evaluated. The transfer figure's title is
tagged "(calibrated outcome model)" and **every** figure carries a source footnote;
`run_eval` prints the caveat before any result numbers; `results.json` carries
`result_source: calibrated_outcome_model`.

**Anti-circularity:** `EpisodeMetrics.team_reward` is the objective DAG frontier
minus objective penalties, once per episode. Orca's `performance_score` /
`learning_signal` are advisory — logged, used for coaching — and **never** summed
into `team_reward`, fed to the bandit's value, or used in the gate's decision.

---

## Reproduce

```bash
python -m eval.run_eval                 # 5 figures -> figures/, results.json, printed report
python -m eval.run_eval --weave         # also log the pitch trace + leaderboard live to Weave
python -m eval.cost_model               # token/$ estimate for a REAL LLM-worker run
```

`figures/` is gitignored (reproducible). The run is deterministic (seeded via
`env.rng.derive_seed`).

## The five figures

| Figure | What it shows |
|---|---|
| `learning_curve.png` | Team frontier rises over training episodes (Full C2 on {A,T2,T3}). |
| `bandit_values.png` | Per-arm value over episodes; Orca learns **explore_heavy** (S1) and **two_pairs** (S3). |
| `transfer.png` | 3 conditions × {train, held-out}; **Full C2 ≥ baselines on held-out {B,C}**. |
| `ablation.png` | Full C2 vs −memory / −coaching / −accept-gate on held-out (each knob's value). |
| `invalid_rate.png` | Invalid-rate drops when coaching turns on (Phase 1) and clears the miner bottleneck. |

## The conditions (§9)

- **static** — fixed roster, no Orca, no comms, no memory.
- **comms** — agents message, but no delegation/coaching/memory.
- **full_c2** — Orca bandit + (phased) coaching + memory + accept-gate.

Transfer test: train Full C2 on `{A,T2,T3}`, freeze the learned bandit + cards,
evaluate all three on held-out `{B,C}`. The frozen Orca's `trained_seeds` proves
`{B,C}` were never trained on (anti-leakage).

## The pitch trace (§10)

`weave_eval.capture_pitch_trace` runs one nested trace: an episode where the
iron-miner repeatedly fails → Orca's coach assigns credit (execution error) and
rewrites the card → the next run clears the bottleneck (lower invalid-rate, deeper
frontier). Each step is a `@weave.op`, so it nests in Weave when telemetry is live.

## Files

| File | Role |
|---|---|
| `outcome_model.py` | **Calibrated scaffold** — synthetic but objective episodes (see caveat). |
| `harness.py` | Runners (`SimRunner`, `RealRunner`), training-with-phasing-and-gate, experiments. |
| `records.py` | Flat `EpisodeRecord` + variance aggregation (`summarize`). |
| `scorers.py` | The 5 Weave scorers (frontier / milestone / time-to-win / invalid-rate / cooperation). |
| `weave_eval.py` | Leaderboard, best-effort `weave.Evaluation`, pitch trace. |
| `plots.py` | The 5 matplotlib figures (lazy import; source footnote stamped). |
| `baselines.py`, `transfer.py`, `ablations.py` | Thin, discoverable entry points to `harness`. |
| `run_eval.py` | One-command demo driver. |
| `cost_model.py` | Token/$ budget estimator for the real run. |

## What's needed to make this real (blocked on other streams)

- **Stream 2 — `LLMWorker`** (hard blocker): replaces the scripted oracle so cards
  actually change behaviour. Swap via `worker_factory=...` (seam pre-validated in
  `tests/test_integration.py`). Minimum for a real (shallow) run.
- **Stream 1 — deep env**: frontier beyond iron so S2–S4, wins, `time_to_win`, and
  the Phase-2 speed bonus come alive. Needed for the full headline.

Then: swap the oracle for `LLMWorker`, make the worker calls async, run via
`RealRunner`, and the same plots/verdict become the real result. Cost: see
`python -m eval.cost_model` (illustrative ~$17 shallow / ~$95 deep vs the ~$150
OpenAI budget; the accept-gate's per-Phase-1 re-runs and ablation re-trainings are
the dominant levers).

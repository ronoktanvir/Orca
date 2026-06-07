# Stream 3 — Orca + Eval

> Read [handoff/README.md](README.md) first (framing, setup, the LLM seam, the 4
> rules). **You carry the headline result.** Orca is a proof that a manager LLM
> learns *transferable delegation* — your bandit curve + transfer plot + the
> failure→fix→improve Weave trace ARE the demo. You also own loop integration.

## You own
`orca/`, `eval/`, `telemetry/`. **Don't edit** other streams' folders or the 7
contracts (additive-only, broadcast first). You also **integrate `train/loop.py`**
(shared glue — pair with Stream 2 on agent construction).

## What you replace
- `orca/orca.py` — `NoOpOrca` (frozen cards, no learning). Build the real Orca on
  the same method surface (`choose_config / observe_outcome / coach / commit`) so
  the loop doesn't churn.
- `orca/bandit.py` — `EpsilonGreedyBandit` skeleton (working ε-greedy). Wire it in.
- `orca/coach.py` — verbal coach + credit reasoning (stub).
- `orca/gate.py` — accept/reject gate (stub returns True).
- `orca/cards.py` — `DEFAULT_ROSTER` + default cards (Phase 0 frozen).
- `eval/` — `baselines.py`, `transfer.py`, `ablations.py`, `plots.py` (skeletons).
- `telemetry/weave_ops.py` — `@op` + safe fallback (working) — deepen with the
  Weave Evaluation harness, leaderboard, and the pitch trace.

LLM seam: `from llm import build_llm; orca_llm = build_llm("orca", settings)`
(gpt-5). The objective DAG frontier (`EpisodeMetrics.team_reward`) is the headline
reward — Orca's scores stay **advisory** (§6.4 anti-circularity).

## Tasks (done-when) — build spec §6, §7.3, §9, §10
- **O1** Trace digest from `EpisodeTrace`. *Compact per-agent + team summary.*
  (start now on stub/foundation traces) [§6.1]
- **O2** Delegation **bandit** — situations, arms, ε-greedy/Thompson, per-episode
  update on `team_reward`. *Arm values update; chosen-arm frontier curve plotted.*
  ▶ the learning curve. [§6.3]
- **O3** Scoring — `performance_score` + `learning_signal` (mostly objective from
  env stats). *Computed from `EpisodeMetrics.agent_stats`.* [§7.3]
- **O4** Verbal coach → behavior-cards + credit reasoning (delegation vs
  execution). *Cards updated with readable rationale, logged to Weave.* [§6.4]
- **O5** Accept/reject gate. *Update kept iff eval-pool mean frontier not
  regressed (within ε); else rollback. Keep a static baseline snapshot.* [§6.5]
- **O6** Phasing controller. *Phase 0 freezes cards; Phase 1 enables coaching;
  Phase 2 enables speed reward post-win.* (`train/phases.py` has the skeleton) [§6.6]
- **O7** Eval harness — baselines (static / comms-no-Orca / Full C2), transfer
  (A/T2/T3 → B/C), ablations (memory/coaching/gate on/off), plots. *Transfer bar
  chart + bandit curve + 1 ablation, with variance over seeds/episodes.* ▶ the
  headline result. [§9]
- **O8** Weave Evaluation + leaderboard + the **failure→fix→improve** trace.
  *Comparison view captured for the pitch.* ▶ [§10]

## Invariants you must keep green
- Headline reward = **objective DAG frontier**, once per episode. Orca's opinion
  is advisory and **never** summed into `team_reward` (anti-circularity §6.4).
- The bandit acts **once per episode** over a tiny discrete space (no PPO/backprop).
- Accept-gate every update; **never train on held-out B/C**; always report
  variance over multiple seeds/episodes (no single anecdote).
- `@op` everything load-bearing so the Weave trace nests (§10). Keep the local/
  no-op fallback intact so `main` runs offline.

## Loop integration (you own this)
In `train/loop.py`, flip the already-wired no-ops on: `observe_outcome` (bandit
update), `coach`, `accept_gate`, plus seed rotation across `TRAIN_SEEDS` and the
phasing controller. Coordinate with Stream 2 to call agents async and swap
`ShallowOracle`→`LLMWorker`. Keep the oracle path runnable (offline fallback).

## Definition of done
The bandit value curve rises across episodes; Orca rewrites a card after spotting
a failure and the next run improves (captured as one nested Weave trace); and the
transfer plot shows **Full C2 ≥ baselines on held-out B/C**. That's the pitch.

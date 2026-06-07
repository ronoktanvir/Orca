# Stream 3 — Orca + Eval (in-depth brief)

> Paste-ready kickoff for the Orca+Eval coding agent. Read alongside
> [handoff/README.md](README.md) and build spec **§6, §7.3, §9, §10**.

## You are the Orca + Eval builder — you carry the headline result.

**What Orca is (read twice):** Orca is a *proof that a manager LLM learns
transferable delegation strategy* over a team of worker LLMs (Architecture C2).
The whole demo IS your output: the **delegation bandit's learning curve**, the
**transfer plot** (Full C2 ≥ baselines on held-out seeds), and one
**failure→fix→improve Weave trace.** **Do NOT build:** PPO or any
gradient/backprop learner, per-step rewards, or anything that lets Orca's own
opinion become the headline reward. Orca acts **once per episode** over a tiny
discrete space — that's a bandit by construction. The headline reward is the
**objective DAG frontier**, always.

**The repo you're inheriting:** Phase 0 is done and green on `main`
(https://github.com/ronoktanvir/Orca): real end-to-end loop, no-op Orca, **live
Weave tracing already working** (project
`ronoktanvir-university-of-california-berkeley/orca`), the shared LLM client built
(`build_llm("orca", settings)` → gpt-5), 62 passing tests, 7 frozen contracts.
You replace the no-op Orca with the real one, build the eval harness, deepen
telemetry — **and you own `train/loop.py` integration.**

## Setup (once)
```bash
git clone https://github.com/ronoktanvir/Orca && cd Orca
uv venv --python 3.11 .venv && uv pip install -r requirements.txt
cp .env.example .env          # keys from Ronok; then `wandb login` with the Weave key
.venv/bin/python run.py       # publishes a live Weave trace; reaches "iron"
.venv/bin/python -m pytest -q # all green
git checkout -b stream3-orca
```

## THE LAWS
1. **Never change the 7 contracts** except additively + broadcast.
2. **Edit `orca/`, `eval/`, `telemetry/`** — plus you **own `train/loop.py`**
   integration (coordinate edits there with Stream 2).
3. **Green-main:** every merge passes `python run.py` + `pytest -q`. Keep the
   local/no-op telemetry fallback intact so `main` runs offline.
4. **Anti-circularity (the big one):** the headline reward is
   `EpisodeMetrics.team_reward` (objective DAG frontier, once/episode). Orca's
   `performance_score`/`learning_signal` are **advisory** and **never** summed into
   `team_reward`. **Never train on held-out B/C.** Always report variance over
   multiple seeds/episodes — no single anecdote.

## The LLM seam
`from llm import build_llm; orca_llm = build_llm("orca", settings)` (gpt-5).
`orca_llm.complete(prompt, schema=...)` → raw text; validate with pydantic.

## The contracts you consume/produce (from `contracts/`)
- **In:** `EpisodeTrace` (raw: `action_records[ActionRecord(round, agent_id,
  action, valid, reason, result)]`, `messages`, `milestone_timeline`,
  `frontier_reached`, `behavior_cards`, `observations`) and `EpisodeMetrics`
  (computed: `team_reward`, `frontier_milestone/value`, `penalties`,
  `invalid_rate`, `idle_fraction`, `agent_stats[AgentStats]`, `milestone_timeline`,
  `won`).
- **Out:** updated `BehaviorCard`s (assignment + directives + priorities + donts,
  bump `version`), per-agent `performance_score`∈[0,1] + `learning_signal`∈[−1,1].
- The no-op Orca already has the method surface you implement against:
  `NoOpOrca.choose_config(history)→OrcaConfig(roster, behavior_cards, arms)`,
  `observe_outcome(config, metrics)`, `coach(trace, metrics)→Proposal`,
  `commit(proposal)`. Keep these signatures so the loop doesn't churn.

## Your files & current stubs
`orca/orca.py` (`NoOpOrca`, `OrcaConfig`, `Proposal`), `orca/bandit.py` (working
`EpsilonGreedyBandit`: `choose(situation)`, `update(situation,arm,frontier)`,
`values()`), `orca/coach.py` (stub), `orca/gate.py` (stub returns True),
`orca/cards.py` (`DEFAULT_ROSTER`, `default_cards`),
`eval/{baselines,transfer,ablations,plots}.py` (skeletons),
`telemetry/weave_ops.py` (`@op` + safe fallback + `init_telemetry(mode, entity,
project, ...)`), `train/phases.py` (`Phase`, `current_phase`).

## Your tasks (in order)

**O1 — Trace digest (§6.1).** From `EpisodeTrace`+`EpisodeMetrics`, build a compact
per-agent + team summary (frontier + milestone timeline, per-agent subtask
completion/invalids/idle/deaths/handoffs/useful messages, bottlenecks: longest
stalls, repeated invalids, starvation). *Start now on foundation traces — not
blocked.* **Done when:** a readable digest object exists.

**O2 — Delegation bandit (§6.3).** ▶ *The learning curve.* Define recurring
situations (S1 early-game role assignment, S2 nether-entry policy, S3
fortress-search formation, S4 end-approach), each with 2–4 arms. Use
`EpsilonGreedyBandit`; value = running mean of episode `team_reward` when an arm
was chosen; **update once per episode**. Wire `Orca.choose_config` to pick arms and
`observe_outcome` to update. **Done when:** arm values move; you can plot
chosen-arm frontier over episodes.

**O3 — Scoring (§7.3).** Compute `performance_score` (mostly objective: subtask
completion + frontier contribution + low invalid/idle, lightly Orca's opinion) and
`learning_signal` (Orca's "adopt this lesson?" dial) from `agent_stats`. **Done
when:** scores come from env stats, not vibes.

**O4 — Verbal coach + credit assignment (§6.4).** Have `orca_llm` read the digest
and reason in NL about credit — **delegation error vs execution error** — then
write the next `BehaviorCard` (assignment + directives), bumping `version`. Log the
rationale to Weave. **Done when:** cards update with readable, logged reasoning.

**O5 — Accept/reject gate (§6.5).** After Orca proposes cards/memory/delegation,
run a small eval batch on the train seed pool; **keep iff mean team frontier ≥
current best − ε, else roll back.** Maintain a static-baseline snapshot. **Done
when:** noisy LLM edits become monotone-ish; rollbacks happen and are logged.

**O6 — Phasing (§6.6).** Use `train/phases.py`: Phase 0 cards frozen (bandit only);
Phase 1 coaching on (accept-gated); Phase 2 speed reward **only after first win**.
**Done when:** the loop respects phase gates.

**O7 — Eval harness (§9).** ▶ *The headline.* Three conditions: static baseline,
comms-no-Orca, Full C2. **Transfer test:** train Full C2 on {A,T2,T3}, freeze,
eval all three on held-out {B,C}. Ablations: memory / coaching / accept-gate
on-off. Plots (matplotlib): learning curve, bandit arm-value curve, transfer bar
chart (3 conditions × {train,held-out}) with variance, one ablation, invalid-rate
over time. **Done when:** the transfer plot shows Full C2 ≥ baselines on B/C.

**O8 — Weave Evaluation + leaderboard + pitch trace (§10).** Use Weave's
Evaluation/comparison + custom scorers (frontier, milestone, time-to-win,
invalid-rate, cooperation-events). Capture the single nested trace where **Orca
spots a failure → edits a card → next run improves.** **Done when:** the
comparison view + that trace are captured for the pitch.

**Loop integration (you own it).** In `train/loop.py`, flip the already-wired
no-ops on — `observe_outcome` (bandit), `coach`, `accept_gate` — plus rotate
`TRAIN_SEEDS` and apply phasing. Pair with Stream 2 to call agents async and swap
`ShallowOracle`→`LLMWorker`. Keep the oracle path runnable as the offline fallback.

## Tests you must add
Bandit update math; gate keeps-vs-rollback logic; scoring is purely from
`agent_stats`; a held-out-seed guard (B/C never appear in training history). Don't
hit the LLM API in `pytest` (mock it).

## Definition of done
The bandit value curve rises across episodes; Orca rewrites a card after spotting a
failure and the next run improves (captured as one nested Weave trace); the
transfer plot shows **Full C2 ≥ baselines on held-out B/C**. That's the pitch —
own it end to end.

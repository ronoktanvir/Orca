# Stream 2 — Agents (in-depth brief)

> Paste-ready kickoff for the Agents coding agent. Read alongside
> [handoff/README.md](README.md) and build spec **§4 (workers)** + **§5 (bus)**.

## You are the Agents builder for Orca.

**What Orca is (read twice):** Orca is a *proof that a manager LLM learns
transferable delegation strategy* over a team of worker LLMs (Architecture C2).
The demo lives or dies on frozen contracts, no coordinate leakage, an objective
DAG-frontier reward, Weave traces, baselines, and held-out-seed transfer. You
build the **4 worker LLMs + the comm bus + execution-memory.** **Do NOT build:**
PPO, per-step rewards, coordinate-based memories, or anything that lets
seed-specific detail (coords, distances, region ids, unique landmark names) into
persistent memory. Workers improve **verbally** (self-written HOW-TO memory) —
not by gradients.

**The repo you're inheriting:** Phase 0 is done and green on `main`
(https://github.com/ronoktanvir/Orca): a real end-to-end loop with a scripted
oracle, no-op Orca, Weave logging, 62 passing tests, and the 7 frozen contracts.
The **shared LLM client is already built and verified** — you call it, you don't
build it. Your job: replace the scripted placeholder with real LLM workers that
play full episodes, message each other, and write transferable memory.

## Setup (once)
```bash
git clone https://github.com/ronoktanvir/Orca && cd Orca
uv venv --python 3.11 .venv && uv pip install -r requirements.txt
cp .env.example .env          # paste the keys Ronok DMs you (never commit them)
.venv/bin/python run.py       # must reach "iron"
.venv/bin/python -m pytest -q # all green
git checkout -b stream2-agents
```

## THE LAWS
1. **Never change the 7 contracts** except additively + broadcast.
2. **Edit only `agents/` and `bus/`.** Loop changes (async, swapping the agent) =
   PR tagged to Stream 3.
3. **Green-main:** every merge passes `python run.py` + `pytest -q`. Keep the
   scripted oracle working as the offline fallback so `main` never goes dark.
4. **Execution-memory = HOW-TO only**, schema'd, cap 8, guard-filtered: **no
   coords, no distances, no region ids, no unique landmark names.** Load-bearing
   demo mechanism and part of the no-leak invariant.

## The LLM seam (already built — use it)
```python
from config import load_config, load_dotenv
from llm import build_llm
load_dotenv(); settings = load_config()
worker_llm = build_llm("worker", settings)            # gpt-5-mini by default
text = worker_llm.complete(prompt, schema=Action)      # JSON mode when schema given; returns RAW TEXT
```
`complete()` returns raw text — **you** validate with pydantic and do the one-shot
repair retry (task A2). Models live in `configs/default.yaml → llm:` (workers
`gpt-5-mini`, swappable to GLM-5.1 via `provider: wandb_inference`). Low
temperature; full prompts/outputs auto-log to Weave.

## The contracts you build against (from `contracts/`)
- **What the worker sees** — `Observation`: `round`, `time_of_day`, `self` (role,
  health, hunger, inventory `dict[str,int]`, status, current_biome, layer), `here`
  (resources_visible, structure, mobs, `exits=[{dir,distance_band,biome_hint}]`,
  frontier_dirs), `teammates` (relative only: distance_band, bearing, role),
  `known_landmarks` (abstract), `recent_messages`, `assignment` (from the
  BehaviorCard), `dag_frontier_reached`. **No coordinates anywhere.**
- **What the worker outputs** — `Action(name: ActionName, args: dict)` + a list of
  `Message`. `ActionName` ∈ move/scout/gather/craft/smelt/place/fight/eat/sleep/
  give_item/request_help/regroup/report/wait. The **env**, not you, decides
  validity — an illegal action loses the turn and is logged `invalid_action`.
- `Message(from→alias, to, type∈report/request_help/share_finding/
  propose_rendezvous/ack/handoff, content, urgency, round)`.
- `BehaviorCard(agent_id, role, assignment, directives, priorities, donts,
  version)` — Orca-authored, read at episode start.
- `ExecutionMemory(agent_id, heuristics: list[Heuristic≤8])`,
  `Heuristic(condition, action, confidence)`.

## Your files & current stubs
`agents/base.py` (`Agent` protocol: `act(obs)→Action`), `agents/scripted.py`
(`ShallowOracle` — KEEP it), `agents/worker.py` (`LLMWorker` — currently raises;
build it; `safe_default()→wait`), `agents/prompts.py` (`ROLE_PRIMERS` done,
`build_worker_prompt` stub), `agents/memory.py` (`guard_filter`/
`looks_seed_specific` working), `bus/bus.py` (`CommBus` t+1 delivery),
`bus/messages.py` (`make_message`).

## Your tasks (in order)

**A1 — Worker turn loop + prompt builder (§4.2–4.3).** ▶ Build
`build_worker_prompt(obs, card, memory, history_summary)`: system =
`ROLE_PRIMERS[role]` + behavior-card + execution-memory + the action menu + the
strict output schema; user = the JSON obs + a compact running history summary +
the team DAG frontier. Build `LLMWorker.act(obs)` to call `worker_llm.complete`
and return an `Action`. **Done when:** one LLM agent plays a full episode.

**A2 — Parse/validate + invalid handling (§4.4).** Validate the model's JSON into
`Action` + `Message[]` with pydantic. On malformed output: **one repair retry**,
else default to `wait` + log `parse_failure`. Never crash. **Done when:** garbage
output degrades gracefully, no exceptions escape.

**A3 — Scale to 4 agents, async parallel (§3.6).** ▶ *The comms-no-Orca baseline.*
Call all 4 free agents concurrently each round (asyncio). Roster =
`orca.cards.DEFAULT_ROSTER` (explorer/miner/tinkerer/support); flip
`run.single_agent_oracle: false`. **Needs a `train/loop.py` change — coordinate
with Stream 3** (give them `LLMWorker(agent_id, llm)` to drop in). **Done when:**
4 agents act per round in parallel; they visibly stall/duplicate without a manager.

**A4 — Comm bus + delivery + history summarization (§5).** Wire `CommBus` into the
loop: messages posted round t delivered round t+1; each agent sees the last K
(`message_window`) addressed to it or `team`; **content never truncated**, but old
turns are summarized into the running history. Log all messages verbatim (Weave).
**Done when:** agents coordinate over the bus and history stays bounded.

**A5 — Execution-memory write + guard filter + learning_signal (§4.5).** At
episode end, prompt the agent for ONLY transferable HOW-TO heuristics
(`{condition, action, confidence}`, cap 8). Run `guard_filter` (strip coord-like/
seed-specific) before persisting. Scale edit magnitude by Orca's `learning_signal`
(+1 bake in, ~0 ignore, −1 weaken/remove). Persist across episodes (accept-gated
by Stream 3). **Done when:** memory persists, the filter strips coord-like content
(test it), `learning_signal` modulates writes.

**A6 — Role primers + card consumption (§4.1, §4.3).** Agents read `assignment`/
directives/priorities/donts from their `BehaviorCard`. Roles are **soft priors,
never hard action masks** — a miner *can* build. **Done when:** changing a card
changes behavior.

## Tests you must add
Malformed-output → repair/wait (no crash); a memory guard test (a coord-laden
heuristic is dropped); a card-consumption test; keep `coord_leak_test` green
(workers must never echo coords into messages/memory). **Mock the LLM in tests —
don't hit the API in `pytest`** (live-test pattern is in `tests/test_llm.py`,
gated by `ORCA_LIVE_LLM=1`).

## Definition of done
4 LLM agents play a full episode on the real env, message over the bus, read their
cards, and write guard-filtered memory across episodes — JSON validity enforced,
no crashes. Demoable: the comms-no-Orca baseline (they stall/duplicate without Orca).

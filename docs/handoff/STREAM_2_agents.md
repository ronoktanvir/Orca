# Stream 2 — Agents

> Read [handoff/README.md](README.md) first (framing, setup, the LLM seam, the 4
> rules). Orca is a proof that a manager LLM learns *transferable delegation* over
> worker LLMs. Your job is the **workers + comms**. **Don't build** coordinate
> memories, per-step rewards, or anything that leaks seed specifics into memory.

## You own
`agents/`, `bus/`. **Don't edit** other streams' folders or the 7 contracts
(additive-only, broadcast first).

## What you replace
- `agents/scripted.py` — `ShallowOracle` (reaches iron). **Keep it** — Stream 1/3
  use it as the oracle/baseline. You ADD the LLM worker alongside it.
- `agents/worker.py` — `LLMWorker` (currently raises NotImplementedError). Build it.
- `agents/prompts.py` — `ROLE_PRIMERS` (done) + `build_worker_prompt` (stub).
- `agents/memory.py` — `guard_filter` / `looks_seed_specific` (working) — extend
  with the LLM-written memory + `learning_signal` modulation + persistence.
- `bus/bus.py` — `CommBus` (t+1 delivery, working) — wire into the loop + add
  history summarization. `bus/messages.py` — `Message` helpers.

The LLM seam is ready: `from llm import build_llm; llm = build_llm("worker",
settings)`; `llm.complete(prompt, schema=...)` returns text (you validate).

## Tasks (done-when) — build spec §4, §5
- **A1** Worker turn loop + prompt builder. *1 LLM agent plays a full episode on
  the (stub) env.* ▶ [§4.2–4.3]
- **A2** JSON parse/validate (pydantic `Action`+`messages`) + invalid handling.
  *Malformed output → one repair retry, else default to `wait` + log
  `parse_failure`; never crashes.* [§4.4]
- **A3** Scale to 4 agents, **async parallel** calls per round. *4 agents act in
  parallel each round.* ▶ this is the comms-no-Orca baseline (they duplicate work).
  [§3.6] (requires a coordinated `train/loop.py` edit — see README integration)
- **A4** Comm bus + delivery (t+1) + history summarization. *Messages logged,
  delivered next round, old turns summarized (content never truncated).* [§5]
- **A5** Execution-memory write + **guard filter** + `learning_signal`
  modulation. *Memory (cap 8) persists across episodes; filter strips coord-like
  content (test it); edit magnitude scales with Orca's `learning_signal`.* [§4.5]
- **A6** Role primers + behavior-card consumption. *Agents read `assignment` /
  directives from the `BehaviorCard` in their obs/system prompt.* [§4.1, §4.3]

## Invariants you must keep green
- Output is **strict JSON, pydantic-validated** (`Action` + `Message`). The
  **env**, not the LLM, decides validity — on an illegal action the agent just
  loses the turn (it's logged `invalid_action`, a feature to show).
- **Execution-memory = HOW-TO only**, schema'd, cap 8, guard-filtered — **no
  coords, no seed-specific landmarks/numbers**. (`looks_seed_specific` + a test.)
- Keep `Agent.act(obs) -> Action`. The oracle path must keep working (offline
  fallback for green-main).
- Low temperature (`agents.temperature`/`llm.temperature`); log full
  prompts/outputs (Weave) for reproducibility.

## Integration touchpoints
- A3 needs `train/loop.py` to call agents async — coordinate with Stream 3 (loop
  owner). Provide `LLMWorker(agent_id, llm)` they can drop in.
- The 4-agent roster is `orca.cards.DEFAULT_ROSTER` (explorer/miner/tinkerer/
  support). Flip `run.single_agent_oracle: false` for the real run.
- New message fields = additive contract change → broadcast.

## Definition of done
4 LLM agents play a full episode on the real env, message over the bus, read
their cards, and write guard-filtered memory across episodes — with no crashes
and JSON validity enforced. Demoable: the comms-no-Orca baseline (they stall/
duplicate without a manager).

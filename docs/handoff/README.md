# Orca — Stream Handoff (read this first)

Phase 0 (the foundation) is **done, green, and on `main`**. The fork is clean, so
the three streams now run in parallel — **one friend per stream**:

| Stream | Owner reads | Owns folders |
|---|---|---|
| **1 — Env depth** | [STREAM_1_env.md](STREAM_1_env.md) | `env/`, `reward/` |
| **2 — Agents** | [STREAM_2_agents.md](STREAM_2_agents.md) | `agents/`, `bus/` |
| **3 — Orca + Eval** | [STREAM_3_orca_eval.md](STREAM_3_orca_eval.md) | `orca/`, `eval/`, `telemetry/` |

> **What Orca is (don't lose the plot):** a proof that a **manager LLM learns
> transferable delegation strategy** over a team of worker LLMs (Architecture
> **C2** — verbal-RL workers + one small delegation bandit). The demo lives or
> dies on **frozen contracts, no coordinate leakage, an objective DAG-frontier
> reward, Weave traces, baselines, and held-out-seed transfer.** **Do NOT** build
> PPO, per-step rewards, real Minecraft, coordinate memories, or a huge env.
> **Target is the dragon**, built up from the working floor — never be demo-less.

The authoritative specs are [`docs/ORCA_master_build_spec.md`](../ORCA_master_build_spec.md)
(the *what*) and [`docs/ORCA_workflow_execution_plan.md`](../ORCA_workflow_execution_plan.md)
(the *who/when/how*). These briefs point you at the right sections.

---

## Setup (everyone, once)

```bash
git clone https://github.com/ronoktanvir/Orca && cd Orca
uv venv --python 3.11 .venv          # or: python3.11 -m venv .venv
uv pip install -r requirements.txt   # pydantic, pyyaml, numpy, pytest, openai
cp .env.example .env                  # then paste the keys Ronok gives you (NOT in the repo)
.venv/bin/python run.py              # should reach "iron", team_reward 0.200
.venv/bin/python -m pytest -q        # should be all green
.venv/bin/python scripts/check_fork_gate.py   # GREEN
```

Keys live in `.env` (gitignored) — `OPENAI_API_KEY`, `WANDB_API_KEY` (Weave),
`WANDB_INFERENCE_API_KEY` (GLM). Also run `wandb login` once with the Weave key so
tracing never prompts. **Never commit a key** — `.gitignore` blocks `.env`.

---

## The 4 rules that keep the fork clean (non-negotiable)

1. **Don't change the 7 contracts** in `contracts/` — additive-only (new
   *optional* fields), and broadcast any change to the whole team. They are the
   interface every stream builds against.
2. **Edit only your own folders.** Need something from another folder? That's
   either a contract field or a coordinated `train/loop.py` edit (see below).
3. **Green-main rule.** Every merge to `main` must pass `python run.py` +
   `pytest -q` (which includes `obs_guard/coord_leak_test.py`). A red `main`
   blocks all three streams — fix forward or revert. Merge daily.
4. **No coordinate leakage, ever.** `serialize_observation` is the only obs path
   and must never read `Region.pos`; `coord_leak_test` must stay green.

---

## The LLM seam (Streams 2 & 3)

The shared client is built: `llm.build_llm(role, settings)` where role is
`"worker"` or `"orca"`. It returns an object with `.complete(prompt, schema=None)`.

```python
from config import load_config, load_dotenv
from llm import build_llm
load_dotenv()
settings = load_config()
worker_llm = build_llm("worker", settings)   # gpt-5-mini by default
text = worker_llm.complete(prompt, schema=MyPydanticModel)  # JSON mode when schema given
```

- Models (from `configs/default.yaml` → `llm:`): **workers = `gpt-5-mini`**,
  **Orca = `gpt-5`** (verified working). Set `llm.provider: wandb_inference` to
  run **GLM-5.1** on the W&B endpoint instead (alternate/ablation, $50 W&B credits).
- `complete()` returns **raw text**; JSON validation + the one-shot repair retry
  is the caller's job (§4.4 — that's Stream 2 A2).
- Budget: ~$150 OpenAI + $50 W&B, no rate limit. `T_max=120` keeps early
  episodes cheap; raise it as agents get competent. Worry about correctness, not $.

---

## Integration: `train/loop.py` is shared glue

It currently runs `ShallowOracle` → no-op Orca → log. As streams land, three edits
are needed there — **coordinate these (Stream 3 owns the loop, pairs with Stream 2):**

- **A3:** call the 4 agents **async/in parallel** each round (currently sequential).
- **A1/A3:** swap `ShallowOracle` → `LLMWorker` (keep the oracle as the §9 oracle/baseline).
- **O2/O6:** enable `orca.observe_outcome` (bandit), `coach`, `accept_gate`, seed
  rotation, and phasing (the calls are already wired as no-ops — flip them on).

Everything else is independent. Keep the oracle path runnable as the offline
fallback so `main` never goes dark.

---

## Checkpoints (whole-team syncs, spec §4)
Fork ✅ → **First real episode** (4 LLM agents) → **First Orca improvement**
(bandit curve up) → **Full-DAG run** (oracle→dragon) → **Transfer result**
(A/T2/T3→B/C plot) → **Freeze**.

Build toward the **minimum lovable demo** first (Orca learns transferable
delegation on a shallow-ish DAG, shown on held-out seeds, audited in Weave), then
push to the dragon.

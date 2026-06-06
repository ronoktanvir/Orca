# Orca — Phase 0 Foundation

**Orca is not "Minecraft."** Orca is a proof that a **manager LLM can learn
transferable delegation strategy** over a team of worker LLMs. The demo lives or
dies on **frozen contracts, no coordinate leakage, an objective DAG-frontier
reward, Weave traces, baselines, and held-out-seed transfer** — *not* on building
a big game. (Architecture **C2**: verbal-RL workers + one small delegation bandit;
no PPO, no per-step rewards, no real Minecraft, no coordinate memories.)

This repo is **Phase 0: the foundation** — a thin but *real* end-to-end loop where
every component is a **stub behind a frozen interface**, so the three parallel
streams (Env depth · Agents · Orca+Eval) can fork cleanly. See
[`docs/ORCA_master_build_spec.md`](docs/ORCA_master_build_spec.md) (the *what*)
and [`docs/ORCA_workflow_execution_plan.md`](docs/ORCA_workflow_execution_plan.md)
(the *who/when/how*).

> **Target is the dragon.** The shallow wood→stone→iron slice here is scaffolding
> to get the loop running — *not* the deliverable. Depth goes into the streams.

---

## Quickstart

```bash
# 1. Create the env (Python 3.11) and install deps
uv venv --python 3.11 .venv          # or: python3.11 -m venv .venv
uv pip install -r requirements.txt   # or: .venv/bin/pip install -r requirements.txt

# 2. Run one full episode end-to-end
.venv/bin/python run.py              # reaches "iron"; logs to runs/ (or Weave if creds)

# 3. Run the tests + the fork-gate checklist
.venv/bin/python -m pytest -q
.venv/bin/python scripts/check_fork_gate.py
```

`python run.py` reaches the **iron** milestone (team_reward `0.200`) in ~14 rounds
and emits an `EpisodeTrace` + `EpisodeMetrics`. Telemetry defaults to `auto`:
Weave **iff** a W&B credential is detectable, otherwise a local-JSONL fallback —
so it always runs offline and never hangs on a login prompt.

---

## What Phase 0 delivers (F1–F6)

| ID | Deliverable | Where |
|----|-------------|-------|
| F1 | Repo + deps + LLM-client interface | [`pyproject.toml`](pyproject.toml), [`llm/`](llm/) |
| F2 | **The 7 frozen contracts** | [`contracts/`](contracts/) |
| F3 | Stub env (~5 regions, wood→stone→iron, validity, **coord-free serializer**, frontier reward) | [`env/`](env/), [`reward/`](reward/) |
| F4 | Scripted placeholder agent (shallow oracle → iron) | [`agents/scripted.py`](agents/scripted.py) |
| F5 | Run loop (env → agent → no-op Orca → log) | [`train/loop.py`](train/loop.py), [`run.py`](run.py) |
| F6 | Weave wiring + safe fallback + config | [`telemetry/`](telemetry/), [`configs/default.yaml`](configs/default.yaml) |

### The seven frozen contracts ([`contracts/`](contracts/))

`Observation` · `Action` · `Message` · `BehaviorCard` · `ExecutionMemory` ·
`EpisodeTrace` · `EpisodeMetrics`. **Frozen after the fork** — changes are
additive-only (new optional fields) and must be broadcast to the team.

### The three coordinate-leak guards (§3.2)

1. **Contracts forbid extras** — every `Observation` sub-model is `extra="forbid"`,
   so a stray `pos` can't even be constructed.
2. **One obs path** — [`env/observation.serialize_observation`](env/observation.py)
   is the *only* place an `Observation` is built, and it never reads `Region.pos`.
3. **A scanner** — [`obs_guard/coord_leak_test.py`](obs_guard/coord_leak_test.py)
   asserts no float pair / no `pos` / no internal region id ever appears in a
   serialized observation (runs under `pytest`).

---

## Repo layout & folder ownership (the handoff)

The fork is clean because folder ownership is locked (workflow §2):

```
contracts/   the 7 frozen pydantic models (shared; owned by nobody)
env/         ← Stream 1   graph-on-plane world, seeds, tech tree, actions, coord-free obs
reward/      ← Stream 1   DAG frontier ladder + penalties
agents/      ← Stream 2   workers, prompts, memory + guard filter, scripted oracle
bus/         ← Stream 2   structured comm bus (t+1 delivery)
orca/        ← Stream 3   no-op Orca, delegation bandit, coach, accept-gate, cards
eval/        ← Stream 3   baselines, transfer, ablations, plots
telemetry/   ← Stream 3   Weave ops + safe fallback
train/       run loop, phasing, checkpoint (integration glue)
obs_guard/   the coordinate-leak invariant
llm/         swappable LLM client interface
configs/     default.yaml — every §15 tuning knob
run.py       entry point      scripts/check_fork_gate.py — the gate checker
```

After the fork, each stream **swaps the stub that lives in its folder** for the
real thing; nobody edits another stream's folder. Stubs are marked with the
stream/task that owns them (e.g. `LLMWorker` → Stream 2 A1).

---

## Fork gate — ✅ green

Verified by `python scripts/check_fork_gate.py`:

- [x] The 7 contracts are committed and **frozen**.
- [x] A shallow episode runs end-to-end and logs a nested trace (Weave or local).
- [x] `coord_leak_test` passes.
- [x] Folder ownership + config structure agreed.

**Do not keep polishing the foundation past this gate — depth goes into the
streams** (env→dragon, real LLM workers, the bandit + coach + accept-gate, the
transfer plot).

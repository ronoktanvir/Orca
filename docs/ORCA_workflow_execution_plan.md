# ORCA — Build Workflow & Execution Plan

**Companion to `ORCA_master_build_spec.md`.** That doc is *what* to build (architecture, schemas, reward, tech tree). **This** doc is *who / when / how* to build it concurrently — the foundation you do first, the fork point where you hand off, and the parallel streams afterward.

> **Target is the dragon.** The shallow version in Phase 0 is *scaffolding to get the loop running*, not the deliverable. Production target = the full DAG through `dragon defeated` (build spec §7.1), then the speed phase. Do not stop at the shallow game.

---

## 0. The core execution model

```
TIME ───────────────────────────────────────────────────────────────────►

   ┌──────────── PHASE 0: FOUNDATION ────────────┐   FORK    ┌── PARALLEL STREAMS ──┐
   │  SOLO · ~6–8h                                │    │      │  Stream 1: Env depth │
   │  contracts + run loop + stub env + Weave     │    │      │  Stream 2: Agents    │
   │  (every component is a STUB behind a         │    │      │  Stream 3: Orca+Eval │
   │   frozen interface)                          │    ▼      └──────────────────────┘
   ════════════════════════════════════════════════╪═══════════════════════════════►
   parallelizing HERE just creates conflicts      hand off   near-zero collisions
   and interface churn → do it solo               the stubs   (each owns its folders)
```

**The one idea:** build a skeleton where every part is a stub behind a frozen contract. Once a thin episode runs end-to-end, each person takes **one stub** and makes it real **in their own folder**. Because components talk only through the frozen pydantic models + the run loop, the streams never touch each other's code.

**Why not parallelize the foundation:** before the contracts exist, parallel work = three people guessing at interfaces, building throwaway mocks, and merge-conflicting on the same core files. The foundation is small and tightly coupled — it's one person's job.

---

## 1. Phase 0 — The Foundation (SOLO, ~6–8h, do this first)

Goal: a shallow but **real** end-to-end loop that runs one episode and logs to Weave, with all interfaces frozen. Build spec refs in brackets.

| ID | Task | Done when (acceptance) | Spec |
|---|---|---|---|
| F1 | Repo + deps (`pydantic`, `weave`, `wandb`, `asyncio`, LLM client behind `llm.complete`) | `python run.py` executes a no-op episode | §11 |
| F2 | **Contracts** — the 7 pydantic models | all import + validate sample instances; **committed and frozen** | §11 |
| F3 | **Stub env** — ~5 regions, shallow DAG (wood→stone→iron), `move/gather/craft` + validity, **coord-free serializer**, frontier reward | `reset()/step()` return valid `Observation`, accept `Action`, compute frontier; **coord-leak test passes** | §3.1–3.3, §3.6, §7.1 |
| F4 | **Placeholder agent** — scripted policy (shallow oracle) | deterministically reaches "iron" on the stub seed | §12 |
| F5 | **Run loop** — env → agent → orca(no-op) → log | one full episode runs end-to-end; emits `EpisodeTrace` + `EpisodeMetrics` | §8 |
| F6 | **Weave wiring + config skeleton** | trace tree visible in Weave; `configs/default.yaml` loads knobs | §10, §15 |

### Fork gate (the "give it off" moment) — ALL must be true:
- [ ] The 7 contracts are committed and **frozen**.
- [ ] A shallow episode runs end-to-end and logs a nested trace to Weave.
- [ ] `coord_leak_test` passes.
- [ ] Folder ownership + config structure agreed.

When these are checked, hand off. **Do not** keep polishing the foundation past this gate — depth goes into the streams.

---

## 2. The handoff — freeze these or you'll get merge pain

The fork is clean **because** these are locked. Changing any of them after the fork requires a broadcast to the whole team:

1. **The pydantic contracts** (`Observation`, `Action`, `Message`, `BehaviorCard`, `ExecutionMemory`, `EpisodeTrace`, `EpisodeMetrics`).
2. **The run-loop signature** (how env/agents/orca plug in).
3. **Folder ownership** (table below).
4. **`configs/default.yaml` structure.**

**The handoff itself:** each stream takes the stub that lives in its folder and swaps it for the real thing. Nobody edits another stream's folder.

| Stream | Owns folders | Replaces stub |
|---|---|---|
| 1 — Env depth | `env/`, `reward/` | the shallow stub env |
| 2 — Agents | `agents/`, `bus/` | the scripted placeholder agent |
| 3 — Orca + Eval | `orca/`, `eval/`, `telemetry/` | the no-op Orca |

---

## 3. The three parallel streams

Each task lists its **done-when** and the spec section it implements. ▶ = "demoable milestone."

### Stream 1 — Env Depth (`env/`, `reward/`)
- **E1** Full tech tree + recipes — *all DAG prerequisites craftable; invalid actions rejected + logged.* [§3.4]
- **E2** Graph-on-plane world + movement by heading + region reveal — *`move(dir)` reveals regions; never emits coords (test).* [§3.1]
- **E3** Seed generator + validator + **full-DAG oracle** — *oracle solves seed A to the dragon; B/C generated and validated winnable.* [§3.7] ▶ *proves the game is beatable*
- **E4** Stochasticity — gather/fight probs, day/night, hunger, death/respawn — *deterministic under seed; logged.* [§3.5]
- **E5** Cooperation mechanics — co-location, `give_item`, superadditive fight formula — *solo blaze p≈0.2, trio p≈0.85; `give_item` needs same region.* [§3.5]
- **E6** Reward — full frontier ladder + penalties + speed phase — *frontier matches milestones; speed bonus only post-win.* [§7] ▶ *full reward to `1.00 dragon defeated`*

### Stream 2 — Agents (`agents/`, `bus/`)
- **A1** Worker turn loop + prompt builder — *1 LLM agent plays a full episode on the real env.* [§4.2–4.3] ▶
- **A2** JSON parse/validate + invalid handling — *malformed output → repair/wait; no crashes.* [§4.4]
- **A3** Scale to 4 agents, async parallel calls — *4 agents act per round in parallel.* [§3.6] ▶ *the no-Orca baseline (they duplicate work)*
- **A4** Comm bus + message schema + delivery + history summarization — *messages logged, delivered t+1, history compacted.* [§5]
- **A5** Execution-memory — schema + write + **guard filter** + `learning_signal` modulation — *memory persists across episodes; filter strips coord-like content (test).* [§4.5]
- **A6** Role primers + behavior-card consumption — *agents read assignment from card.* [§4.1, §4.3]

### Stream 3 — Orca + Eval (`orca/`, `eval/`, `telemetry/`)
- **O1** Trace digest from `EpisodeTrace` — *compact per-agent + team summary.* [§6.1] *(can start immediately on foundation traces)*
- **O2** Delegation **bandit** — situations, arms, ε-greedy/Thompson, per-episode update — *arm values update; chosen-arm frontier curve plotted.* [§6.3] ▶ *the learning curve*
- **O3** Scoring — `performance_score` + `learning_signal` (mostly objective) — *scores computed from env stats.* [§7.3]
- **O4** Verbal coach → behavior-cards + credit reasoning — *cards updated with readable rationale, logged.* [§6.4]
- **O5** Accept/reject gate — *update kept iff eval-pool frontier not regressed; else rollback.* [§6.5]
- **O6** Phasing controller — *Phase 0 freezes cards; Phase 2 enables speed post-win.* [§6.6]
- **O7** Eval harness — baselines, transfer (A/T2/T3 → B/C), ablations, plots — *transfer bar chart + bandit curve + 1 ablation.* [§9] ▶ *the headline result*
- **O8** Weave Evaluation + leaderboard + the failure→fix→improve trace — *comparison view captured for the pitch.* [§10] ▶

---

## 4. Cross-stream dependencies & integration cadence

Most work is independent. The few real touchpoints:

- **Agents need an env** → they start on the **foundation stub env** immediately (A1), and pick up Env-depth features (E1–E6) as they land. They are never blocked.
- **Orca needs real traces** → O1–O3 develop against **foundation/stub traces** right away; O4–O7 need real multi-agent episodes (after A3). Never blocked, just richer over time.
- **Cooperation** is a joint result of E5 (env superadditive) + Agents choosing to co-locate + O2/O4 (Orca learning to exploit it). Integrate and tune together at the cooperation checkpoint.

**Integration discipline:** each person owns their folders; **merge to `main` at least daily**; every merge must pass a **smoke test** (`run.py` completes one episode + `coord_leak_test`). Keep a green `main` at all times.

**Named checkpoints (whole team syncs):**
1. **Fork** — foundation done, contracts frozen.
2. **First real episode** — 4 LLM agents on the (stub-or-real) env.
3. **First Orca improvement** — bandit curve moves across episodes.
4. **Full-DAG run** — env depth done; oracle (then agents) reach the dragon.
5. **Transfer result** — A/T2/T3 → B/C plot in hand.
6. **Freeze** (~H27) — features locked; only polish + pitch + backup video.

---

## 5. Scaling by team size

| Phase 0 (foundation) | Then… | 1 person | 2 people | 3 people |
|---|---|---|---|---|
| **One person** does it (pair on contracts if ≥2, so everyone buys in) | fork | do streams in order: Env → Agents → Orca | P1: Env depth · P2: Agents+Orca | P1: Env · P2: Agents · P3: Orca+Eval |

The foundation is **never** split across people — too small, too coupled. After the fork, the back-loaded stream is **Orca+Eval** (it carries eval, plots, pitch), so once Env depth is solid (~checkpoint 4) the Env owner shifts to help Orca+Eval.

---

## 6. Whole-project build order (always keep `main` runnable)

Each layer is a working system and a fallback demo. Streams contribute to layers in parallel, but the *integration* order is:

1. Skeleton (F1–F6) → 2. Tiny env + oracle (F3–F4) → 3. One LLM agent (A1) → 4. Four agents + bus (A3–A4) → 5. **Orca bandit + gate (O2,O5)** → 6. Memory + guard (A5) → 7. Held-out seeds + transfer + ablation (E3,O7) → 8. Cooperation depth + full DAG to dragon (E5–E6) + pitch (O8).

**Minimum lovable demo (the floor):** layers 1–7 on a shallow-ish DAG = "Orca learns transferable delegation, shown on held-out seeds, audited in Weave." **Full target:** layer 8 = the dragon + speed phase. Build toward the floor first so you're never demo-less, then push to the dragon.

---

## 7. Demo-readiness = the 5 success-ladder items (build spec §1)

| # | Success criterion | Owned by | Done when |
|---|---|---|---|
| 1 | Static-role baseline performs poorly | O7 | baseline run stalls mid-DAG |
| 2 | 4 agents communicate but stall/duplicate without Orca | A3–A4, O7 | comms-no-Orca baseline recorded |
| 3 | Orca scores, rewrites cards, **improves** future runs | O2,O4,O5 | bandit curve up + frontier rises across episodes |
| 4 | Improvement **transfers** to held-out B/C | E3, O7 | transfer plot: Full C2 > baselines on B/C |
| 5 | Every decision auditable in Weave | O8, telemetry | the failure→fix→improve trace captured |

---

## 8. Concurrency risks & unblocking

- **Foundation slips → everyone waits.** Keep it minimal (shallow DAG, 5 regions, scripted agent). Resist gold-plating; depth is the streams' job.
- **Contract churn after fork → merge hell.** Freeze the 7 models; any change is broadcast + a shared 5-min sync. Add *new* optional fields rather than changing existing ones.
- **Orca+Eval is back-loaded.** Start O1–O3 against foundation traces on day one; reassign the Env owner to help with eval/plots after checkpoint 4.
- **Green-main rule.** A broken `main` blocks all three streams. Every merge passes the smoke test or it's reverted.

---

## 9. Flat task board (paste into Linear / Notion / a whiteboard)

**Phase 0 (solo):**
- [ ] F1 repo + deps  ·  - [ ] F2 contracts (FREEZE)  ·  - [ ] F3 stub env + coord-leak test  ·  - [ ] F4 placeholder agent  ·  - [ ] F5 run loop  ·  - [ ] F6 Weave + config
- [ ] **FORK GATE** met → hand off

**Stream 1 — Env depth:**
- [ ] E1 tech tree + recipes  ·  - [ ] E2 graph-on-plane + movement  ·  - [ ] E3 seeds + validator + full oracle  ·  - [ ] E4 stochasticity/day-night/death  ·  - [ ] E5 cooperation mechanics  ·  - [ ] E6 full reward + speed phase

**Stream 2 — Agents:**
- [ ] A1 worker loop + prompts  ·  - [ ] A2 parse/validate  ·  - [ ] A3 4 agents async  ·  - [ ] A4 comm bus + history  ·  - [ ] A5 memory + guard filter  ·  - [ ] A6 role primers + card consumption

**Stream 3 — Orca + Eval:**
- [ ] O1 trace digest  ·  - [ ] O2 delegation bandit  ·  - [ ] O3 scoring  ·  - [ ] O4 verbal coach + credit  ·  - [ ] O5 accept/reject gate  ·  - [ ] O6 phasing  ·  - [ ] O7 eval/transfer/ablations/plots  ·  - [ ] O8 Weave dashboard + pitch trace

**Checkpoints:** - [ ] Fork  ·  - [ ] First real episode  ·  - [ ] First Orca improvement  ·  - [ ] Full-DAG run  ·  - [ ] Transfer result  ·  - [ ] Freeze (~H27)

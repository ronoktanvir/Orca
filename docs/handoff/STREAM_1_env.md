# Stream 1 — Env Depth (in-depth brief)

> Paste-ready kickoff for the Env Depth coding agent. Read alongside
> [handoff/README.md](README.md) and build spec **§3 (env)** + **§7 (reward)**.

## You are the Env Depth builder for Orca.

**What Orca is (read twice):** Orca is a *proof that a manager LLM learns
transferable delegation strategy* over a team of worker LLMs (Architecture C2 —
verbal-RL workers + one small delegation bandit). The demo lives or dies on
**frozen contracts, no coordinate leakage, an objective DAG-frontier reward,
Weave traces, baselines, and held-out-seed transfer.** Your env exists to make
that *provable and winnable* — it is **not** a game to gold-plate. **Do NOT
build:** PPO, per-step reward shaping, real Minecraft, coordinate-based memories,
or coordinate leakage. **Target is the dragon**, grown up from the working
shallow env so `main` is never demo-less.

**The repo you're inheriting:** Phase 0 is done and green on `main`
(https://github.com/ronoktanvir/Orca): a thin but real end-to-end loop — a
5-region stub env, a scripted oracle that reaches "iron," a no-op Orca, the run
loop, Weave logging, 62 passing tests. Everything talks through the 7 frozen
pydantic contracts in `contracts/`. Your job: deepen the env from the shallow
`wood→stone→iron` slice all the way to `dragon_defeated`.

## Setup (once)
```bash
git clone https://github.com/ronoktanvir/Orca && cd Orca
uv venv --python 3.11 .venv && uv pip install -r requirements.txt
cp .env.example .env          # paste the keys Ronok DMs you (never commit them)
.venv/bin/python run.py       # must reach "iron", team_reward 0.200
.venv/bin/python -m pytest -q # all green
git checkout -b stream1-env
```

## THE LAWS (breaking these breaks the whole team)
1. **Never change the 7 contracts** in `contracts/` except additively (new
   *optional* fields) + broadcast. Everyone builds against them.
2. **Edit only `env/` and `reward/`.** Loop changes → PR tagged to Stream 3
   (they own `train/loop.py`).
3. **Green-main:** every merge passes `python run.py` + `pytest -q` (includes
   `obs_guard/coord_leak_test.py`). Merge to `main` daily.
4. **No coordinate leakage, ever.** `env/observation.serialize_observation` is
   the ONLY place an `Observation` is built and it must NEVER read `Region.pos`.
   Compass bearings + distance *bands* only — never numbers, never region ids.

## The interface you must preserve (so Agents + the loop don't break)
- `StubEnv(seed, episode_idx, agents=[(id,Role)], *, t_max, day_length,
  message_window, stop_at_milestone, behavior_cards)` with `.reset()→{id:Observation}`,
  `.observe(id)→Observation`, `.step({id:Action})→StepResult(records, messages,
  milestone_events)`, `.done`, `.terminated_reason`, `.round_idx`, `.frontier`
  (a `Milestone`), `.world`, `.all_records`, `.all_messages`, `.milestone_timeline`.
- `serialize_observation(...)` signature + the coord-free guarantee.
- `reward_computer(trace, *, agent_roles, weights, baseline_steps)→EpisodeMetrics`.

If you must change a signature, update `train/loop.py` in the **same PR** and tell
the team.

## Your files & current stubs
`env/world.py` (Region w/ hidden `pos`, World coord-free perception),
`env/seeds.py` (`make_world`, `TRAIN_SEEDS=("A","T2","T3")`,
`HELDOUT_SEEDS=("B","C")`), `env/techtree.py` (recipes/gates/`detect_milestone`),
`env/actions.py` (`resolve_action`), `env/observation.py`, `env/rng.py`
(`make_rng` — sha256-seeded, process-stable), `env/stub_env.py`, `reward/dag.py`
(full `MILESTONE_VALUE` ladder to 1.0 already authored), `reward/reward.py`.

## Your tasks (do in order; each ends green on main)

**E1 — Full tech tree + recipes (§3.4).** Extend `env/techtree.py` to the whole
tree: smelting (furnace+coal: iron_ore→iron_ingot; raw food→cooked), iron
tools/shield/bucket/flint_and_steel, diamond (iron-pickaxe gate), obsidian (two
routes: diamond-pickaxe on cooled lava OR bucket+lava_pool water-trick — reward
the clever route), portal frame (10 obsidian + flint_and_steel → lit
nether_portal), blaze_rod→blaze_powder, ender_pearl, eye_of_ender (~12),
end_portal. Add `smelt`/`place` resolution in `env/actions.py`. **Done when:**
every DAG prerequisite is craftable; illegal crafts/smelts are rejected with a
reason and logged as `invalid_action`.

**E2 — Graph-on-plane world + movement + reveal + sub-worlds (§3.1).** Replace
the 5-region layout with a real region graph on the hidden plane: 20–40 Overworld
+ 8–15 Nether nodes + an End node. `move(dir)` walks the best-aligned heading and
reveals the next region; `scout` reveals adjacent detail. Nether via built+lit
portal → fixed per-seed entry node; End via activated end_portal in the
stronghold. Structures (fortress/stronghold) discoverable; emit *biome-type* soft
hints only. **Done when:** repeated `move(N)` walks an axis, regions reveal, and
`coord_leak_test` is still green with the richer obs.

**E3 — Seed generator + validator + full-DAG oracle (§3.7).** ▶ *Proves the game
is beatable.* Generator places regions/abundances/structures for `{A,T2,T3,B,C}`
with **guaranteed reachability** of every DAG prerequisite (validate at gen
time). Write a scripted **oracle** (extend the `agents/scripted.py` pattern but
keep the shallow one) that solves seed A to `dragon_defeated`. Generate B/C and
assert winnable. **Done when:** oracle reaches the dragon on A; B/C validated
winnable; a greedy/myopic agent stalls mid-DAG (that gap is the whole result).
**Never train on B/C.**

**E4 — Stochasticity, day/night, hunger, death/respawn (§3.5).** Gather/fight
yields ~Binomial via `make_rng(seed, episode_idx, round, agent_id)` (deterministic
+ logged). `day_length` from config (night → hostile mobs, riskier;
`sleep`/bed skips night if safe). Hunger drains (faster moving/fighting); 0 hunger
→ health drain. Death at health 0 → respawn at start after `respawn_cost`,
dropping non-equipped inventory; death is a team penalty. **Done when:** identical
seeds reproduce identical episodes, across processes.

**E5 — Cooperation mechanics (§3.5).** Co-location (`SAME_REGION`) gates
`give_item` and `regroup`. Superadditive fights for blaze/dragon:
`p_success = logistic(a·(n_colocated−1) + b·combined_gear + c·avg_health −
d·difficulty)`. **Done when:** solo blaze p≈0.2, trio p≈0.85; `give_item` requires
same region; inventory stays per-agent (no global stash).

**E6 — Full reward + speed phase (§7).** Confirm `reward/dag.py` ladder matches
milestones (already authored to 1.0); wire real milestone detection for the deep
tree in `detect_milestone`. Penalties (death/invalid/idle) small per §7.2. **Speed
phase:** `speed_bonus = max(0,(baseline_steps−current_steps)/baseline_steps)` —
**only post-first-win**, never in Phase 0/1. **Done when:** frontier value tracks
the deepest milestone (max-frontier, once/episode); speed bonus gated on a win.

## Tests you must add
For every new action/recipe: a validity-rejection test. A determinism test (same
seed → same trace, cross-process). Extend `obs_guard/coord_leak_test.py` for any
new obs field. An oracle-reaches-dragon test on A. B/C winnable-validation tests.

## Workflow
Branch `stream1-env`, small PRs, merge to `main` ≥ daily, each green. When E3
lands (full-DAG oracle), ping the team — then help Stream 3 with eval/plots
(spec §5).

## Definition of done
Oracle → `dragon_defeated` on A; B/C validated winnable; greedy stalls mid-DAG;
full reward ladder to 1.00 with post-win speed phase; `coord_leak_test` green.
Demoable milestone: the full-DAG oracle run.

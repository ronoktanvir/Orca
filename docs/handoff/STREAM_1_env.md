# Stream 1 — Env Depth

> Read [handoff/README.md](README.md) first (framing, setup, the 4 rules).
> Orca is a proof that a manager LLM learns *transferable delegation* — the env
> exists to make that provable, not to be a game. **Don't build** PPO, per-step
> rewards, real Minecraft, or coordinate leakage. **Target = the dragon**, built
> up from the working shallow env so `main` is never demo-less.

## You own
`env/`, `reward/`. **Don't edit** other streams' folders or the 7 contracts in
`contracts/` (additive-only, broadcast first).

## What you replace
The shallow stub. Today it's real but tiny — keep its **interface** stable (or
update `train/loop.py` in lockstep, coordinated):

- `env/stub_env.py` — `StubEnv(reset/observe/step/done/frontier/terminated_reason)`.
- `env/world.py` — `Region` (hidden `pos`), `World` coord-free perception helpers.
- `env/seeds.py` — 5-region hand-placed seed A; `make_world(seed)`.
- `env/techtree.py` — wood→stone→iron recipes/gates + `detect_milestone`.
- `env/actions.py` — move/scout/gather/craft/wait/report + validity rejection.
- `env/observation.py` — `serialize_observation` (the ONLY obs path; never reads `pos`).
- `reward/dag.py` — full milestone→value ladder (already complete to the dragon).
- `reward/reward.py` — `reward_computer(trace, ...)` → `EpisodeMetrics`.

## Tasks (done-when) — build spec §3, §7
- **E1** Full tech tree + recipes (wood→…→eyes of ender→dragon). *All DAG
  prerequisites craftable; invalid actions rejected + logged.* [§3.4]
- **E2** Graph-on-plane world + movement by heading + region reveal + Nether/End
  subgraphs + structures. *`move(dir)` reveals regions; coords never emitted
  (coord_leak_test green).* [§3.1]
- **E3** Seed generator + validator + **full-DAG oracle**. *Oracle solves seed A
  to the dragon; B/C generated and validated winnable.* ▶ proves the game is
  beatable. [§3.7] (extend `scripts/`/oracle; keep a shallow oracle too)
- **E4** Stochasticity — gather/fight probs, day/night, hunger, death/respawn.
  *Deterministic under seed (use `env/rng.make_rng`); logged.* [§3.5]
- **E5** Cooperation — co-location, `give_item`, **superadditive** fight formula.
  *Solo blaze p≈0.2, trio p≈0.85; `give_item` needs SAME_REGION.* [§3.5]
- **E6** Reward — full frontier ladder + penalties + **speed phase** (post-win
  only). *Frontier matches milestones; speed bonus never before a win.* ▶ [§7]

## Invariants you must keep green
- `serialize_observation` is the **only** obs builder and **never** touches
  `Region.pos`; no float pair / `pos` / internal region id ever reaches an
  Observation. (`pytest`/`coord_leak_test` enforces this — add cases as you add
  obs fields.)
- Determinism: every stochastic draw goes through `make_rng(seed, ep, round,
  agent)` (sha256-seeded, process-stable). Same inputs → same outcome.
- Reward is **max-frontier** (deepest milestone), computed **once per episode**;
  penalties small (§7.2). Don't add per-step shaping.
- Keep the `StubEnv` method surface (or PR a `train/loop.py` change with it).

## Integration touchpoints
- New action durations / `busy` semantics: the loop queries only `free` agents —
  preserve that contract.
- Adding obs fields = additive contract change → broadcast.
- New seeds: register in `env/seeds.py` (`TRAIN_SEEDS`/`HELDOUT_SEEDS`); **never
  train on B/C**.

## Definition of done
Oracle reaches `dragon_defeated` on seed A; B/C validated winnable; a greedy
agent stalls mid-DAG (that gap is the result); `coord_leak_test` green; full
reward ladder to `1.00`. Demoable milestone: the full-DAG oracle run.

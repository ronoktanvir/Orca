# ORCA — Master Build Spec (Architecture C2)

**WeaveHacks 4 · hierarchical cooperative multi-agent system · Minecraft-lite**
**Version 1.0 — single source of truth. Any model/engineer should be able to build from this doc alone.**

---

## 0. Read-me-first: what this is and the locked decisions

We are building **Orca**: a manager LLM that *learns to delegate, coach, and coordinate* a team of 4 worker LLMs playing an abstract Minecraft-lite game, and we *prove* it learned transferable strategy (not memorized terrain) via held-out seeds. We are **NOT** training on real Minecraft and **NOT** doing gradient MARL on the workers.

**Architecture = C2 (Hybrid):** workers are strong LLMs that improve *verbally* (self-written execution memory); Orca is the one component with a real learned signal — a small **delegation bandit** — plus natural-language coaching. Learning lives at two decoupled levels, which kills the coupled-nonstationarity problem.

### Locked decisions (do not relitigate without updating this table)

| # | Decision | Choice | Why |
|---|---|---|---|
| D1 | Learning paradigm | **Verbal RL + 1 small bandit** (no gradient MARL, no backprop on LLMs) | Feasible in 30h; fits "orchestration" theme; keeps our novel ideas |
| D2 | Orca's mechanism | **Verbal coaching + discrete delegation bandit** | Readable feedback AND a quantitative learning curve |
| D3 | Worker infra | **Strong hosted LLM, macro-action granularity**, model swappable via config | Quality (money is not a constraint); macro-actions keep episode throughput high in 14h |
| D4 | World model | **Region graph embedded in a hidden 2D plane** | "Explore +X in a straight line" works; never emits coordinates |
| D5 | Persistent state | **Two artifacts:** Orca behavior-card (per agent) + agent execution-memory (per agent); both schema'd + guard-filtered + accept-gated | Two-level learning; agents refine HOW-TO, Orca owns delegation/coaching |
| D6 | Reward | **One episode-level team scalar** = max DAG frontier − penalties; speed bonus only after first win | Nothing does per-step gradient → no per-step shaping needed |
| D7 | Seeds | **5 total: 3 train (rotating) + 2 held-out (B/C)** | Transfer eval; resolves the 8-vs-3 contradiction |
| D8 | Cooperation | **Encoded in env dynamics** (superadditive fights + co-location), not hoped for | Agents only cooperate if it's the dominant strategy |

### Key reframes that resolve earlier confusion
- **No per-step reward shaping.** In C2 the only learner that consumes a scalar is Orca's bandit, which updates **once per episode** with the team frontier score. Workers learn from Orca's *words*, not a gradient. So "potential-based shaping" (which mattered only for the abandoned gradient-MARL path) is **dropped**.
- **Two reward scalars, two jobs.** `performance_score ∈ [0,1]` = how well things went (outcome). `learning_signal ∈ [−1,1]` = Orca's *"how hard should you adopt this lesson"* dial that **scales how much an agent edits its memory** (+1 bake it in, 0 ignore, −1 reverse).
- **Credit assignment (delegation vs execution)** is done by **Orca reasoning in natural language over the trace** (readable in Weave), not by counterfactual rollouts.

---

## 1. Hypothesis & success criteria

**Hypothesis:** A manager that reviews full episode traces, scores/coaches workers, and learns *who to assign to what* will (a) beat a static-role baseline and a no-Orca comms baseline, and (b) carry its improvement to **unseen seeds** better than a memorized/static policy.

**Demo success ladder (build toward all five):**
1. Static-role baseline performs poorly (stalls / duplicates work).
2. 4 agents that communicate but have no Orca still duplicate work or stall.
3. Orca reviews traces, scores agents, rewrites behavior-cards, and **measurably improves** future runs (the bandit curve goes up; frontier rises).
4. Orca's improved strategy **transfers** to held-out seeds B/C better than baselines (the headline plot).
5. Every decision is auditable in **Weave**, including a trace where Orca spots a failure → changes a policy → next run improves.

**Floor vs north-star:** target the **dragon** (the env is tunable so it's reachable), but the *guaranteed* demo result is "full system reaches deeper DAG frontier than baselines and transfers." Never be demo-less.

**Judging-axis mapping:** harness sophistication = the hierarchy + comms + memory discipline + accept-gate (spend polish here); technical execution = the bandit curve + transfer + ablations + baseline; creativity = strategy-not-coords memory + learning_signal dial; sponsor = Weave everywhere; utility = "a manager that learns to run an agent team" generalizes beyond Minecraft.

---

## 2. System architecture

```
                          ┌──────────────────────────────────────────────┐
                          │                  ORCA  (between episodes)      │
                          │  • reads summarized trace + objective metrics  │
                          │  • delegation BANDIT  → who does what          │
                          │  • verbal COACH       → behavior-cards + scores│
                          │  • accept/reject gate → keep update iff eval ↑ │
                          └───────────────▲───────────────┬───────────────┘
            behavior-cards, role assignment │             │ performance_score, learning_signal,
            (read at episode start)         │             │ verbal feedback  → agent execution-memory
                                            │             ▼
   ┌───────────────────────── EPISODE (within-run, turn-based rounds) ─────────────────────────┐
   │                                                                                            │
   │   Worker 1        Worker 2        Worker 3        Worker 4      ← strong LLMs, soft roles   │
   │   (Explorer)      (Miner)         (Tinkerer)      (Support)                                 │
   │      │  obs/act      │               │               │                                     │
   │      └──────┬────────┴───────┬───────┴───────┬───────┘                                     │
   │             ▼                ▼               ▼                                              │
   │      ┌────────────┐   ┌──────────────┐  ┌──────────────────┐                               │
   │      │ COMM BUS   │   │  ENVIRONMENT │  │  REWARD computer  │                              │
   │      │ (structured│◄─►│ graph-on-    │─►│  DAG frontier,    │                              │
   │      │  messages) │   │ plane world  │  │  penalties, speed │                              │
   │      └────────────┘   └──────────────┘  └──────────────────┘                               │
   │                                                                                            │
   └────────────────────────────── all events logged to WEAVE (@weave.op) ──────────────────────┘
```

**Data flow per episode:** Orca emits config (role assignment + behavior-cards) → run episode (workers act over rounds, message via bus, env enforces validity) → reward computer outputs team frontier + per-agent objective stats → Orca scores/coaches, updates bandit + cards → accept-gate re-evaluates on seed pool → keep or roll back. Repeat.

---

## 3. Environment spec (the biggest build item)

### 3.1 World model — graph embedded in a hidden 2D plane

The world is a set of **region nodes**. Each region has a HIDDEN `(x, y)` position used **only by the env** to compute bearings and distances. **Agents never receive `(x, y)`.**

```python
@dataclass
class Region:
    id: str                      # internal only, e.g. "r_07" (never shown verbatim as a landmark name)
    biome: Biome                 # forest, plains, mountains, caves, desert, swamp, ocean,
                                 # nether_wastes, soul_sand_valley, basalt_delta, warped_forest,
                                 # stronghold, end
    pos: tuple[float, float]     # HIDDEN. env-only. never emitted.
    resources: dict[Resource,float]  # abundance multipliers, e.g. {IRON_ORE: 0.8, COAL: 0.5}
    structure: Structure | None  # FORTRESS | STRONGHOLD | None
    layer: Layer                 # OVERWORLD | NETHER | END
```

- **Edges** are derived from positions: two regions are adjacent if within a distance threshold. Each edge exposes a **compass bearing** (N/NE/E/.../NW, 8-way) and a **distance band** (`ADJACENT | NEAR | FAR`), never a number.
- **`move(direction)`** = the agent travels in a compass heading. The env finds the discovered-or-frontier region best aligned with that heading and moves the agent there, revealing it. Repeatedly moving the same heading = "explore in a straight line along an axis" — a *transferable* strategy that does not depend on coordinates. ✅ This is exactly the behavior you wanted to allow.
- **Nether** is a separate sub-graph (own region nodes). Entered via a built+lit portal in any Overworld region; you arrive at a fixed-per-seed Nether entry node. **End** is a single region reached via activated End portal in the stronghold.
- **Structures** (fortress in Nether, stronghold in Overworld) are special nodes discoverable by exploration. The env may emit **soft hints** tied to biome ("fortress-like structures tend to appear in nether_wastes/soul_sand_valley") — hint is about biome *type*, never location.

### 3.2 Observation schema (exactly what a worker sees each turn)

**No coordinates. No region count. No global map.** Egocentric + relative only.

```json
{
  "round": 42,
  "time_of_day": "night",            // day | dusk | night | dawn  (from round % day_length)
  "self": {
    "role": "miner",                  // current assignment (soft)
    "health": 0.6, "hunger": 0.4,
    "inventory": {"cobblestone": 12, "iron_ore": 3, "wooden_pickaxe": 1},
    "status": "free",                 // free | busy(action,rounds_left)
    "current_biome": "mountains",
    "layer": "overworld"
  },
  "here": {                            // current region, what's perceivable
    "resources_visible": ["coal", "iron_ore"],
    "structure": null,
    "mobs": ["zombie"],               // present this round (time-dependent)
    "exits": [                         // discovered/adjacent directions
      {"dir": "N",  "distance_band": "NEAR", "biome_hint": "caves"},
      {"dir": "SE", "distance_band": "FAR",  "biome_hint": "unknown"}
    ],
    "frontier_dirs": ["E", "W"]        // unexplored headings you can move toward
  },
  "teammates": [                       // relative only
    {"agent": "agent_1", "distance_band": "NEAR", "bearing": "NW", "role": "explorer"},
    {"agent": "agent_3", "distance_band": "SAME_REGION", "bearing": null, "role": "tinkerer"}
  ],
  "known_landmarks": [                 // ABSTRACT, transferable, never coordinates
    {"type": "lava_pool", "rel_dir": "S", "distance_band": "FAR"},
    {"type": "village_food_source", "rel_dir": "N", "distance_band": "NEAR"}
  ],
  "recent_messages": [ /* last K bus messages visible to this agent */ ],
  "assignment": "Mine iron until you have 6 ingots, then regroup with tinkerer.",  // from behavior-card
  "dag_frontier_reached": "iron_acquired"   // team progress so far (shared signal)
}
```

> ⚠️ **Coordinate-leak guard (build this as a hard invariant):** a single function `serialize_observation()` is the ONLY place obs is built, and it has no access to `Region.pos`. Add a unit test asserting no float pair / no `pos` / no internal region id ever appears in a serialized observation. This invariant is also what protects the memory rule (§4.5).

### 3.3 Action space (macro-actions; env resolves over rounds)

Each macro-action has a **duration in rounds** (agent is `busy` until done; busy agents auto-emit `WAIT`) and a **success model**. The LLM picks one macro-action per free turn.

| Action | Args | Duration | Effect / success model |
|---|---|---|---|
| `move` | direction | 1–3 (by distance band) | Relocate toward heading; reveals next region |
| `scout` | — | 1 | Reveal more detail of current+adjacent regions (resources, hints, structure) |
| `gather` | resource | 2–5 | If tool-gate met & resource present: yield `~Binom(n,p)`; else **invalid** |
| `craft` | item | 1 | If recipe inputs present (+ crafting_table if required): consume→produce; else **invalid** |
| `smelt` | item | 8–12 | Needs furnace + fuel (coal); converts ore→ingot / raw→cooked food |
| `place` | item | 1 | Place block (e.g., portal frame obsidian); validity checked |
| `fight` | target | 2–6 | Stochastic; **superadditive** for blaze/dragon (§3.5); risk of damage/death |
| `eat` | food | 1 | Restore hunger if cooked food present |
| `sleep` | — | 2 | Skip to day, restore some health, **only if safe** (no hostile mobs in region) |
| `give_item` | agent,item,n | 1 | Transfer; **requires teammate in SAME_REGION** else invalid |
| `request_help` | task | 1 | Emit structured help message to bus |
| `regroup` | agent | 1–3 | Move toward a teammate's current region (uses their bearing/distance band) |
| `report` | status | 1 | Emit status message to bus (for Orca's trace + teammates) |
| `wait` | — | 1 | No-op (auto for busy agents) |

**Validity enforcement (core feature #6 of your spec):** the env validates every action against world state. Invalid actions (craft without inputs, mine without tool, give_item with no one present) are **rejected, produce no effect, cost 0–1 rounds, and are logged as `invalid_action`** with a reason string. Invalid-action rate is a tracked metric and a small penalty.

### 3.4 Resources, tech tree & recipes (author this table; tune numbers later)

**Biome → primary resources** (abundance is per-seed; this is the *type* mapping, shared across seeds):

| Biome | Resources |
|---|---|
| forest / jungle / taiga | wood, food(animals) |
| plains | food(animals/crops), wood(sparse) |
| mountains | cobblestone, coal, iron_ore |
| caves (deep) | cobblestone, coal, iron_ore, **diamond** |
| desert | sand, cactus, (sparse) |
| swamp | clay, (slime) |
| near-lava regions | **lava_pool** (obsidian via water/bucket), flint(gravel) |
| nether_wastes / soul_sand_valley | (fortress likely), blaze(in fortress), nether_wart |
| basalt_delta | basalt, (magma) |
| warped_forest | **enderman** (ender_pearl) |

**Tech tree / recipes** (→ = crafts to; gate = tool needed to mine):

```
wood → planks → sticks ; planks → crafting_table
planks+sticks → wooden_pickaxe → (gate) cobblestone, coal
cobblestone → stone_pickaxe, stone_sword, furnace
(stone_pickaxe gate) iron_ore ; iron_ore + coal --smelt--> iron_ingot
iron_ingot → iron_pickaxe, iron_sword, shield, bucket, flint_and_steel(+flint)
(iron_pickaxe gate) diamond → diamond_pickaxe, diamond_sword/armor
obsidian: (a) diamond_pickaxe mines a cooled lava_pool, OR (b) bucket + lava_pool (water trick) → cheaper route (reward clever strategy!)
10 obsidian --place frame-- + flint_and_steel --place--> nether_portal (lit)
[NETHER] fortress → fight blaze (superadditive) → blaze_rod → blaze_powder
[NETHER or warped_forest] fight enderman → ender_pearl
blaze_powder + ender_pearl → eye_of_ender   (need ~12)
eyes locate + activate → stronghold's end_portal → END
[END] fight ender_dragon (superadditive, hardest) → WIN
```

**DAG milestones (the reward ladder is in §7):** wood → stone tools → stable food → shelter/bed → iron → shield/bucket → lava/obsidian → portal built → Nether entered → fortress reached → blaze rods → ender pearls → eyes of ender → stronghold found → End portal active → End entered → dragon defeated.

### 3.5 Stochasticity & cooperation mechanics

- **Gather/fight succeed probabilistically.** `gather` yield ~ `Binomial(attempts, p_resource * tool_bonus * region_abundance)`. Keep `p` such that a competent agent makes steady progress; a tool-less or wrong-biome agent fails.
- **Day/night:** `day_length = 100` rounds (tune). Night → hostile mobs spawn in non-sheltered regions; movement/gather riskier; `sleep`/bed skips night if safe.
- **Hunger** drains each round (faster when fighting/moving); 0 hunger → health drain. Forces the Support role to matter.
- **Death** = health hits 0. **Default: dead agent respawns at the team's start region after `respawn_cost` rounds, dropping non-equipped inventory** (so death is costly but not episode-ending). Death count is a team penalty.
- **Superadditive fights (THE cooperation incentive):** for `blaze` and `ender_dragon`,
  `p_success = logistic( a·(n_colocated − 1) + b·combined_gear_score + c·avg_health − d·difficulty )`.
  Solo success is low; 2–3 co-located, well-geared agents succeed reliably. This makes "send a pair to the fortress / bring the team to the End" the *rational* strategy → Orca can discover it.
- **Co-location:** `give_item`, `regroup` resolution, and superadditive bonuses require teammates in `SAME_REGION`. This is why distance-band/bearing-to-teammates exists.
- **Inventory is per-agent, no global stash** → handoffs (`give_item`) genuinely matter (e.g., tinkerer needs the miner's iron).

### 3.6 Episode lifecycle

- **Turn-based synchronous rounds.** Each round: every `free` agent is queried for one macro-action (parallel LLM calls); actions resolve **simultaneously**; bus delivers messages; clock advances 1 round; durations make slow actions span multiple rounds.
- **Termination:** dragon defeated (WIN) **or** `round == T_max` (default `T_max = 600` rounds; tune) **or** all agents permanently failed. `baseline_steps` for the speed phase = median rounds-to-win across the first batch of wins.
- **Determinism:** every stochastic draw uses a seeded RNG `rng = Random(seed, episode_idx, round, agent_id)`. Same inputs → same outcome. LLM calls use low temperature; full prompts/outputs logged so runs are reconstructable.

### 3.7 Seeds (layout, not rules)

A **seed** = (biome graph topology, region positions, resource abundances, structure placements, Nether/stronghold locations). **Rules, tech tree, recipes, and probabilities are identical across all seeds.** Pool of 5: `{A, T2, T3}` for training (rotate), `{B, C}` held out for transfer eval only. A seed generator places ~20–40 Overworld regions + ~8–15 Nether regions with guaranteed reachability of every DAG prerequisite (validate at gen time: a scripted "oracle" path must exist).

---

## 4. Worker agents

### 4.1 Roles (soft priors, never hard masks)

Four agents with default roles. Roles bias what Orca tends to assign and what the system prompt emphasizes — they **never** restrict the action space. Orca can reassign anyone; a miner *can* build.

- **Explorer** — scouting, revealing regions/landmarks, finding biomes & structures, pathfinding headings.
- **Miner** — cobblestone/coal/iron/diamond, lava/obsidian logistics.
- **Tinkerer** — crafting, smelting, gear, portal construction, eyes of ender.
- **Support/Fighter** — food/hunger, combat, beds/shelter, escorting, reviving.

### 4.2 The worker turn (within an episode)

```
for each free agent this round:
    obs   = serialize_observation(world, agent)          # §3.2, coord-free
    prompt = build_worker_prompt(agent)                  # §4.3
    out   = LLM(prompt, temperature=0.2)                 # strong model
    action, messages = parse_and_validate(out)           # §4.4
    enqueue(action); post(messages -> bus)
```

All four free agents are called **in parallel** (async) each round.

### 4.3 Worker prompt structure

- **System prompt** = `ROLE_PRIMER[role]` + `behavior_card[agent]` (Orca-authored: current assignment, coaching directives, priorities, do/don'ts) + `execution_memory[agent]` (agent's own transferable heuristics) + `ACTION_SPEC` (the macro-action menu + validity rules) + `OUTPUT_SCHEMA`.
- **User prompt** = the JSON observation (§3.2) + a short, compacted **history summary** (NOT full history — running summary updated each round; message *content* is never truncated, but old turns are summarized) + the current team DAG frontier.
- **Output (strict JSON, validated):**

```json
{
  "reasoning": "one or two sentences, logged to Weave, never stored to memory",
  "action": {"name": "gather", "args": {"resource": "iron_ore"}},
  "messages": [
    {"to": "team", "type": "report", "content": "Found caves with iron to the N", "urgency": 0.4}
  ]
}
```

### 4.4 Parsing & invalid handling

- JSON-schema validate (pydantic). On malformed output: one repair retry, else default to `wait` + log `parse_failure`.
- The **env** (not the LLM) is the source of truth for validity. If the chosen action is illegal given world state, env rejects it, logs `invalid_action(reason)`, and the agent simply loses the turn. This is a feature to showcase ("agent asked to craft diamond armor with no diamonds → rejected & logged").

### 4.5 Execution-memory (agent-written, the verbal "learning" of workers)

- **Schema'd, not free-form.** A bounded list (cap ~8 entries) of `{"condition": "...", "action": "...", "confidence": 0–1}` heuristics about HOW to execute, e.g. `{"condition":"need iron but only wooden pickaxe","action":"craft stone pickaxe first","confidence":0.8}`.
- **Written at episode end** via a structured prompt that asks ONLY for transferable HOW-TO rules (biome→resource, ordering, tool prerequisites). Free-form notes are disallowed by schema.
- **Guard-filtered** before persisting (regex + cheap LLM check): strip anything seed-specific (numbers that look like coords/distances, unique landmark names). This filter is a concrete, demoable "harness sophistication" mechanism.
- **learning_signal modulates the write:** the magnitude of edits an agent may make this episode is scaled by Orca's `learning_signal` for that agent (+1 → may add/strengthen rules; ~0 → no change; −1 → weaken/remove the rule Orca flagged). This is the "listen critically or not" dial.
- **Accept-gated** (§6.4): memory + card updates are only *kept* if they don't regress the eval pool.

> Distinction to keep crisp: **execution-memory = HOW to do tasks (agent owns).** **behavior-card = WHO does what + coaching (Orca owns).** Both persist across episodes; both are guard-filtered.

---

## 5. Communication bus

### 5.1 Message schema (structured only — no free chat)

```json
{
  "from": "agent_1",
  "to": "team",                       // "team" | "agent_k" | "orca"
  "type": "request_help",             // report | request_help | share_finding | propose_rendezvous | ack | handoff
  "content": "Need food before continuing Nether search",   // length NOT capped (your call)
  "urgency": 0.7,                     // 0–1, helps prioritize attention
  "round": 42
}
```

### 5.2 Delivery & visibility

- **Turn-based delivery:** messages posted in round *t* are delivered in round *t+1* (avoids within-round causality loops). Simple and debuggable.
- Each agent's observation includes `recent_messages` = last K messages addressed to it or `team` (K ~ 8; summarize older). **Content is never truncated**; only old turns are summarized out of the running history.
- All messages are logged to Weave verbatim — the message log is part of the demo ("watch them coordinate") and part of Orca's trace input.
- Optional **bandwidth realism** (stretch): cap *messages-per-agent-per-round* (e.g., ≤2) instead of length, to prevent spam without limiting expressiveness.

---

## 6. Orchestrator Orca (the differentiator)

Orca runs **between episodes**. It is the only component with a learned numeric signal (the delegation bandit) plus an LLM coaching layer. It never acts inside an episode.

### 6.1 Inputs (the summarized trace)

A compact, structured episode digest (NOT the raw token stream):
- DAG frontier reached + timeline of milestones (round at which each was hit).
- Per-agent objective stats: assigned task, subtask completion (bool/■), invalid-action count, idle rounds, deaths, items gathered/crafted, handoffs given/received, useful messages.
- Cooperation events: co-located fights attempted/won, rendezvous success.
- Bottlenecks: longest stalls, repeated invalid actions, starvation/death causes.
- The previous behavior-cards + execution-memories (for diffing).

### 6.2 Outputs

For each agent: `performance_score ∈ [0,1]`, `learning_signal ∈ [−1,1]`, and **verbal feedback** (the coaching that becomes the next behavior-card). At the team level: the **delegation choice** for the next episode (from the bandit) and a short strategic plan with contingencies (e.g., "if no fortress found by round 200, pair agents 1+4 and search soul_sand biomes together").

### 6.3 The delegation bandit (the quantitative learner)

Keep it **small and per-situation** so it learns within dozens of episodes.

- **Decision points (situations):** a small fixed set of recurring strategic forks, e.g.
  `S1: early-game role assignment` (which agent → explorer/miner/tinkerer/support),
  `S2: nether-entry policy` (enter when gear≥X vs immediately),
  `S3: fortress-search formation` (solo / two-pairs / all-together),
  `S4: end-approach` (regroup-all vs split).
- **Arms:** each situation has 2–4 discrete options (a small menu).
- **Context (optional, coarse):** bucket the situation by a few discrete features (e.g., `seed_family`, `phase`) so the bandit isn't fragmented; default to **non-contextual per-situation** if data is scarce.
- **Algorithm:** ε-greedy or Thompson sampling over arms, value = running mean of **episode team frontier** (the §7 scalar) observed when that arm was chosen. ~30 lines. Update **once per episode**.
- **Output curve = the demo's RL evidence:** plot value estimates / chosen-arm frontier over episodes → "Orca learned that *two-pairs* beats *solo* for fortress search."

> Why a bandit and not PPO: Orca acts once per episode over a tiny discrete space → it's a (contextual) bandit by construction. No backprop, no instability, learns fast, gives a clean curve.

### 6.4 Verbal coaching + credit assignment (the readable layer)

Orca reads the trace and **reasons in natural language about credit**, distinguishing *delegation* errors from *execution* errors, e.g. *"agent 2's mining stalled because it lacked a stone pickaxe — execution gap; keep the assignment but add 'craft stone pickaxe before mining iron' to its card."* vs *"agent 3 idled 40 rounds waiting for iron — delegation error; assign mining earlier or to a second agent."* This reasoning is logged to Weave and is **more compelling than a counterfactual plot judges can't see**.

- The coaching is written into the next **behavior-card** (assignment + directives + priorities).
- **Anti-circularity safeguard:** `performance_score` is computed **mostly from objective env stats** (subtask completion, frontier contribution, invalid/idle/deaths) and only lightly from Orca's opinion. Orca cannot inflate its own metric because the headline reward (§7) is the **objective DAG frontier**, not Orca's scores.

### 6.5 Accept/reject gate (anti-noise — critical)

Do **not** blindly keep every update. After Orca proposes new cards/memory/delegation:
1. Run a small **eval batch** (e.g., 2–3 episodes across the training seed pool) with the proposed update.
2. **Keep** it iff mean team frontier ≥ current best − ε; otherwise **roll back** to the previous version.
3. Maintain a **static baseline** snapshot for comparison throughout.
This hill-climb-with-rollback is what turns noisy LLM edits into monotone-ish improvement, and it's a great thing to show.

### 6.6 Phasing of Orca's authority (nonstationarity control)

- **Phase 0 (warmup):** worker behavior-cards FROZEN to sensible defaults; Orca only learns **delegation** (bandit). Removes coupled nonstationarity while the bandit finds good assignments.
- **Phase 1:** Orca may also **edit behavior-cards / approve execution-memory** (small, structured, accept-gated).
- **Phase 2 (post first win):** activate the **speed reward** (§7.4). Do not enable before a first win — it destabilizes learning.

---

## 7. Reward system (one episode-level scalar; two advisory dials)

### 7.1 Team frontier reward (the headline, drives the bandit)

**Max frontier, not cumulative** (so agents can't farm easy subtasks). The episode's base reward = the value of the **deepest DAG milestone reached**:

```
0.05  wood / basic tools
0.12  stable food + shelter/bed
0.20  iron tooling
0.30  nether portal built
0.40  Nether entered
0.55  fortress found
0.65  blaze rods acquired
0.75  ender pearls + eyes of ender
0.80  stronghold found
0.85  End entered
1.00  dragon defeated
```

### 7.2 Penalties (subtracted from the frontier base)

`team_reward = frontier_value − w_d·deaths − w_i·invalid_rate − w_idle·idle_fraction` (small weights, e.g. 0.02/0.05/0.05; clip to ≥0). These shape *quality* without overwhelming the progression signal.

### 7.3 The two advisory dials (NOT the headline reward)

- `performance_score[agent] ∈ [0,1]`: mostly objective (subtask completion + frontier contribution + low invalid/idle), lightly Orca's opinion. Used in feedback + logged; **not** summed into the team reward.
- `learning_signal[agent] ∈ [−1,1]`: Orca's "adopt this lesson?" dial; **scales the agent's memory edit magnitude** (§4.5). Not a reward; a control signal on learning.

### 7.4 Speed phase (only after first dragon kill)

```
final_reward = completion_reward + speed_bonus
speed_bonus  = max(0, (baseline_steps − current_steps) / baseline_steps)   # only if WIN
```
`baseline_steps` = median rounds-to-win of the first batch of wins. Episodes that exceed a time cap terminate and are penalized for slowness. **Never enable in Phase 0/1.**

### 7.5 "Individual shaping," clarified (this was the confusing part)

In C2 there is **no per-step gradient**, so "shared team reward + light individual shaping" means:
- **Shared:** the single team frontier scalar (§7.1) is what the **bandit** consumes and what we report. Every agent is credited with the same team outcome.
- **Individual shaping:** the per-agent *objective stats* (completed assigned task, useful comms, answered help, avoided invalids, correct handoffs, reduced a bottleneck) feed **Orca's `performance_score` and coaching**, and are **logged for ablation** — they are a *small* nudge inside Orca's judgment, deliberately kept minor so agents don't optimize local chores over team progress. They are **not** a separate reward stream to the workers.

---

## 8. Training / run loop & phasing

```python
orca = Orca()                      # bandit + LLM coach, starts from default cards
history = []
for episode in range(N_EPISODES):
    seed   = TRAIN_SEEDS[episode % len(TRAIN_SEEDS)]      # rotate A/T2/T3
    config = orca.choose_config(history)                  # bandit picks arms; cards from current best
    trace  = run_episode(seed, config)                    # §3.6, parallel worker calls
    metrics= reward_computer(trace)                        # §7 team frontier + stats
    orca.bandit_update(config.arms, metrics.team_frontier)
    proposal = orca.coach(trace, metrics)                 # new cards + memory edits + scores
    if accept_gate(proposal, EVAL_SEEDS_SMALL):           # §6.5 keep iff no regression
        orca.commit(proposal)
    log_to_weave(episode, seed, config, trace, metrics, proposal)
    history.append((seed, config, metrics))
```

- **Phase 0** (≈ episodes 0–15): cards frozen, bandit-only. **Phase 1** (≈ 15–N): coaching on, accept-gated. **Phase 2**: after first WIN, switch reward to speed (§7.4).
- **Parallelism:** run several episodes concurrently (async). Throughput is bounded by LLM latency/rate-limits, not money — request a rate-limit bump for the event and cap `T_max` to keep episodes short early.
- **Checkpoint** Orca state (cards, memories, bandit tables) every episode so you can resume / roll back / demo any point.

---

## 9. Evaluation protocol (the headline result)

**Conditions to compare:**
1. **Static baseline** — fixed roles, no Orca updates, no memory.
2. **Comms-no-Orca** — agents message but no delegation/coaching/memory.
3. **Full C2** — Orca bandit + coaching + memory + gate.

**Transfer test (the money plot):** train Full C2 on `{A,T2,T3}`; freeze Orca's learned cards + bandit; evaluate all three conditions on **held-out `{B,C}`** for `n` episodes each. Plot mean team frontier ± variance per condition, train vs held-out. **Claim:** Full C2 ≥ baselines on held-out → it learned *transferable strategy*, not terrain.

**Ablations (2–3 bars are plenty):** memory ON/OFF · coaching ON/OFF · accept-gate ON/OFF (shows the gate's anti-noise value). Each ablation = one line/bar on the same axes.

**Report:** learning curve (frontier vs episode on train), bandit arm-value curve, transfer bar chart (3 conditions × {train,held-out}), one ablation chart, invalid-action-rate over time. Always report over multiple seeds/episodes with variance — never a single anecdote.

---

## 10. Weave instrumentation (sponsor axis — make it load-bearing)

Decorate everything with `@weave.op` so traces nest automatically:
- `run_episode`, `worker_turn` (inputs=obs, outputs=action+messages), `env.step`, `reward_computer`, `orca.choose_config`, `orca.coach`, `accept_gate`.
- **Log:** every observation, every action + validity result, **invalid-action rate**, every bus message, inventory-over-time, DAG-milestone timeline, Orca's scores + verbal feedback, **behavior-card/memory diffs between episodes**, bandit arm values, seed A/B/C eval results, baseline comparison.
- **Weave Evaluation harness** for the §9 comparisons (custom scorers: frontier, milestone, time-to-win, invalid-rate, cooperation-events); use the **comparison view** as your results section and the **leaderboard** to rank conditions.
- **The pitch trace:** capture one nested trace where **Orca notices a failure → edits a card → the next run clears that bottleneck**. This single auditable trace *is* the demo.

---

## 11. Tech stack & repo layout

- **Python 3.11**, `asyncio` for parallel worker calls, `pydantic` for all schemas (obs, action, message, card, memory), `weave` + `wandb`, one LLM client behind an interface `llm.complete(prompt, schema)` so the model is swappable (hosted ↔ local vLLM) via config. **Strong model for workers and Orca** (money is not a constraint); keep `temperature` low for determinism.

```
orca/
  env/        world.py (graph-on-plane), seeds.py (generator+validator), techtree.py,
              actions.py (resolve+validity), observation.py (coord-free serializer), rng.py
  agents/     worker.py (turn loop), prompts.py, memory.py (schema+guard filter)
  bus/        messages.py, bus.py
  orca/       orca.py, bandit.py, coach.py, gate.py, cards.py
  reward/     dag.py (frontier ladder), reward.py (penalties, speed phase)
  train/      loop.py, phases.py, checkpoint.py
  eval/       baselines.py, transfer.py, ablations.py, plots.py
  obs_guard/  coord_leak_test.py   # invariant test: no coords ever leak
  telemetry/  weave_ops.py
  configs/    default.yaml (all tuning knobs in §15)
  run.py
```

**Interface contract to lock in hour 1 (lets people parallelize):** `Observation`, `Action`, `Message`, `BehaviorCard`, `ExecutionMemory`, `EpisodeTrace`, `EpisodeMetrics` pydantic models. Build everyone against these.

---

## 12. 30-hour build plan (thin vertical slice first)

Assumes a small team (2–4). If solo, do the same order and cut harder. **Get an ugly end-to-end loop before widening.**

- **H0–2 — Contracts + skeleton.** Define all pydantic schemas (§11). Stub env (5 regions), 1 agent, no-op Orca, Weave wired. One episode runs and logs *today*.
- **H2–8 — Env core.** Graph-on-plane world + seed generator + validator, full action resolution + validity, coord-free observation serializer + **leak test**, tech tree + reward DAG frontier. Scripted "oracle" agent reaches dragon on seed A (proves the env is winnable).
- **H8–14 — Workers + comms.** Worker turn loop, prompts, JSON parse/validate, bus + structured messages, history summarization. 4 agents play a full episode; reach mid-DAG on seed A. **(This is your fallback demo.)**
- **H14–20 — Orca.** Trace digest, `performance_score`/`learning_signal`, behavior-cards, the **delegation bandit**, verbal coach, **accept/reject gate**, execution-memory write + guard filter. Bandit curve starts moving across episodes.
- **H20–25 — Cooperation + transfer.** Superadditive fights + co-location; run held-out `{B,C}`; baselines (static, comms-no-Orca). Produce transfer plot + 1–2 ablations.
- **H25–28 — Polish + Weave dashboard.** Curves, the "failure→fix→improvement" trace, leaderboard. **Record a backup demo video.** Freeze features by H27.
- **H28–30 — Pitch.** Rehearse §14 against the judging axes.

**Team split:** (1) Env + reward/DAG + seeds. (2) Workers + memory + comms. (3) Orca + bandit + gate + eval/plots + Weave. Integrate continuously against the §11 contracts.

**Cut list (when behind, in order):** drop accept-gate eval batch → single-episode keep; drop one ablation; 2 agents instead of 4; shallower DAG (stop at End-entered); Phase 2 speed reward; comms-no-Orca baseline.

---

## 13. Risks & mitigations (condensed)

- **Noisy improvement / few episodes** → fast env, async parallelism, accept-gate for monotone-ish gains, report variance, run overnight, lean on qualitative "failure→fix" trace if curves are jumpy.
- **"Is it just the smart model?"** → the static + comms-no-Orca **baselines** and the transfer gap are the answer. Non-negotiable to run.
- **LLM-judge unfaithfulness** → headline reward is the **objective DAG frontier**, not Orca's scores; Orca's opinion is advisory only.
- **Coordinate leakage** → coord-free serializer is the single obs path + invariant test + memory guard filter. Triple-checked.
- **Coupled nonstationarity** → Phase 0 freezes workers while the bandit learns; updates are small, structured, accept-gated.
- **Reward farming** → max-frontier (not cumulative) + monotonic DAG; speed reward only post-win.
- **Env too easy (toy) vs too hard (never wins)** → tune so the **scripted oracle wins**, a **greedy baseline stalls mid-DAG**, and the full system reaches the dragon. That gap is the result.
- **Determinism vs LLM stochasticity** → seeded env RNG, low temp, full prompt/output logging.

---

## 14. Pitch / demo script (have ready by H27)

1. **Hook (10s):** "We built a manager AI that *learns to manage* a team of AIs — and we prove it learned *strategy*, not the map."
2. **Problem:** delegation + credit assignment in cooperative agent teams.
3. **System (show the Weave trace tree):** Orca → 4 role-biased workers → structured comms → coord-free Minecraft-lite. Memory-discipline + guard-filter mechanism on screen.
4. **The result:** held-out seeds B/C — Full C2 beats static + comms-no-Orca → transferable strategy. One ablation bar.
5. **A concrete learned behavior:** the bandit curve + "Orca learned two-pairs beats solo for the fortress," shown in the trace where it *noticed the failure and fixed the card*.
6. **Honest limits + generalization:** name the milestone you hit; "this manager-over-workers pattern generalizes to ops/coding/research swarms."
7. **Sponsor:** "all observable, evaluable, reproducible in Weave."

Don't claim the dragon unless you killed it; an honest, well-evidenced narrower result wins.

---

## 15. Tuning knobs (starting values — all in `configs/default.yaml`)

| Knob | Start | Notes |
|---|---|---|
| `T_max` (rounds/episode) | 600 | raise as agents get competent |
| `day_length` | 100 | night danger cadence |
| `N_EPISODES` | as many as 14h allows | bounded by LLM latency/rate-limit |
| Phase 0 length | 15 episodes | bandit warmup, cards frozen |
| worker temperature | 0.2 | determinism |
| memory cap | 8 heuristics/agent | keeps prompts bounded |
| penalty weights | death 0.02, invalid 0.05, idle 0.05 | keep << frontier steps |
| superadditive `a,b,c,d` | tune so solo p≈0.2, trio p≈0.85 | the cooperation incentive |
| bandit | ε=0.2 greedy or Thompson | per-situation arms |
| accept-gate ε | 0.0–0.02 | tolerance for keeping an update |
| train seeds | A, T2, T3 | rotate |
| held-out seeds | B, C | eval only, never trained on |

---

### Final invariant checklist (pin this)
- [ ] No coordinate ever leaves the env (serializer + test).
- [ ] Reward is **max-frontier**, computed once per episode.
- [ ] Bandit updates once/episode on the team frontier; gives a curve.
- [ ] Orca's scores are advisory; objective DAG is the headline.
- [ ] Every update passes the accept/reject gate.
- [ ] Held-out B/C are never trained on.
- [ ] Everything is a `@weave.op`.
- [ ] Scripted oracle proves each seed is winnable before agents run.

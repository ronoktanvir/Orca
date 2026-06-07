"""Full-DAG scripted oracle + greedy contrast + winnability validator (E3, §3.7).

Proves each seed is *beatable*: a deterministic scripted solver drives one agent
all the way to ``dragon_defeated`` using ONLY the real action/transition mechanics
(gather/craft/smelt/place/fight + ``move {to: …}`` portal travel). It is the
"prove the env is winnable" tool the build plan calls for (§12 / §3.7) — not a
learning agent, so it is allowed privileged read access to the world to navigate
and locate resources/structures; every *action* it takes still goes through the
env's normal validity checks (so it provably emits zero invalid actions).

The shallow ``agents/scripted.ShallowOracle`` (coordinate-free, reaches iron) is
kept as-is; this lives in ``env/`` so Stream 1 owns the full-DAG prover.

A :class:`GreedyAgent` is the deliberate contrast: it grabs the locally-easy iron
tier and then stalls — it never builds a portal, so it never reaches the dragon.
That oracle-wins / greedy-stalls gap is the headline result (§13).
"""

from __future__ import annotations

from collections import deque

from contracts import Action
from contracts.enums import ActionName, Bearing, Layer, Milestone, Role, Structure

from .stub_env import StubEnv


class _Solver:
    """Privileged driver around a live :class:`StubEnv` for a single agent."""

    def __init__(self, env: StubEnv, agent_id: str) -> None:
        self.env = env
        self.aid = agent_id
        self.world = env.world
        self.invalid_actions = 0

    # -- state shortcuts ---------------------------------------------------- #
    @property
    def _agent(self):
        return self.world.agents[self.aid]

    @property
    def _inv(self) -> dict[str, int]:
        return self._agent.inventory

    def _have(self, item: str, n: int = 1) -> bool:
        return self._inv.get(item, 0) >= n

    def _won(self) -> bool:
        return Milestone.DRAGON_DEFEATED in self.world.world_milestones

    # -- one action through the real env ------------------------------------ #
    def _act(self, action: Action) -> bool:
        if self.env.done:
            return False
        res = self.env.step({self.aid: action})
        rec = res.records[0] if res.records else None
        if rec is None:
            return False
        if not rec.valid:
            self.invalid_actions += 1
            return False
        return True

    # -- privileged geometry helpers (navigation only; no coord leak to obs) - #
    def _bfs_dist(self, src: str, dst: str) -> float:
        if src == dst:
            return 0
        seen = {src}
        q = deque([(src, 0)])
        while q:
            rid, d = q.popleft()
            for nb, _w in self.world.neighbors(rid):
                if nb == dst:
                    return d + 1
                if nb not in seen:
                    seen.add(nb)
                    q.append((nb, d + 1))
        return float("inf")

    def _nearest(self, predicate, layer=None) -> str | None:
        """Nearest (BFS) region in ``layer`` (default: current) matching predicate."""
        cur = self._agent.region_id
        layer = layer or self.world.regions[cur].layer
        best, best_d = None, float("inf")
        for rid, region in self.world.regions.items():
            if region.layer != layer or not predicate(region):
                continue
            d = self._bfs_dist(cur, rid)
            if d < best_d:
                best_d, best = d, rid
        return best

    def goto(self, target: str, cap: int = 400) -> bool:
        """Walk to ``target`` (same layer) via real MOVE actions. Prefers a move
        that strictly reduces BFS distance; escapes ties via unvisited neighbors."""
        visited: set[str] = set()
        for _ in range(cap):
            cur = self._agent.region_id
            if cur == target:
                return True
            visited.add(cur)
            cur_d = self._bfs_dist(cur, target)
            options = []  # (resulting_dist, unvisited?, bearing)
            for bearing in Bearing:
                nb = self.world.resolve_move(cur, bearing)
                if nb is not None:
                    options.append((self._bfs_dist(nb, target), nb not in visited, bearing))
            if not options:
                return False
            closer = [o for o in options if o[0] < cur_d]
            if closer:
                choice = min(closer, key=lambda o: o[0])
            else:
                unvisited = [o for o in options if o[1]]
                pool = unvisited or options
                choice = min(pool, key=lambda o: o[0])
            if not self._act(Action(name=ActionName.MOVE, args={"direction": choice[2].value})):
                return False
        return self._agent.region_id == target

    # -- macro steps -------------------------------------------------------- #
    def gather(self, resource: str, n: int) -> bool:
        if self._have(resource, n):
            return True
        target = self._nearest(lambda r: r.resources.get(resource, 0) > 0)
        if target is None:
            return False
        for _ in range(n * 5 + 25):
            if self._have(resource, n):
                return True
            if self._agent.region_id != target and not self.goto(target):
                return False
            if not self._act(Action(name=ActionName.GATHER, args={"resource": resource})):
                return False
        return self._have(resource, n)

    def craft(self, item: str, times: int = 1) -> bool:
        return all(self._act(Action(name=ActionName.CRAFT, args={"item": item})) for _ in range(times))

    def smelt(self, item: str, times: int = 1) -> bool:
        return all(self._act(Action(name=ActionName.SMELT, args={"item": item})) for _ in range(times))

    def place(self, item: str) -> bool:
        return self._act(Action(name=ActionName.PLACE, args={"item": item}))

    def fight(self, target: str, times: int = 1) -> bool:
        return all(self._act(Action(name=ActionName.FIGHT, args={"target": target})) for _ in range(times))

    def portal(self, to_layer: str) -> bool:
        return self._act(Action(name=ActionName.MOVE, args={"to": to_layer}))

    # -- phases ------------------------------------------------------------- #
    def basics_to_iron(self) -> bool:
        """wood -> planks/sticks/table -> wooden+stone tools + furnace -> iron ingots."""
        ok = (
            self.gather("wood", 10)
            and self.craft("planks", 8)  # 8 wood -> 32 planks
            and self.craft("sticks", 4)  # 8 planks -> 16 sticks
            and self.craft("crafting_table")
            and self.craft("wooden_pickaxe")
            and self.gather("cobblestone", 14)
            and self.craft("stone_pickaxe")
            and self.craft("furnace")
            and self.gather("coal", 11)
            and self.gather("iron_ore", 9)
            and self.smelt("iron_ore", 8)  # 8 iron_ingot
        )
        return ok

    def iron_to_portal(self) -> bool:
        """iron gear (pickaxe/bucket/flint&steel) -> diamond -> obsidian (water
        trick) -> lit nether portal."""
        ok = (
            self.craft("iron_pickaxe")
            and self.craft("bucket")
            and self.gather("flint", 2)
            and self.craft("flint_and_steel")
            and self.gather("diamond", 4)
            and self.craft("diamond_pickaxe")  # diamond tier reached
            and self.gather("lava_pool", 10)
            and self.craft("obsidian", 10)  # water-trick route (needs the bucket)
            and self.craft("nether_portal")
            and self.place("nether_portal")  # lights it in the current Overworld region
        )
        return ok

    def conquer_nether(self) -> bool:
        """Enter the Nether, beat blazes (rods) + endermen (pearls), craft 12 eyes."""
        if not self.portal("nether"):
            return False
        fortress = self._nearest(lambda r: r.structure == Structure.FORTRESS)
        if fortress is None or not self.goto(fortress):
            return False
        # 6 blaze rods -> 12 blaze powder; 12 ender pearls; 12 eyes of ender.
        return (
            self.fight("blaze", 6)
            and self.fight("enderman", 12)
            and self.craft("blaze_powder", 6)
            and self.craft("eye_of_ender", 12)
        )

    def slay_dragon(self) -> bool:
        """Return Overworld, activate the stronghold End portal, enter the End, win."""
        if not self.goto(self.world.nether_entry_id) or not self.portal("overworld"):
            return False
        if not self.goto(self.world.stronghold_id):
            return False
        return (
            self.craft("end_portal")
            and self.place("end_portal")
            and self.portal("end")
            and self.fight("ender_dragon")
        )

    def run(self) -> bool:
        if not (self.basics_to_iron() and self.iron_to_portal()
                and self.conquer_nether() and self.slay_dragon()):
            return False
        return self._won()

    def run_greedy(self) -> bool:
        """Greedy/myopic contrast: take the easy iron tier, then stop. Never builds
        a portal, so it stalls mid-DAG and never reaches the dragon."""
        self.basics_to_iron()
        return self._won()  # always False — that's the point


class FullDagOracle:
    """Scripted solver that drives a seed to ``dragon_defeated`` (§3.7)."""

    def __init__(self, agent_id: str = "oracle") -> None:
        self.agent_id = agent_id
        self.invalid_actions = 0

    def solve(self, env: StubEnv) -> bool:
        solver = _Solver(env, self.agent_id)
        won = solver.run()
        self.invalid_actions = solver.invalid_actions
        return won


class GreedyAgent:
    """Myopic baseline that stalls mid-DAG (reaches iron, never the dragon)."""

    def __init__(self, agent_id: str = "greedy") -> None:
        self.agent_id = agent_id
        self.invalid_actions = 0

    def solve(self, env: StubEnv) -> bool:
        solver = _Solver(env, self.agent_id)
        won = solver.run_greedy()
        self.invalid_actions = solver.invalid_actions
        return won


def _build_env(seed: str, t_max: int, agent_id: str) -> StubEnv:
    env = StubEnv(
        seed=seed,
        agents=[(agent_id, Role.TINKERER)],
        t_max=t_max,
        stop_at_milestone=None,  # run the full DAG, don't stop at iron
    )
    env.reset()
    return env


def validate_winnable(seed: str, t_max: int = 8000) -> bool:
    """True iff the full-DAG oracle reaches ``dragon_defeated`` on ``seed`` (§3.7)."""
    env = _build_env(seed, t_max, "oracle")
    return FullDagOracle("oracle").solve(env)


__all__ = ["FullDagOracle", "GreedyAgent", "validate_winnable"]

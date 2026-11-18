import math
import random
import time
from collections import defaultdict

from .rules import english, total_size
from .search import DOOR, Search, door_search, targets
from .sim import ACTIONS, MOVES, RESET, bfs, changes, from_obs, observe, step
from .worlds import world_rules


def result(env, **extra):
    return {"solved": env.done, "steps": env.steps, **extra}


class History:
    def __init__(self, initial_obs):
        self.initial_obs = initial_obs
        self.actions, self.observations = [], []

    def add(self, a, obs):
        self.actions.append(a)
        self.observations.append(obs)

    @property
    def last(self):
        return self.observations[-1] if self.observations else self.initial_obs

    def before(self, i):
        return self.observations[i - 1] if i else self.initial_obs


# --- simple agents ---

class Oracle:
    # cheats on purpose: knows the true rules. Only used as the lower bound.
    def __init__(self, world, budget=100):
        self.rules, self.budget = world_rules(world), budget

    def run(self, env):
        for a in bfs(env.initial_state, self.rules) or []:
            if env.steps >= self.budget:
                break
            env.step(a)
        return result(env)


class RandomAgent:
    def __init__(self, budget=100, seed=0):
        self.budget, self.rng = budget, random.Random(seed)

    def run(self, env):
        while not env.done and env.steps < self.budget:
            env.step(self.rng.choice(ACTIONS + (RESET,)))
        return result(env)


class Explorer:
    # tries every object once (interact, or bump for doors). context=True: an object counts as untried
    # again whenever everything else visibly looks different from the last time it was tried.
    def __init__(self, seed=0, context=False):
        self.tried = set()
        self.rng = random.Random(seed)
        self.context = context

    def ctx(self, s, oid):
        if not self.context:
            return ()
        others = tuple(sorted((o.id, tuple(sorted(o.props.items()))) for o in s.objs.values() if o.id != oid))
        return others, tuple(sorted(s.inv))

    def record(self, s, action):
        if action == "interact":
            for o in s.at(s.agent):
                self.tried.add((o.id, "interact", self.ctx(s, o.id)))
        elif action in MOVES:
            dx, dy = MOVES[action]
            door = s.closed_door((s.agent[0] + dx, s.agent[1] + dy))
            if door is not None:
                self.tried.add((door.id, "bump", self.ctx(s, door.id)))

    def next_action(self, s):
        best = None
        for o in s.objs.values():      # world order on purpose: ties go to the first object
            if o.pos is None or o.type == "box":
                continue
            verb = "bump" if o.type == "door" else "interact"
            if (o.id, verb, self.ctx(s, o.id)) in self.tried:
                continue
            if verb == "bump":
                plan = bump_plan(s, o)
            else:
                path = bfs(s, (), goal=lambda t, p=o.pos: t.agent == p, max_nodes=5000)
                plan = None if path is None else path + ["interact"]
            if plan and (best is None or len(plan) < len(best[0])):
                best = (plan, f"try {verb} on untested {o.id}")
        if best:
            return best[0][0], best[1]
        return self.rng.choice(list(MOVES) + ["interact", "wait"]), "random exploration (everything tried)"


class ExploreOnly:
    def __init__(self, budget=100, seed=0, context=True):
        self.budget = budget
        self.explorer = Explorer(seed, context)

    def run(self, env):
        init = env.initial_state
        obs = env.initial_obs
        while not env.done and env.steps < self.budget:
            now = from_obs(init, obs)
            walk = bfs(now, (), max_nodes=50_000)
            a = walk[0] if walk else self.explorer.next_action(now)[0]
            self.explorer.record(now, a)
            obs, _ = env.step(a)
        return result(env)


# --- experiments (macro actions) ---

class Macro:
    def __init__(self, name, actions):
        self.name, self.actions = name, actions
        self.gain = self.p_goal = 0.0
        self.expect = None

    @property
    def cost(self):
        return max(1, len(self.actions))

    def __repr__(self):
        return f"<{self.name} {self.cost}>"


def bump_plan(s, door):
    best = None
    for a, (dx, dy) in MOVES.items():
        stand = (door.pos[0] - dx, door.pos[1] - dy)
        path = bfs(s, (), goal=lambda t, p=stand: t.agent == p, max_nodes=5000)
        if path is not None and (best is None or len(path) + 1 < len(best)):
            best = path + [a]
    return best


def walk(s, path):
    for a in path:
        s = step(s, a, ())
    return s


def experiments(now, waits=(1, 3, 6, 10), pairs=True, nodes=20_000):
    out = [Macro(f"wait {k}", ["wait"] * k) for k in waits]
    objs = sorted((o for o in now.objs.values() if o.pos is not None), key=lambda o: o.id)
    for o in objs:
        if o.type == "box":
            continue
        if o.type == "door":
            plan = bump_plan(now, o)
            if plan:
                out.append(Macro(f"bump {o.id}", plan))
            continue
        path = bfs(now, (), goal=lambda t, p=o.pos: t.agent == p, max_nodes=5000)
        if path is None:
            continue
        out.append(Macro(f"interact {o.id}", path + ["interact"]))
        if pairs:
            there = walk(now, path + ["interact"])
            for o2 in objs:
                if o2.id == o.id or o2.type in ("box", "door"):
                    continue
                p2 = bfs(there, (), goal=lambda t, p=o2.pos: t.agent == p, max_nodes=5000)
                if p2 is not None:
                    out.append(Macro(f"interact {o.id} then {o2.id}", path + ["interact"] + p2 + ["interact"]))
        if o.type == "plate":
            s = walk(now, path)
            off = next((a for a in MOVES if step(s, a, ()).agent != s.agent), None)
            if off:
                out.append(Macro(f"step on and off {o.id}", path + [off]))
    for b in (o for o in objs if o.type == "box"):
        for p in (o for o in objs if o.type == "plate"):
            path = bfs(now, (), goal=lambda t, b=b.id, p=p.pos: t.objs[b].pos == p, max_nodes=nodes)
            if path:
                out.append(Macro(f"push {b.id} onto {p.id}", path))
    return out


# --- the search agent ---

class Member:
    __slots__ = ("door", "rules", "state", "logw")

    def __init__(self, door, rules, state, logw):
        self.door, self.rules, self.state, self.logw = door, rules, state, logw


class Pool:
    # side models (shortest rule set per switch/lamp/key) + many door candidates, each replayed to now
    def __init__(self, init, cap=300, lam=0.3, scorer=None, scorer_weight=1.0, ranker_boost=2.7, max_calls=10):
        self.init, self.cap, self.lam = init, cap, lam
        self.scorer, self.scorer_weight, self.ranker_boost, self.max_calls = scorer, scorer_weight, ranker_boost, max_calls
        self.side = ()
        self.members = []
        self.boost = {}
        self.calls = 0
        self.searches = 0
        self.search_seconds = 0.0
        self._last_ranked = None

    def refresh(self, h, side=True, rank=True):
        t0 = time.time()
        args = (h.actions, h.observations, h.initial_obs)
        if side:
            rules = []
            for t in targets(self.init):
                if t == DOOR:
                    continue
                found = Search(self.init, t, cap=1, budget=300_000).search(*args)
                if found:
                    rules += found[0]
            self.side = tuple(rules)
        good, loose = [], []
        for door in door_search(self.init, cap=self.cap).search(*args):
            rules = self.side + door
            s, fits = self.init, True
            for a, obs in zip(h.actions, h.observations):
                s = self.init if a == RESET else step(s, a, rules)
                fits = fits and DOOR.value(observe(s)) == DOOR.value(obs)
            m = Member(door, rules, s, -self.lam * total_size(door) + self.boost.get(door, 0.0))
            (good if fits else loose).append(m)
        # teacher forcing can let through a door candidate that a wrong side model breaks on full replay
        self.members = good or loose
        self.searches += 1
        self.search_seconds += time.time() - t0
        if rank:
            self.rank(h)

    def rank(self, h):
        if self.scorer is None or self.calls >= self.max_calls or len(self.members) < 2:
            return
        learned = getattr(self.scorer, "scores", False)
        top = sorted(self.members, key=lambda m: -m.logw)[: len(self.members) if learned else 200]
        key = frozenset(m.door for m in top)
        if key == self._last_ranked:
            return
        self._last_ranked = key
        self.calls += 1
        self.boost = {}
        for m in top:
            m.logw = -self.lam * total_size(m.door)
        if learned:
            for m, x in zip(top, self.scorer(h, [m.door for m in top])):
                m.logw += self.scorer_weight * x
                self.boost[m.door] = self.scorer_weight * x
        else:
            for i in set(self.scorer(h, [m.door for m in top])):
                if 0 <= i < len(top):
                    top[i].logw += self.ranker_boost
                    self.boost[top[i].door] = self.ranker_boost

    def observe(self, action, obs):
        keep, side_wrong = [], False
        for m in self.members:
            m.state = self.init if action == RESET else step(m.state, action, m.rules)
            pred = observe(m.state)
            if DOOR.value(pred) == DOOR.value(obs):
                keep.append(m)
                side_wrong = side_wrong or pred != obs
        dead = len(self.members) - len(keep)
        self.members = keep
        return dead, side_wrong

    def ranked(self):
        if not self.members:
            return []
        top = max(m.logw for m in self.members)
        w = [math.exp(m.logw - top) for m in self.members]
        z = sum(w)
        return sorted(((m, x / z) for m, x in zip(self.members, w)), key=lambda mw: -mw[1])


class SearchAgent:
    def __init__(self, budget=100, selector="eig", cap=300, lam=0.3, goal_bonus=2.0, tau=0.5, top_plans=8,
                 plan_nodes=20_000, pairs=True, seed=0, scorer=None, scorer_weight=1.0, detail=0):
        self.budget, self.selector, self.cap, self.lam = budget, selector, cap, lam
        self.goal_bonus, self.tau, self.top_plans, self.plan_nodes = goal_bonus, tau, top_plans, plan_nodes
        self.pairs, self.seed, self.scorer, self.scorer_weight, self.detail = pairs, seed, scorer, scorer_weight, detail

    def run(self, env):
        init = env.initial_state
        h = History(env.initial_obs)
        pool = Pool(init, self.cap, self.lam, self.scorer, self.scorer_weight)
        explorer = Explorer(self.seed, context=True)
        rng = random.Random(self.seed)
        trace = [{"type": "start", "world": env.public_world}]
        pool.refresh(h)
        queue, expect = [], None
        while not env.done and env.steps < self.budget:
            if not queue:
                queue, expect, why, info = self.choose(pool, h, explorer, rng, env.steps)
                trace.append({"type": "decide", "t": env.steps, "why": why, **info})
            a = queue.pop(0)
            explorer.record(from_obs(init, h.last), a)
            obs, _ = env.step(a)
            h.add(a, obs)
            dead, side_wrong = pool.observe(a, obs)
            seen = [c for c in changes(h.before(len(h.actions) - 1), obs) if not c.startswith("agent moved")]
            if seen or dead or side_wrong or not pool.members:
                pool.refresh(h, side=bool(seen) or side_wrong, rank=bool(seen))
            if expect is not None and obs != expect.pop(0):
                queue, expect = [], None   # surprised: the plan was made on wrong beliefs
            trace.append({"type": "step", "t": env.steps, "action": a, "changes": seen, "eliminated": dead,
                          "members": len(pool.members)})
        ranked = pool.ranked()
        best = ranked[0][0] if ranked else None
        trace.append({"type": "end", "solved": env.done, "steps": env.steps})
        return result(env, searches=pool.searches, search_seconds=round(pool.search_seconds, 1),
                      llm_calls=pool.calls, map_rules=list(best.rules) if best else None,
                      map_door=list(best.door) if best else None, trace=trace)

    def choose(self, pool, h, explorer, rng, steps):
        now = from_obs(pool.init, h.last)
        ranked = pool.ranked()
        info = {"members": len(ranked)}
        if ranked:
            info["map"] = " AND ".join(english(r) for r in ranked[0][0].door)
            info["map_p"] = round(ranked[0][1], 3)
        if self.detail:
            info["top"] = [{"rule": " AND ".join(english(r) for r in m.door), "p": round(p, 4)}
                           for m, p in ranked[: self.detail]]
        plans = self.goal_plans(pool, ranked, now, steps)

        if self.selector == "map":
            if plans and ranked and plans[0][1] is ranked[0][0]:
                mac = plans[0][0]
                self.score([mac], ranked[:1], pool.init)
                return mac.actions, mac.expect, f"follow the most probable model's plan ({mac.cost} steps)", info
            return self.explore(explorer, now, info)

        macros = [p for p, _ in plans] + experiments(now, pairs=self.pairs, nodes=self.plan_nodes)
        self.score(macros, ranked, pool.init)
        likely = [m for m in macros if m.name.startswith("goal") and m.p_goal >= self.tau]
        value = lambda m: (m.gain + self.goal_bonus * m.p_goal) / m.cost  # noqa: E731
        if likely:
            pick = max(likely, key=lambda m: m.p_goal / m.cost)
        elif self.selector == "random":
            exps = [m for m in macros if not m.name.startswith("goal")]
            pick = rng.choice(exps) if exps else None
        else:
            pick = max(macros, key=value, default=None)
            if pick and pick.gain <= 1e-9 and pick.p_goal <= 1e-9:
                pick = None
        if pick is None:
            return self.explore(explorer, now, info)
        info.update(gain=round(pick.gain, 3), p_goal=round(pick.p_goal, 3))
        if self.detail:
            info["options"] = [{"name": m.name, "cost": m.cost, "gain": round(m.gain, 3), "p_goal": round(m.p_goal, 3),
                                "chosen": m is pick} for m in sorted(macros, key=value, reverse=True)[: self.detail]]
        why = f"{pick.name} ({pick.cost} steps, {pick.gain:.2f} bits, P(goal) {pick.p_goal:.2f})"
        return list(pick.actions), list(pick.expect) if pick.expect else None, why, info

    def explore(self, explorer, now, info):
        path = bfs(now, (), max_nodes=self.plan_nodes)
        if path:
            return path, None, f"walk to the goal with known physics ({len(path)} steps)", info
        a, why = explorer.next_action(now)
        return [a], None, why, info

    def goal_plans(self, pool, ranked, now, steps):
        out, seen = [], set()
        for m, _ in ranked[: self.top_plans]:
            plan = bfs(m.state, m.rules, max_nodes=self.plan_nodes)
            if plan is None and steps > 0:
                fresh = bfs(pool.init, m.rules, max_nodes=self.plan_nodes)
                plan = None if fresh is None else [RESET] + fresh
            if plan is None or tuple(plan) in seen:
                continue
            seen.add(tuple(plan))
            name = "goal plan of a model (" + " AND ".join(english(r) for r in m.door) + ")"
            out.append((Macro(name, plan), m))
        path = bfs(now, (), max_nodes=self.plan_nodes)
        if path and tuple(path) not in seen:
            out.append((Macro("goal: walk with known physics", path), None))
        return out

    def score(self, macros, ranked, init):
        for mac in macros:
            parts = defaultdict(float)
            goal = 0.0
            for m, w in ranked:
                s, seen = m.state, []
                for a in mac.actions:
                    s = init if a == RESET else step(s, a, m.rules)
                    seen.append(observe(s))
                parts[tuple(seen)] += w
                goal += w * s.done
            mac.gain = -sum(p * math.log2(p) for p in parts.values() if p > 0)
            mac.p_goal = goal
            mac.expect = list(max(parts.items(), key=lambda kv: kv[1])[0]) if parts else None

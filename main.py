from collections import deque
from itertools import product
import random

TYPES = {"S": "switch", "D": "door", "P": "plate"}

MOVES = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
ACTIONS = list(MOVES) + ["interact"]


def load(m):
    walls = set()
    objs = {}
    agent = None
    goal = None
    counts = {}
    for y, row in enumerate(m):
        for x, c in enumerate(row):
            if c == "#":
                walls.add((x, y))
            elif c == "A":
                agent = (x, y)
            elif c == "G":
                goal = (x, y)
            elif c in TYPES:
                counts[c] = counts.get(c, 0) + 1
                o = {"type": TYPES[c], "pos": (x, y)}
                if c == "S":
                    o["on"] = False
                if c == "D":
                    o["open"] = False
                objs[c + str(counts[c])] = o
    return walls, goal, {"agent": agent, "objs": objs, "done": False}


def clone(state):
    return {
        "agent": state["agent"],
        "objs": {k: dict(o) for k, o in state["objs"].items()},
        "done": state["done"],
    }


def step(walls, goal, state, action, rules):
    s = clone(state)
    events = []
    if action == "interact":
        for name, o in s["objs"].items():
            if o["pos"] == s["agent"]:
                events.append(("interact", name))
                if o["type"] == "switch":
                    o["on"] = not o["on"]
    else:
        dx, dy = MOVES[action]
        x, y = s["agent"]
        nxt = (x + dx, y + dy)
        if nxt in walls:
            return s
        for o in s["objs"].values():
            if o["type"] == "door" and o["pos"] == nxt and not o["open"]:
                return s
        s["agent"] = nxt
        for name, o in s["objs"].items():
            if o["pos"] == nxt:
                events.append(("enter", name))
    for event, obj, target, prop, value in rules:
        if (event, obj) in events:
            s["objs"][target][prop] = value
    if s["agent"] == goal:
        s["done"] = True
    return s


def key(state):
    objs = tuple((k, tuple(sorted(o.items()))) for k, o in sorted(state["objs"].items()))
    return state["agent"], objs, state["done"]


def bfs(walls, goal, start, rules):
    seen = {key(start)}
    q = deque([(start, [])])
    while q:
        s, path = q.popleft()
        if s["done"]:
            return path
        for a in ACTIONS:
            n = step(walls, goal, s, a, rules)
            k = key(n)
            if k in seen:
                continue
            seen.add(k)
            q.append((n, path + [a]))
    return None


def candidates(objs):
    doors = [k for k in objs if objs[k]["type"] == "door"]
    others = [k for k in objs if objs[k]["type"] != "door"]
    options = []
    for door in doors:
        rules = []
        for event in ["interact", "enter"]:
            for name in others:
                rules.append((event, name, door, "open", True))
        options.append(rules)
    return options


def pick(strategy, walls, goal, state, hyps, rng):
    if strategy == "shortest":
        best = None
        for h in list(hyps):
            plan = bfs(walls, goal, state, h)
            if plan is None:
                hyps.remove(h)
            elif best is None or len(plan) < len(best[1]):
                best = (h, plan)
        return best
    order = list(hyps)
    if strategy == "random":
        rng.shuffle(order)
    for h in order:
        plan = bfs(walls, goal, state, h)
        if plan is None:
            hyps.remove(h)
        else:
            return h, plan
    return None


def solve(m, true_rules, strategy="first", seed=0, verbose=False):
    rng = random.Random(seed)
    walls, goal, state = load(m)
    hyps = list(product(*candidates(state["objs"])))
    if verbose:
        print(len(hyps), "hypotheses")
    steps = 0
    while not state["done"] and steps < 100:
        chosen = pick(strategy, walls, goal, state, hyps, rng)
        if chosen is None:
            if verbose:
                print("no hypotheses left")
            break
        h, plan = chosen
        if verbose:
            print("trying", h, plan)
        for a in plan:
            new = step(walls, goal, state, a, true_rules)
            steps += 1
            hyps = [x for x in hyps if step(walls, goal, state, a, x) == new]
            state = new
            if h not in hyps:
                if verbose:
                    print("  wrong:", h)
                break
    if verbose:
        print("done" if state["done"] else "failed", "in", steps, "steps, left:", hyps)
    return state["done"], steps

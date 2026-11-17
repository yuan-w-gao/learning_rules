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


def move(walls, state, action):
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
            return s, events
        for o in s["objs"].values():
            if o["type"] == "door" and o["pos"] == nxt and not o["open"]:
                return s, events
        s["agent"] = nxt
        for name, o in s["objs"].items():
            if o["pos"] == nxt:
                events.append(("enter", name))
    return s, events


def step(walls, goal, state, action, rules):
    s, events = move(walls, state, action)
    for event, obj, target, prop, value, cond in rules:
        if (event, obj) not in events:
            continue
        if cond is not None:
            cobj, cprop, cvalue = cond
            if s["objs"][cobj][cprop] != cvalue:
                continue
        s["objs"][target][prop] = value
    if s["agent"] == goal:
        s["done"] = True
    return s


def run(walls, goal, state, plan, rules):
    for a in plan:
        state = step(walls, goal, state, a, rules)
    return state


def key(state):
    objs = tuple((k, tuple(sorted(o.items()))) for k, o in sorted(state["objs"].items()))
    return state["agent"], objs, state["done"]

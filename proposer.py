from itertools import product

from planner import bfs
from rules import conditions
from world import ACTIONS, move, run, step


def switches(state):
    return tuple(sorted((k, o["on"]) for k, o in state["objs"].items() if o["type"] == "switch"))


def propose(walls, before, action, door, conds):
    s, events = move(walls, before, action)
    out = []
    for event, obj in events:
        if s["objs"][obj]["type"] == "door":
            continue
        out.append((event, obj, door, "open", True, None))
        if conds:
            for c in conditions(s["objs"]):
                cobj, cprop, cvalue = c
                if s["objs"][cobj][cprop] == cvalue:
                    out.append((event, obj, door, "open", True, c))
    return out


def consistent(walls, goal, rule, door, history):
    for before, action, after in history:
        pred = step(walls, goal, before, action, [rule])
        if pred["objs"][door]["open"] != after["objs"][door]["open"]:
            return False
    return True


def update(walls, goal, cands, history, conds):
    before, action, after = history[-1]
    for door in cands:
        if cands[door] is None:
            if after["objs"][door]["open"] and not before["objs"][door]["open"]:
                new = propose(walls, before, action, door, conds)
                cands[door] = [r for r in new if consistent(walls, goal, r, door, history)]
        else:
            cands[door] = [r for r in cands[door] if consistent(walls, goal, r, door, history[-1:])]


def hypotheses(cands):
    options = [c if c is not None else [None] for c in cands.values()]
    return [tuple(r for r in combo if r is not None) for combo in product(*options)]


def tried(walls, history):
    out = set()
    for before, action, after in history:
        s, events = move(walls, before, action)
        for e in events:
            out.add((e, switches(s)))
    return out


def untried(walls, state, done):
    for a in ACTIONS:
        s, events = move(walls, state, a)
        for e in events:
            if s["objs"][e[1]]["type"] != "door" and (e, switches(s)) not in done:
                return a
    return None


def explore(walls, goal, state, history):
    done = tried(walls, history)
    path = bfs(walls, goal, state, [], lambda s: untried(walls, s, done) is not None)
    if path is None:
        return None
    return path + [untried(walls, run(walls, goal, state, path, []), done)]

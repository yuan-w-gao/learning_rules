from itertools import product

from planner import bfs
from rules import conditions
from world import ACTIONS, move, run, step


def switches(state):
    return tuple(sorted((k, o["on"]) for k, o in state["objs"].items() if o["type"] == "switch"))


def propose(walls, before, action, door, feats):
    s, events = move(walls, before, action)
    effects = ["toggle"] if "toggles" in feats else []
    if not before["objs"][door]["open"]:
        effects = [True] + effects
    elif "traps" in feats:
        effects = [False] + effects
    out = []
    for event, obj in events:
        if s["objs"][obj]["type"] == "door":
            continue
        for value in effects:
            out.append((event, obj, door, "open", value, None))
            if "conds" in feats:
                for c in conditions(s["objs"]):
                    cobj, cprop, cvalue = c
                    if s["objs"][cobj][cprop] == cvalue:
                        out.append((event, obj, door, "open", value, c))
    return out


def consistent(walls, goal, rules, door, history):
    for before, action, after in history:
        pred = step(walls, goal, before, action, rules)
        if pred["objs"][door]["open"] != after["objs"][door]["open"]:
            return False
    return True


def update(walls, goal, cands, history, feats):
    before, action, after = history[-1]
    for door in cands:
        changed = after["objs"][door]["open"] != before["objs"][door]["open"]
        if cands[door] is None:
            if changed:
                new = propose(walls, before, action, door, feats)
                cands[door] = [(r,) for r in new if consistent(walls, goal, (r,), door, history)]
            continue
        kept = [c for c in cands[door] if consistent(walls, goal, c, door, history[-1:])]
        if not kept and changed and "traps" in feats:
            new = propose(walls, before, action, door, feats)
            kept = [c + (r,) for c in cands[door] for r in new if consistent(walls, goal, c + (r,), door, history)]
        cands[door] = kept


def hypotheses(cands):
    options = [c if c is not None else [()] for c in cands.values()]
    return [tuple(r for rules in combo for r in rules) for combo in product(*options)]


def tried(walls, history):
    out = set()
    for before, action, after in history:
        s, events = move(walls, before, action)
        for e in events:
            out.add((e, switches(s)))
    return out


def untried(walls, state, done, by_setting):
    seen = {e for e, setting in done}
    covered = {(e, k, v) for e, setting in done for k, v in setting}
    for a in ACTIONS:
        s, events = move(walls, state, a)
        for e in events:
            if s["objs"][e[1]]["type"] == "door":
                continue
            if by_setting and any((e, k, v) not in covered for k, v in switches(s)):
                return a
            if not by_setting and e not in seen:
                return a
    return None


def explore(walls, goal, state, history, feats, rules):
    done = tried(walls, history)
    for by_setting in [False, True] if "conds" in feats else [False]:
        path = bfs(walls, goal, state, rules, lambda s: untried(walls, s, done, by_setting) is not None)
        if path is not None:
            return path + [untried(walls, run(walls, goal, state, path, rules), done, by_setting)]
    return None


def possible(walls, goal, state, cands):
    rules = [r[:4] + (True,) + r[5:] for c in cands.values() if c for rules in c for r in rules if r[4] is not False]
    return bfs(walls, goal, state, rules)

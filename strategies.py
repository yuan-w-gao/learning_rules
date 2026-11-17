from planner import bfs
from world import key, run


def experiments(walls, goal, state, hyps):
    out = []
    for name, o in state["objs"].items():
        if o["type"] == "door":
            continue
        path = bfs(walls, goal, state, [], lambda s, p=o["pos"]: s["agent"] == p)
        if path is None:
            continue
        plans = [path + ["interact"]]
        if path:
            plans.append(path)
        for plan in plans:
            groups = {}
            for h in hyps:
                k = key(run(walls, goal, state, plan, h))
                groups[k] = groups.get(k, 0) + 1
            score = sum(g * g for g in groups.values()) / len(hyps)
            if len(groups) > 1:
                out.append((score, plan))
    return out


def try_in_order(walls, goal, state, hyps, order):
    for h in order:
        plan = bfs(walls, goal, state, h)
        if plan is None:
            hyps.remove(h)
        else:
            return h, plan
    return None


def first(walls, goal, state, hyps, rng):
    return try_in_order(walls, goal, state, hyps, list(hyps))


def shuffled(walls, goal, state, hyps, rng):
    order = list(hyps)
    rng.shuffle(order)
    return try_in_order(walls, goal, state, hyps, order)


def shortest(walls, goal, state, hyps, rng):
    best = None
    for h in list(hyps):
        plan = bfs(walls, goal, state, h)
        if plan is None:
            hyps.remove(h)
        elif best is None or len(plan) < len(best[1]):
            best = (h, plan)
    return best


def best_experiment(walls, goal, state, hyps, rng, rank):
    exps = experiments(walls, goal, state, hyps)
    if not exps:
        return first(walls, goal, state, hyps, rng)
    score, plan = min(exps, key=rank)
    return None, plan


def info(walls, goal, state, hyps, rng):
    return best_experiment(walls, goal, state, hyps, rng, lambda e: (e[0], len(e[1])))


def nearest(walls, goal, state, hyps, rng):
    return best_experiment(walls, goal, state, hyps, rng, lambda e: (len(e[1]), e[0]))


STRATEGIES = {
    "first": first,
    "random": shuffled,
    "shortest": shortest,
    "info": info,
    "nearest": nearest,
}


def pick(strategy, walls, goal, state, hyps, rng):
    if not hyps:
        return None
    return STRATEGIES[strategy](walls, goal, state, hyps, rng)

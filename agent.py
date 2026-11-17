from itertools import product
import random

from proposer import explore, hypotheses, possible, update
from rules import candidates, features
from strategies import pick
from world import key, step


def solve(env, strategy="first", seed=0, feats=(), verbose=False):
    feats = features(feats)
    rng = random.Random(seed)
    walls, goal, state = env.public()
    hyps = list(product(*candidates(state["objs"], feats)))
    most = len(hyps)
    if verbose:
        print(len(hyps), "hypotheses")
    while not state["done"] and env.steps < env.budget:
        chosen = pick(strategy, walls, goal, state, hyps, rng)
        if chosen is None:
            if verbose:
                print("no hypotheses left")
            break
        h, plan = chosen
        if verbose:
            print("trying", h or "experiment", plan)
        for a in plan:
            if env.steps >= env.budget:
                break
            new = env.step(a)
            hyps = [x for x in hyps if step(walls, goal, state, a, x) == new]
            state = new
            if h is not None and h not in hyps:
                if verbose:
                    print("  wrong:", h)
                break
    if verbose:
        print("done" if state["done"] else "failed", "in", env.steps, "steps, left:", hyps)
    return state["done"], env.steps, most


def solve_proposing(env, strategy="first", seed=0, feats=(), verbose=False):
    feats = features(feats)
    rng = random.Random(seed)
    walls, goal, state = env.public()
    cands = {k: None for k, o in state["objs"].items() if o["type"] == "door"}
    start = key(state)
    history = []
    most = 0
    while not state["done"] and env.steps < env.budget:
        if any(c == [] for c in cands.values()):
            if verbose:
                print("no hypotheses left")
            break
        hyps = hypotheses(cands)
        most = max(most, len(hyps))
        chosen = None
        if possible(walls, goal, state, cands) is not None:
            chosen = pick(strategy, walls, goal, state, hyps, rng)
        if chosen is None:
            h, plan = hyps[0], explore(walls, goal, state, history, feats, hyps[0])
            if plan is None and "traps" in feats and key(state) != start:
                h, plan = None, ["reset"]
            if plan is None:
                if verbose:
                    print("nothing left to try")
                break
            if verbose:
                print("exploring", plan)
        else:
            h, plan = chosen
            if verbose:
                print("trying", h or "experiment", plan, f"({len(hyps)} hypotheses)")
        for a in plan:
            if env.steps >= env.budget:
                break
            new = env.step(a)
            if a != "reset":
                history.append((state, a, new))
                update(walls, goal, cands, history, feats)
            state = new
            if h is not None and h not in hypotheses(cands):
                if verbose:
                    print("  wrong:", h)
                break
    if verbose:
        print("done" if state["done"] else "failed", "in", env.steps, "steps, candidates:", cands)
    return state["done"], env.steps, most

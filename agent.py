from itertools import product
import random

from rules import candidates
from strategies import pick
from world import load, step


def solve(m, true_rules, strategy="first", seed=0, conds=False, verbose=False):
    rng = random.Random(seed)
    walls, goal, state = load(m)
    hyps = list(product(*candidates(state["objs"], conds)))
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
            print("trying", h or "experiment", plan)
        for a in plan:
            new = step(walls, goal, state, a, true_rules)
            steps += 1
            hyps = [x for x in hyps if step(walls, goal, state, a, x) == new]
            state = new
            if h is not None and h not in hyps:
                if verbose:
                    print("  wrong:", h)
                break
    if verbose:
        print("done" if state["done"] else "failed", "in", steps, "steps, left:", hyps)
    return state["done"], steps

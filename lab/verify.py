import random
from collections import deque

from .rules import split
from .sim import ACTIONS, MOVES, RESET, observe, step
from .worlds import initial_state, world_rules

PROBE_ACTIONS = list(MOVES) + ["interact"] * 3 + ["wait"]


def door_view(s):
    return s.agent, s.done, tuple(sorted((o.id, o.props.get("open", False)) for o in s.objs.values() if o.type == "door"))


def equivalent(init, a, b, view=observe, max_pairs=20_000):
    # joint BFS over (state under a, state under b); any reachable difference in view = not equivalent.
    # Running out of pairs counts as equivalent (nothing found).
    start = (init, init)
    seen = {(init.key(), init.key())}
    q = deque([start])
    while q and len(seen) < max_pairs:
        sa, sb = q.popleft()
        for act in ACTIONS:
            na, nb = step(sa, act, a), step(sb, act, b)
            if view(na) != view(nb):
                return False
            k = (na.key(), nb.key())
            if k not in seen:
                seen.add(k)
                q.append((na, nb))
    return True


def probes(world, n=40, horizon=40, seed=0):
    rng = random.Random(seed)
    rules = world_rules(world)
    out = []
    for _ in range(n):
        s = initial_state(world)
        walk = []
        for _ in range(horizon):
            a = rng.choice(PROBE_ACTIONS)
            s = step(s, a, rules)
            walk.append((a, door_view(s)))
        out.append(walk)
    return out


def passes_probes(world, rules, walks):
    for walk in walks:
        s = initial_state(world)
        for a, want in walk:
            s = step(s, a, rules)
            if door_view(s) != want:
                return False
    return True


def consistent(world, door, actions):
    # candidate door rules + the true side rules reproduce the door through the history
    truth = world_rules(world)
    side = split(truth)[1]
    rules = side + tuple(door)
    init = initial_state(world)
    s, real = init, init
    for a in actions:
        if a == RESET:
            s, real = init, init
            continue
        s, real = step(s, a, rules), step(real, a, truth)
        if door_view(s) != door_view(real):
            return False
    return True


def correct(world, door, walks=None):
    truth = world_rules(world)
    rules = split(truth)[1] + tuple(door)
    walks = walks if walks is not None else probes(world)
    return passes_probes(world, rules, walks) and equivalent(initial_state(world), rules, truth, door_view, 10_000)


def probe_accuracy(world, rules, n=200, horizon=40, seed=0):
    rng = random.Random(seed)
    truth = world_rules(world)
    hits = 0
    for _ in range(n):
        a_state = b_state = initial_state(world)
        for _ in range(rng.randint(1, horizon)):
            act = rng.choice(PROBE_ACTIONS)
            a_state, b_state = step(a_state, act, rules), step(b_state, act, truth)
        hits += observe(a_state) == observe(b_state)
    return hits / n

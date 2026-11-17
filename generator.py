import random

from planner import bfs
from rules import candidates, conditions, features
from world import load


def generate(seed, doors=2, feats=()):
    feats = features(feats)
    rng = random.Random(seed)
    crng = random.Random(f"{seed}-conds")
    trng = random.Random(f"{seed}-toggles")
    xrng = random.Random(f"{seed}-traps")
    w, h = 5 + 5 * doors, 6
    rooms = [[(x, y) for x in range(1, 6) for y in range(1, h - 1)]]
    for i in range(1, doors):
        rooms.append([(x, y) for x in range(2 + 5 * i, 6 + 5 * i) for y in range(1, h - 1)])
    last = [(x, y) for x in range(2 + 5 * doors, 4 + 5 * doors) for y in range(1, h - 1)]
    inner = [p for room in rooms for p in room]
    tries = 0
    while True:
        tries += 1
        grid = [["#"] * w for _ in range(h)]
        for x, y in inner + last:
            grid[y][x] = "."
        for i in range(doors):
            grid[rng.randint(1, h - 2)][6 + 5 * i] = "D"
        x, y = rng.choice(rooms[0])
        grid[y][x] = "A"
        x, y = rng.choice(last)
        grid[y][x] = "G"
        free = [(x, y) for x, y in inner if grid[y][x] == "."]
        for c in "S" * doors + "P" * (doors - 1):
            x, y = rng.choice(free)
            free.remove((x, y))
            grid[y][x] = c
        m = ["".join(row) for row in grid]
        walls, goal, state = load(m)
        rules = [rng.choice(r) for r in candidates(state["objs"])]
        if "conds" in feats:
            rules = [r[:5] + (crng.choice(conditions(state["objs"])),) if crng.random() < 0.5 else r for r in rules]
        if "toggles" in feats:
            rules = [r[:4] + ("toggle",) + r[5:] if trng.random() < 0.5 else r for r in rules]
        if "traps" in feats:
            objs = state["objs"]
            door = xrng.choice([k for k in objs if objs[k]["type"] == "door"])
            beyond = [k for k in objs if objs[k]["type"] != "door" and objs[k]["pos"][0] > objs[door]["pos"][0]]
            if not beyond:
                continue
            rules = rules + [(xrng.choice(["enter", "interact"]), xrng.choice(beyond), door, "open", False, None)]
        if bfs(walls, goal, state, rules) is None:
            continue
        if bfs(walls, goal, state, []) is not None:
            continue
        return m, rules, tries

import random

from main import bfs, candidates, load


def generate(seed):
    rng = random.Random(seed)
    w, h = 15, 6
    room1 = [(x, y) for x in range(1, 6) for y in range(1, h - 1)]
    room2 = [(x, y) for x in range(7, 11) for y in range(1, h - 1)]
    room3 = [(x, y) for x in range(12, 14) for y in range(1, h - 1)]
    tries = 0
    while True:
        tries += 1
        grid = [["#"] * w for _ in range(h)]
        for x, y in room1 + room2 + room3:
            grid[y][x] = "."
        grid[rng.randint(1, h - 2)][6] = "D"
        grid[rng.randint(1, h - 2)][11] = "D"
        x, y = rng.choice(room1)
        grid[y][x] = "A"
        x, y = rng.choice(room3)
        grid[y][x] = "G"
        free = [(x, y) for x, y in room1 + room2 if grid[y][x] == "."]
        for c in "SSP":
            x, y = rng.choice(free)
            free.remove((x, y))
            grid[y][x] = c
        m = ["".join(row) for row in grid]
        walls, goal, state = load(m)
        rules = [rng.choice(r) for r in candidates(state["objs"])]
        if bfs(walls, goal, state, rules) is None:
            continue
        if bfs(walls, goal, state, []) is not None:
            continue
        return m, rules, tries

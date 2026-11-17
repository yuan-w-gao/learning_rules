from collections import deque

from world import ACTIONS, key, step


def bfs(walls, goal, start, rules, stop=lambda s: s["done"]):
    seen = {key(start)}
    q = deque([(start, [])])
    while q:
        s, path = q.popleft()
        if stop(s):
            return path
        for a in ACTIONS:
            n = step(walls, goal, s, a, rules)
            k = key(n)
            if k in seen:
                continue
            seen.add(k)
            q.append((n, path + [a]))
    return None

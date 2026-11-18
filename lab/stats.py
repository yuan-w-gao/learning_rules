import random

GROUPS = [("tier 1", {1}), ("tiers 2-3", {2, 3}), ("tier 4", {4}), ("tiers 1-4", {1, 2, 3, 4}), ("held-out", {5})]


def cost(row, budget=100):
    # a failure costs the whole budget, so giving up early can't look fast
    return row["steps"] if row["solved"] else budget


def bootstrap(xs, n=2000, seed=0):
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return sum(xs) / len(xs), means[int(0.025 * n)], means[int(0.975 * n)]


def paired(rows, a, b, tiers, budget=100, n=5000):
    by = {}
    for r in rows:
        if r["tier"] in tiers and r["agent"] in (a, b):
            by.setdefault((r["family"], r["seed"]), {})[r["agent"]] = cost(r, budget)
    diffs = [d[a] - d[b] for d in by.values() if a in d and b in d]
    if not diffs:
        return None
    rng = random.Random(0)
    boot = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n))
    return dict(n=len(diffs), mean=sum(diffs) / len(diffs), lo=boot[int(0.025 * n)], hi=boot[int(0.975 * n)],
                wins=sum(d < 0 for d in diffs), losses=sum(d > 0 for d in diffs))

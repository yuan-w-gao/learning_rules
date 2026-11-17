import time

from generator import generate
from main import solve

STRATEGIES = ["first", "random", "shortest", "info", "nearest"]


def evaluate(n):
    worlds = [generate(seed)[:2] for seed in range(n)]
    for strategy in STRATEGIES:
        start = time.time()
        solved = []
        failed = []
        for seed, (m, rules) in enumerate(worlds):
            ok, steps = solve(m, rules, strategy, seed)
            if ok:
                solved.append(steps)
            else:
                failed.append(seed)
        took = time.time() - start
        print(
            f"{strategy:9} solved {len(solved)}/{n}, avg steps {sum(solved) / len(solved):.1f}, "
            f"max {max(solved)}, {took:.1f}s"
        )
        if failed:
            print("  failed seeds:", failed)


if __name__ == "__main__":
    m, rules, tries = generate(0)
    print("\n".join(m))
    print("rules:", rules, "tries:", tries)
    for strategy in STRATEGIES:
        print()
        print(strategy)
        solve(m, rules, strategy, verbose=True)
    print()
    evaluate(100)

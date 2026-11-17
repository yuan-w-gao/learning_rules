import time

from generator import generate
from main import solve

STRATEGIES = ["first", "random", "shortest", "info", "nearest"]


def evaluate(n, world_conds=False, agent_conds=False, strategies=STRATEGIES):
    print(f"worlds with conditions: {world_conds}, agent guesses conditions: {agent_conds}")
    worlds = [generate(seed, world_conds)[:2] for seed in range(n)]
    for strategy in strategies:
        start = time.time()
        solved = []
        failed = []
        for seed, (m, rules) in enumerate(worlds):
            ok, steps = solve(m, rules, strategy, seed, agent_conds)
            if ok:
                solved.append(steps)
            else:
                failed.append(seed)
        took = time.time() - start
        line = f"{strategy:9} solved {len(solved)}/{n}"
        if solved:
            line += f", avg steps {sum(solved) / len(solved):.1f}, max {max(solved)}"
        print(line + f", {took:.1f}s")
        if failed:
            print("  failed seeds:", failed[:10], "..." if len(failed) > 10 else "")
    print()


if __name__ == "__main__":
    m, rules, tries = generate(0)
    print("\n".join(m))
    print("rules:", rules, "tries:", tries)
    for strategy in STRATEGIES:
        print()
        print(strategy)
        solve(m, rules, strategy, verbose=True)
    print()
    fast = [s for s in STRATEGIES if s != "shortest"]
    evaluate(100)
    evaluate(100, world_conds=True)
    evaluate(100, world_conds=True, agent_conds=True, strategies=fast)
    evaluate(100, world_conds=False, agent_conds=True, strategies=fast)

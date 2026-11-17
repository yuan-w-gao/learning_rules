import time

from generator import generate
from agent import solve, solve_proposing
from env import Env
from strategies import STRATEGIES

SOLVERS = {"enumerate": solve, "propose": solve_proposing}


def evaluate(n, solver="enumerate", world_conds=False, agent_conds=False, strategies=STRATEGIES):
    print(f"{solver}: worlds with conditions: {world_conds}, agent guesses conditions: {agent_conds}")
    worlds = [generate(seed, world_conds)[:2] for seed in range(n)]
    for strategy in strategies:
        start = time.time()
        solved = []
        failed = []
        for seed, (m, rules) in enumerate(worlds):
            ok, steps = SOLVERS[solver](Env(m, rules), strategy, seed, agent_conds)
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
    for solver in SOLVERS:
        print()
        print(solver, "first")
        SOLVERS[solver](Env(m, rules), "first", verbose=True)
    print()
    fast = [s for s in STRATEGIES if s != "shortest"]
    for world_conds, agent_conds in [(False, False), (True, False), (True, True), (False, True)]:
        evaluate(100, "enumerate", world_conds, agent_conds, fast if agent_conds else STRATEGIES)
        evaluate(100, "propose", world_conds, agent_conds)

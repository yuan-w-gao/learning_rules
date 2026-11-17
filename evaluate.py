import time

from generator import generate
from agent import solve, solve_proposing
from env import Env
from strategies import STRATEGIES

SOLVERS = {"enumerate": solve, "propose": solve_proposing}


def evaluate(n, solver="enumerate", world=(), agent=(), strategies=STRATEGIES, doors=2):
    print(f"{solver}: {doors} doors, world {sorted(world)}, agent guesses {sorted(agent)}")
    worlds = [generate(seed, doors, world)[:2] for seed in range(n)]
    for strategy in strategies:
        start = time.time()
        solved = []
        failed = []
        most = 0
        resets = 0
        for seed, (m, rules) in enumerate(worlds):
            env = Env(m, rules, 50 * doors)
            ok, steps, h = SOLVERS[solver](env, strategy, seed, agent)
            resets += env.resets
            most = max(most, h)
            if ok:
                solved.append(steps)
            else:
                failed.append(seed)
        took = time.time() - start
        line = f"{strategy:9} solved {len(solved)}/{n}"
        if solved:
            line += f", avg steps {sum(solved) / len(solved):.1f}, max {max(solved)}"
        print(line + f", most hypotheses {most}, resets {resets}, {took:.1f}s")
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
    for world, agent in [((), ()), (("conds",), ()), (("conds",), ("conds",)), ((), ("conds",))]:
        evaluate(100, "enumerate", world, agent, fast if agent else STRATEGIES)
        evaluate(100, "propose", world, agent)

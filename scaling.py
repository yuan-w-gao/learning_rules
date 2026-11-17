import math

from evaluate import evaluate
from generator import generate
from rules import candidates
from world import load

LIMIT = 1_000_000

if __name__ == "__main__":
    for doors in (2, 3, 4):
        n = 50 if doors < 4 else 20
        for feats in [(), ("conds",)]:
            m, _, _ = generate(0, doors, feats)
            count = math.prod(len(c) for c in candidates(load(m)[2]["objs"], feats))
            if count <= LIMIT:
                evaluate(n, "enumerate", feats, feats, ["first"], doors)
            else:
                print(f"enumerate: {doors} doors, {list(feats)}: skipped, {count:,} hypotheses\n")
            evaluate(n, "propose", feats, feats, ["first"], doors)

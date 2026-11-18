"""Try to find the rules yourself.  python scripts/play.py --tier 2 --seed 3
keys: w a s d move, e interact, . wait, r reset, q quit (then enter)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.env import Env  # noqa: E402
from lab.sim import changes  # noqa: E402
from lab.worlds import draw, generate  # noqa: E402

KEYS = {"w": "up", "s": "down", "a": "left", "d": "right", "e": "interact", ".": "wait", "r": "reset"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--family", default=None)
    args = ap.parse_args()
    w = generate(args.tier, args.seed, args.family)
    env = Env(w)
    print(f"{w['name']}: open the door D and reach G. Lowercase = on/open. Best possible: {w['optimal']} steps.")
    obs = env.initial_obs
    while not env.done and env.steps < env.budget:
        print(draw(w, env._state))      # peeking at the state is fine for a human playing
        k = input("> ").strip()
        if k == "q":
            break
        if k not in KEYS:
            continue
        before = obs
        obs, _ = env.step(KEYS[k])
        print(", ".join(c for c in changes(before, obs) if not c.startswith("agent")) or "")
    print(f"{'solved' if env.done else 'not solved'} in {env.steps} steps. The rules were:\n{w['rules']}")


if __name__ == "__main__":
    main()

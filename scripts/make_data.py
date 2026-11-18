"""Training data from search episodes on training seeds (never tier 5).

scorer:   snapshots of (world, history, door candidates, correct?) for the yes/no scorer
proposer: snapshots of (world, history, true door rules as text) for SFT / GRPO
"""
import argparse
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.agents import History, SearchAgent  # noqa: E402
from lab.env import Env  # noqa: E402
from lab.prompts import history_text, world_text  # noqa: E402
from lab.rules import english, split, text, total_size  # noqa: E402
from lab.search import door_search  # noqa: E402
from lab.sim import changes  # noqa: E402
from lab.verify import correct, probes  # noqa: E402
from lab.worlds import FAMILIES, generate, public, world_rules  # noqa: E402

TRAIN_FAMILIES = [f for f, (t, _) in FAMILIES.items() if t <= 4] + ["random"]


def snapshots(job):
    kind, agent, family, seed, per_episode, cap = job
    tier = FAMILIES[family][0]
    world = generate(tier, seed, family)
    env = Env(world)
    SearchAgent(selector=agent.split("_")[1], seed=seed).run(env)
    h = History(env.initial_obs)
    picks = []
    for i, (a, obs) in enumerate(zip(env.actions, env.observations)):
        h.add(a, obs)
        before = env.observations[i - 1] if i else env.initial_obs
        seen = [c for c in changes(before, obs) if not c.startswith("agent moved")]
        if seen and "reached the goal" not in seen:
            picks.append(i + 1)
    rng = random.Random(f"{agent}-{family}-{seed}")
    picks = sorted(rng.sample(picks, min(per_episode, len(picks))))
    rows = []
    wtxt = world_text(public(world))
    walks = probes(world)
    for t in picks:
        hist = History(env.initial_obs)
        for a, obs in zip(env.actions[:t], env.observations[:t]):
            hist.add(a, obs)
        row = {"agent": agent, "tier": tier, "family": family, "seed": seed, "t": t, "world_text": wtxt,
               "history": history_text(hist)}
        if kind == "scorer":
            cands = door_search(env.initial_state, cap=cap).search(hist.actions, hist.observations, hist.initial_obs)
            row["cands"] = [" AND ".join(english(r) for r in c) for c in cands]
            row["dls"] = [total_size(c) for c in cands]
            row["labels"] = [correct(world, c, walks) for c in cands]
        else:
            row["actions"] = hist.actions
            row["target"] = text(split(world_rules(world))[0])
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["scorer", "proposer"])
    ap.add_argument("--agents", nargs="+", default=["search_eig", "search_random"])
    ap.add_argument("--families", nargs="+", default=TRAIN_FAMILIES)
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--seed-offset", type=int, default=100)
    ap.add_argument("--val-from", type=int, default=145, help="seeds >= this go to validation")
    ap.add_argument("--per-episode", type=int, default=4)
    ap.add_argument("--cap", type=int, default=300)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--test", action="store_true", help="evaluation snapshots: tier 5 allowed, split='test', never trained on")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not args.test:
        if any(FAMILIES[f][0] == 5 for f in args.families):
            sys.exit("tier 5 is held out: no training data from it")
        if args.seed_offset < 100:
            sys.exit("training seeds start at 100 so they never overlap the evaluation seeds")
    out = Path(args.out or f"runs/data/{args.kind}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    jobs = [(args.kind, a, f, s, args.per_episode, args.cap) for a in args.agents for f in args.families
            for s in range(args.seed_offset, args.seed_offset + args.seeds)]
    n = 0
    with ProcessPoolExecutor(args.workers) as ex, open(out, "w") as f:
        for rows in ex.map(snapshots, jobs):
            for r in rows:
                r["split"] = "test" if args.test else "val" if r["seed"] >= args.val_from else "train"
                f.write(json.dumps(r) + "\n")
                n += 1
    print(f"{n} snapshots -> {out}")


if __name__ == "__main__":
    main()

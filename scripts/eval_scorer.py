"""How well does the scorer rank the true candidate? Rank key: -lam * description length + w * score.
Needs the scorer served by vLLM (vllm serve <base> --enable-lora --lora-modules scorer=runs/scorer/adapter).
"""
import argparse
import json
import random
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.prompts import SCORER_HEADER  # noqa: E402


def auc(pos, neg):
    allv = sorted((v, i < len(pos)) for i, v in enumerate(pos + neg))
    ranks, i = {}, 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        for k in range(i, j):
            ranks[k] = (i + j + 1) / 2
        i = j
    r_pos = sum(ranks[k] for k, (_, p) in enumerate(allv) if p)
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--model", default="scorer")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--weights", type=float, nargs="+", default=[0, 0.5, 1, 2])
    ap.add_argument("--split", default="val")
    args = ap.parse_args()
    import openai
    client = openai.OpenAI(base_url=args.base_url, api_key="none")

    def score(prompt):
        r = client.completions.create(model=args.model, prompt=prompt, max_tokens=1, temperature=0, logprobs=20)
        top = r.choices[0].logprobs.top_logprobs[0]
        yes = max((v for t, v in top.items() if t.strip().lower() == "yes"), default=-30.0)
        no = max((v for t, v in top.items() if t.strip().lower() == "no"), default=-30.0)
        return yes - no

    rows = [json.loads(l) for l in open(args.data)]
    rows = [r for r in rows if r["split"] == args.split and any(r["labels"])]
    print(f"{len(rows)} snapshots with a correct candidate")
    with ThreadPoolExecutor(args.workers) as ex:
        for r in rows:
            prefix = f"{SCORER_HEADER}\n\n# World\n{r['world_text']}\n\n# History\n{r['history']}\n\n# Candidate\n"
            r["scores"] = list(ex.map(score, [prefix + c + "\n\nIs this the true mechanism? Answer:" for c in r["cands"]]))
    pos = [s for r in rows for s, y in zip(r["scores"], r["labels"]) if y]
    neg = [s for r in rows for s, y in zip(r["scores"], r["labels"]) if not y]
    print(f"AUC of the raw score: {auc(pos, neg):.3f}")
    rng = random.Random(0)
    for w in args.weights:
        firsts = []
        for r in rows:
            trials = []
            for _ in range(20):    # random tie-breaking
                keys = [(-args.lam * d + w * s, rng.random()) for d, s in zip(r["dls"], r["scores"])]
                order = sorted(range(len(keys)), key=lambda i: keys[i], reverse=True)
                trials.append(next(k for k, i in enumerate(order) if r["labels"][i]) + 1)
            firsts.append(sum(trials) / len(trials))
        top = lambda k: sum(f <= k for f in firsts) / len(firsts)  # noqa: E731
        print(f"w={w}: mean rank {statistics.mean(firsts):.1f}, median {statistics.median(firsts):.1f}, "
              f"top1 {top(1):.2f}, top5 {top(5):.2f}, top20 {top(20):.2f}")


if __name__ == "__main__":
    main()

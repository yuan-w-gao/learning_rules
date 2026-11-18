"""Can a model write the door rules directly? Scored by the verifier: parsed / consistent / correct.

  vllm:   python scripts/eval_proposer.py runs/data/proposer.jsonl --model sft --k 8
  claude: python scripts/eval_proposer.py runs/data/proposer.jsonl --claude claude-opus-5-5 --cap-usd 5 [--dry-run]
          (Message Batches API, half price; resumable via a .pending.json next to the answers)
"""
import argparse
import json
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.llm import Ledger, anthropic_key, cost_usd  # noqa: E402
from lab.prompts import door_prompt  # noqa: E402
from lab.proposer import END, judge  # noqa: E402
from lab.worlds import generate  # noqa: E402


def judge_row(a):
    row, text = a
    return judge(generate(row["tier"], row["seed"], row["family"]), row["actions"], text)


def clean(text):
    lines = [l for l in text.splitlines() if not l.strip().startswith("```")]
    return "\n".join(lines).split(END)[0] + END


def report(rows, verdicts, name):
    groups = defaultdict(list)
    for r, v in zip(rows, verdicts):
        groups["held-out (tier 5)" if r["tier"] == 5 else "tiers 1-4"].append(v)
        groups[r["family"]].append(v)
    print(f"--- {name}")
    for g, vs in groups.items():
        print(f"{g:22} n={len(vs):3}  parsed {sum(v['parsed'] for v in vs) / len(vs):.2f}  "
              f"consistent {sum(v['consistent'] for v in vs) / len(vs):.2f}  correct {sum(v['correct'] for v in vs) / len(vs):.2f}")


def vllm_answers(rows, model, n, temperature, url, max_tokens=160, seed=0):
    import openai
    client = openai.OpenAI(base_url=url, api_key="none", timeout=900)

    def one(r):
        resp = client.completions.create(model=model, prompt=door_prompt(r["world_text"], r["history"]), n=n,
                                         temperature=temperature, max_tokens=max_tokens, stop=[END], seed=seed)
        return [c.text for c in resp.choices]

    with ThreadPoolExecutor(64) as ex:
        return list(ex.map(one, rows))


def claude_answers(rows, model, effort, cap, assume_out, dry_run, out, poll):
    import anthropic
    suffix = "\n\nReply with only the rules, one per line, then a line END."
    prompts = [door_prompt(r["world_text"], r["history"]) + suffix for r in rows]
    projected = sum(cost_usd(model, len(p) / 2.3, assume_out, batch=True) for p in prompts)
    ledger = Ledger(run=f"proposer-{model}")
    print(f"{len(rows)} rows, projected ${projected:.2f} (ledger so far ${ledger.total():.2f})")
    if dry_run:
        return None
    if ledger.total() + projected > cap:
        sys.exit(f"would exceed the cap of ${cap}")
    client = anthropic.Anthropic(api_key=anthropic_key())
    pending = out.with_suffix(".pending.json")
    if pending.exists():
        batch_id = json.loads(pending.read_text())["batch_id"]
    else:
        reqs = [{"custom_id": f"row-{i}", "params": {"model": model, "max_tokens": 16000, "output_config": {"effort": effort},
                                                     "messages": [{"role": "user", "content": p}]}}
                for i, p in enumerate(prompts)]
        batch_id = client.messages.batches.create(requests=reqs).id
        pending.write_text(json.dumps({"batch_id": batch_id, "rows": len(rows), "submitted": time.time()}))
    while client.messages.batches.retrieve(batch_id).processing_status != "ended":
        time.sleep(poll)
    answers, tin, tout = [""] * len(rows), 0, 0
    for res in client.messages.batches.results(batch_id):
        i = int(res.custom_id.split("-")[1])
        if res.result.type == "succeeded":
            msg = res.result.message
            answers[i] = next((b.text for b in msg.content if b.type == "text"), "")
            tin, tout = tin + msg.usage.input_tokens, tout + msg.usage.output_tokens
    ledger.record(model, cost_usd(model, tin, tout, batch=True), batch_id=batch_id, rows=len(rows),
                  input_tokens=tin, output_tokens=tout)
    pending.unlink()
    return [[clean(a)] for a in answers]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--model", default="sft")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--claude", default=None)
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--rows", choices=["heldout", "heldout+fam", "all"], default="all")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cap-usd", type=float, default=None)
    ap.add_argument("--assume-out", type=int, default=3000)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data) if json.loads(l)["split"] != "train"]
    if args.rows != "all":
        held = [r for r in rows if r["tier"] == 5]
        if args.rows == "heldout+fam":
            rng = random.Random(0)
            fams = sorted({r["family"] for r in rows if r["tier"] < 5})
            held += [x for f in fams for x in rng.sample([r for r in rows if r["family"] == f],
                                                         min(4, sum(r["family"] == f for r in rows)))]
        rows = held
    if args.limit:
        rows = rows[: args.limit]
    pool = ProcessPoolExecutor()

    if args.claude:
        if args.cap_usd is None and not args.dry_run:
            sys.exit("--cap-usd is required for paid runs")
        out = Path(args.out or f"runs/api/proposer/{args.claude}_{args.effort}.jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)
        answers = claude_answers(rows, args.claude, args.effort, args.cap_usd, args.assume_out, args.dry_run, out, args.poll)
        if answers is None:
            return
        verdicts = list(pool.map(judge_row, [(r, a[0]) for r, a in zip(rows, answers)]))
        with open(out, "w") as f:
            for r, a, v in zip(rows, answers, verdicts):
                f.write(json.dumps({"family": r["family"], "seed": r["seed"], "t": r["t"], "answer": a[0], **v}) + "\n")
        report(rows, verdicts, args.claude)
        return

    greedy = vllm_answers(rows, args.model, 1, 0.0, args.base_url)
    report(rows, list(pool.map(judge_row, [(r, g[0]) for r, g in zip(rows, greedy)])), f"{args.model} greedy")
    samples = vllm_answers(rows, args.model, args.k, 1.0, args.base_url)
    hits = []
    for r, ss in zip(rows, samples):
        vs = list(pool.map(judge_row, [(r, s) for s in ss]))
        hits.append(any(v["correct"] for v in vs))
    held = [h for r, h in zip(rows, hits) if r["tier"] == 5]
    rest = [h for r, h in zip(rows, hits) if r["tier"] < 5]
    print(f"correct@{args.k}: tiers 1-4 {sum(rest) / max(1, len(rest)):.2f}, held-out {sum(held) / max(1, len(held)):.2f}")


if __name__ == "__main__":
    main()

import argparse
import importlib
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.agents import ExploreOnly, Oracle, RandomAgent, SearchAgent  # noqa: E402
from lab.env import BudgetExceeded, Env  # noqa: E402
from lab.llm import DryRunLLM, Ledger, make_llm  # noqa: E402
from lab.stats import GROUPS, bootstrap, cost  # noqa: E402
from lab.verify import correct, probe_accuracy  # noqa: E402
from lab.worlds import FAMILIES, generate, tier_of  # noqa: E402

PAID = {"reflection_haiku", "reflection_opus", "policy_haiku", "policy_opus"}
SLOW = PAID | {"reflection", "llm_pool", "search_eig_llm", "search_eig_learned", "search_eig", "search_map",
               "search_random"}


def claude(model, args, run):
    if args.dry_run:
        return DryRunLLM(model)
    return make_llm("anthropic", model=model, effort=args.effort, cap_usd=args.cap_usd, run=run)


def make_agent(name, world, seed, args):
    b = args.budget
    if name == "oracle":
        return Oracle(world, b)
    if name == "random":
        return RandomAgent(b, seed)
    if name in ("explore", "explore_ctx"):
        return ExploreOnly(b, seed, context=name == "explore_ctx")
    if name in ("search_map", "search_random", "search_eig"):
        return SearchAgent(b, selector=name.split("_")[1], cap=args.cap, seed=seed, detail=args.trace_detail)
    if name == "search_eig_heuristic":
        from lab.scorer import HeuristicScorer
        return SearchAgent(b, cap=args.cap, seed=seed, scorer=HeuristicScorer(), scorer_weight=args.scorer_weight)
    if name == "search_eig_learned":
        from lab.scorer import LearnedScorer
        return SearchAgent(b, cap=args.cap, seed=seed, scorer=LearnedScorer(world), scorer_weight=args.scorer_weight)
    if name == "search_eig_llm":
        from lab.scorer import LLMRanker
        llm = make_llm("vllm", model=args.model, seed=seed, extra_body={"chat_template_kwargs": {"enable_thinking": True}})
        return SearchAgent(b, cap=args.cap, seed=seed, scorer=LLMRanker(llm, world))
    if name == "llm_pool":
        from lab.proposer import PoolAgent
        llm = make_llm("vllm", model=args.model, seed=seed, extra_body={"chat_template_kwargs": {"enable_thinking": True}})
        return PoolAgent(llm, b, seed=seed)
    if name == "reflection":
        from lab.reflection import ReflectionAgent
        llm = make_llm("vllm", model=args.model, seed=seed, extra_body={"chat_template_kwargs": {"enable_thinking": True}})
        return ReflectionAgent(llm, b, seed=seed)
    if name in PAID:
        from lab.prompts import REFLECT_UNKNOWN
        from lab.reflection import ReflectionAgent
        kind, size = name.split("_")
        llm = claude(f"claude-{size}-5-5", args, run=args.out)
        return ReflectionAgent(llm, b, system=REFLECT_UNKNOWN, write_notes=kind == "reflection", seed=seed)
    if name.startswith("plugin:"):
        _, mod, fn = name.split(":")
        return getattr(importlib.import_module(mod), fn)(budget=b, seed=seed)
    raise ValueError(f"unknown agent {name}")


def episode(name, family, seed, args):
    t0 = time.time()
    world = generate(tier_of(family), seed, family)
    row = {"agent": name, "family": family, "tier": tier_of(family), "seed": seed, "optimal": world["optimal"]}
    if name in PAID and not args.dry_run and Ledger().total() >= args.cap_usd:
        return None   # soft cap: don't start new paid episodes
    env = Env(world, args.budget)
    try:
        res = make_agent(name, world, seed, args).run(env)
        res = res if isinstance(res, dict) else {}
    except BudgetExceeded:
        res = {}
    except Exception:
        row.update(solved=False, steps=env.steps, crash=traceback.format_exc(), seconds=round(time.time() - t0, 1))
        return row
    row.update(solved=env.done, steps=env.steps)
    for k in ("searches", "search_seconds", "llm_calls", "llm_errors", "input_tokens", "output_tokens", "usd"):
        if k in res:
            row[k] = res[k]
    if res.get("map_door") is not None:
        row["door_correct"] = correct(world, res["map_door"])
        row["probe_accuracy"] = probe_accuracy(world, tuple(res["map_rules"]))
    if name in PAID:
        row["dry_run"] = args.dry_run
    row["seconds"] = round(time.time() - t0, 1)
    if args.traces and "trace" in res:
        p = Path(args.out) / "traces" / f"{name}-{world['name']}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res["trace"], default=list))
    return row


def summarise(path, budget=100):
    rows = [json.loads(l) for l in open(Path(path) / "results.jsonl")]
    agents = sorted({r["agent"] for r in rows}, key=lambda a: [r["agent"] for r in rows].index(a))
    print(f"{'agent':22}" + "".join(f"{g:>22}" for g, _ in GROUPS) + "   solved (held-out)")
    for a in agents:
        line = f"{a:22}"
        for _, tiers in GROUPS:
            xs = [cost(r, budget) for r in rows if r["agent"] == a and r["tier"] in tiers]
            if xs:
                m, lo, hi = bootstrap(xs)
                line += f"{m:7.1f} [{lo:5.1f},{hi:5.1f}]"
            else:
                line += f"{'-':>22}"
        held = [r for r in rows if r["agent"] == a and r["tier"] == 5]
        line += f"   {sum(r['solved'] for r in held)}/{len(held)}" if held else ""
        print(line)
    usd = sum(r.get("usd", 0) for r in rows if not r.get("dry_run"))
    if usd:
        print(f"API spend in this run: ${usd:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", nargs="+", default=["oracle", "explore_ctx", "search_map", "search_random", "search_eig"])
    ap.add_argument("--families", nargs="+", default=[f for f, (t, _) in FAMILIES.items() if t < 5])
    ap.add_argument("--held-out", action="store_true", help="also run the tier-5 families (final evaluation only)")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--cap", type=int, default=300)
    ap.add_argument("--scorer-weight", type=float, default=1.0)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--cap-usd", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--traces", action="store_true")
    ap.add_argument("--trace-detail", type=int, default=0)
    ap.add_argument("--out", default="runs/eval/main")
    ap.add_argument("--summarise", metavar="RUN")
    args = ap.parse_args()

    if args.summarise:
        summarise(args.summarise, args.budget)
        return
    if any(a in PAID for a in args.agents) and args.cap_usd is None and not args.dry_run:
        sys.exit("paid agents need --cap-usd (or --dry-run first)")
    families = list(args.families)
    if args.held_out:
        families += [f for f, (t, _) in FAMILIES.items() if t == 5 and f not in families]
    if any(tier_of(f) == 5 for f in args.families) and not args.held_out:
        sys.exit("tier-5 families are held out: pass --held-out, and only for the final evaluation")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=1))
    res_path = out / "results.jsonl"
    done = set()
    if res_path.exists():
        rows = [json.loads(l) for l in open(res_path)]
        if any(r.get("dry_run") for r in rows) != args.dry_run and any("dry_run" in r for r in rows):
            sys.exit("don't mix dry-run and real episodes in one --out")
        keep = [r for r in rows if "crash" not in r]
        with open(out / "crashes.jsonl", "a") as f:
            for r in rows:
                if "crash" in r:
                    f.write(json.dumps(r) + "\n")
        res_path.write_text("".join(json.dumps(r) + "\n" for r in keep))
        done = {(r["agent"], r["family"], r["seed"]) for r in keep}

    jobs = [(a, f, s) for a in args.agents for f in families
            for s in range(args.seed_offset, args.seed_offset + args.seeds) if (a, f, s) not in done]
    jobs.sort(key=lambda j: (j[0] not in SLOW, -tier_of(j[1])))
    workers = args.workers or (4 if any(a in PAID for a in args.agents) else os.cpu_count())
    print(f"{len(jobs)} episodes on {workers} workers -> {res_path}")
    with ProcessPoolExecutor(workers) as ex, open(res_path, "a") as f:
        futs = {ex.submit(episode, *j, args): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            if row is None:
                continue
            f.write(json.dumps(row) + "\n")
            f.flush()
            tag = "CRASH" if "crash" in row else ("ok" if row["solved"] else "fail")
            print(f"[{i}/{len(jobs)}] {row['agent']:18} {row['family']:16} s{row['seed']} {tag:5} {row['steps']:3} steps")
    summarise(out, args.budget)


if __name__ == "__main__":
    main()

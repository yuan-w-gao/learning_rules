"""GRPO on the door proposer, starting from the SFT adapter. Needs a GPU and a vLLM server on the same box:

  VLLM_ALLOW_RUNTIME_LORA_UPDATING=True vllm serve Qwen/Qwen3-8B --enable-lora --max-lora-rank 16 --max-loras 2 \
      --gpu-memory-utilization 0.3 --max-model-len 8192 --enable-prefix-caching

Reward from the verifier: 1.0 correct, 0.2 consistent but wrong, 0 if it doesn't parse or contradicts the history.
Group-normalised advantages, plain REINFORCE-style loss (one update per batch, so no ratio/clipping), no KL.
"""
import argparse
import json
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.prompts import door_prompt  # noqa: E402
from lab.proposer import END, judge  # noqa: E402
from lab.worlds import generate  # noqa: E402


def judge_row(args):
    row, text = args
    world = generate(row["tier"], row["seed"], row["family"])
    return judge(world, row["actions"], text)


def sample(client, model, rows, n, temperature, max_tokens, seed):
    out = []
    for r in rows:
        resp = client.completions.create(model=model, prompt=door_prompt(r["world_text"], r["history"]), n=n,
                                         temperature=temperature, max_tokens=max_tokens, stop=[END], seed=seed)
        out.append([(c.text, c.finish_reason == "stop") for c in resp.choices])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--sft", required=True, help="SFT adapter dir")
    ap.add_argument("--base", default="Qwen/Qwen3-8B")
    ap.add_argument("--vllm", default="http://localhost:8000")
    ap.add_argument("--out", default="runs/grpo")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--prompts", type=int, default=32)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--val-every", type=int, default=25)
    ap.add_argument("--val-rows", type=int, default=200)
    args = ap.parse_args()

    import openai
    import requests
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(l) for l in open(args.data)]
    train = [r for r in rows if r["split"] == "train"]
    val = random.Random(1).sample([r for r in rows if r["split"] == "val"], args.val_rows)
    tok = AutoTokenizer.from_pretrained(args.base)
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="cuda")
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    model = PeftModel.from_pretrained(base, args.sft, is_trainable=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    client = openai.OpenAI(base_url=args.vllm + "/v1", api_key="none", timeout=900)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.jsonl", "a")
    pool = ProcessPoolExecutor(16)
    rng = random.Random(0)

    current = None

    def swap(step):
        nonlocal current
        path = (out / "adapter").resolve()
        model.save_pretrained(path)
        name = f"policy{step}"
        r = requests.post(args.vllm + "/v1/load_lora_adapter", json={"lora_name": name, "lora_path": str(path)})
        r.raise_for_status()
        if current:
            requests.post(args.vllm + "/v1/unload_lora_adapter", json={"lora_name": current})
        current = name

    def logp_of(prompt, completion):
        p = tok.encode(prompt, add_special_tokens=False)
        c = tok.encode(completion, add_special_tokens=False)
        p = p[-(args.max_len - len(c)):]
        ids = torch.tensor([p + c], device="cuda")
        logits = model(input_ids=ids).logits[0, len(p) - 1:-1].float()
        return torch.log_softmax(logits, -1).gather(1, ids[0, len(p):, None]).sum()

    swap(0)
    for step in range(1, args.steps + 1):
        t0 = time.time()
        batch = rng.sample(train, args.prompts)
        samples = sample(client, current, batch, args.group, 1.0, args.max_tokens, seed=step)
        sec_gen = time.time() - t0
        verdicts = list(pool.map(judge_row, [(r, txt) for r, group in zip(batch, samples) for txt, _ in group]))
        examples, rewards = [], []
        for gi, (r, group) in enumerate(zip(batch, samples)):
            rs = [v["reward"] for v in verdicts[gi * args.group:(gi + 1) * args.group]]
            rewards += rs
            if statistics.pstdev(rs) < 1e-6:
                continue        # everyone got the same reward: nothing to learn from this group
            t = torch.tensor(rs)
            adv = ((t - t.mean()) / (t.std() + 1e-4)).tolist()
            prompt = door_prompt(r["world_text"], r["history"])
            for (txt, finished), a in zip(group, adv):
                examples.append((prompt, txt + (END + tok.eos_token if finished else ""), a))
        loss_total = 0.0
        if examples:
            model.train()
            for i in range(0, len(examples), args.micro):
                chunk = examples[i:i + args.micro]
                loss = sum(-a * logp_of(p, c) / args.max_tokens for p, c, a in chunk) / len(examples)
                loss.backward()
                loss_total += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            swap(step)
        rec = {"step": step, "reward": sum(rewards) / len(rewards),
               "correct": sum(v["correct"] for v in verdicts) / len(verdicts),
               "consistent": sum(v["consistent"] for v in verdicts) / len(verdicts),
               "parsed": sum(v["parsed"] for v in verdicts) / len(verdicts),
               "examples": len(examples), "loss": loss_total, "sec_gen": round(sec_gen), "sec": round(time.time() - t0)}
        if step % args.val_every == 0:
            greedy = sample(client, current, val, 1, 0.0, args.max_tokens, seed=0)
            vv = list(pool.map(judge_row, [(r, g[0][0]) for r, g in zip(val, greedy)]))
            for k in ("parsed", "consistent", "correct"):
                rec[f"val_{k}"] = sum(v[k] for v in vv) / len(vv)
            model.save_pretrained(out / f"adapter_step{step}")
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(rec)


if __name__ == "__main__":
    main()

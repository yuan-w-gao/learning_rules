"""LoRA-tune a base model to answer " yes"/" no" to "is this candidate the true door mechanism?".

Needs a GPU (we used one A100 for Qwen3-8B). Data: scripts/make_data.py scorer.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.prompts import SCORER_HEADER  # noqa: E402


def examples(rows, rng, per=2):
    out = []
    for r in rows:
        pos = [i for i, y in enumerate(r["labels"]) if y]
        neg = [i for i, y in enumerate(r["labels"]) if not y]
        if not pos or not neg:
            continue
        hard = sorted(neg, key=lambda i: r["dls"][i])[:per]          # short wrong candidates are the confusing ones
        rest = [i for i in neg if i not in hard]
        picked = rng.sample(pos, min(per, len(pos))) + hard + rng.sample(rest, min(per, len(rest)))
        prefix = f"{SCORER_HEADER}\n\n# World\n{r['world_text']}\n\n# History\n{r['history']}\n\n# Candidate\n"
        for i in picked:
            out.append((prefix + r["cands"][i] + "\n\nIs this the true mechanism? Answer:", int(not r["labels"][i])))
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--base", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", default="runs/scorer")
    ap.add_argument("--max-examples", type=int, default=16000)
    ap.add_argument("--val-examples", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rng = random.Random(args.seed)
    rows = [json.loads(l) for l in open(args.data)]
    train = examples([r for r in rows if r["split"] == "train"], rng)[: args.max_examples]
    val = examples([r for r in rows if r["split"] == "val"], rng)[: args.val_examples]
    print(f"{len(train)} train / {len(val)} val examples")

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    answer_ids = [tok.encode(" yes", add_special_tokens=False)[0], tok.encode(" no", add_special_tokens=False)[0]]

    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
                                                 device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                             "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    def batch_loss(batch):
        enc = tok([p for p, _ in batch], return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_len).to("cuda")
        logits = model(**enc).logits
        last = enc["attention_mask"].sum(1) - 1
        two = logits[torch.arange(len(batch)), last][:, answer_ids].float()
        y = torch.tensor([t for _, t in batch], device="cuda")
        return torch.nn.functional.cross_entropy(two, y), (two.argmax(1) == y).float().mean().item()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    total = len(train) // (args.bs * args.accum)

    def lr_at(s):
        if s < args.warmup:
            return (s + 1) / args.warmup
        return 0.5 * (1 + math.cos(math.pi * (s - args.warmup) / max(1, total - args.warmup)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_config.json").write_text(json.dumps(vars(args), indent=1))
    log = open(out / "log.jsonl", "a")

    def evaluate():
        model.eval()
        accs = []
        with torch.no_grad():
            for i in range(0, len(val), args.bs):
                accs.append(batch_loss(val[i:i + args.bs])[1])
        model.train()
        return sum(accs) / max(1, len(accs))

    print("val acc before:", evaluate())
    model.train()
    t0 = time.time()
    for s in range(total):
        for k in range(args.accum):
            i = (s * args.accum + k) * args.bs
            loss, acc = batch_loss(train[i:i + args.bs])
            (loss / args.accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad()
        rec = {"step": s + 1, "loss": round(loss.item(), 4), "acc": round(acc, 3), "lr": sched.get_last_lr()[0],
               "sec": round(time.time() - t0)}
        if (s + 1) % args.eval_every == 0 or s + 1 == total:
            rec["val_acc"] = round(evaluate(), 4)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(rec)
    model.save_pretrained(out / "adapter")


if __name__ == "__main__":
    main()

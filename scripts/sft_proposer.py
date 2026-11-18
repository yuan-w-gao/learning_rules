"""SFT a base model to write the door rules (text, then END) from world + history. Needs a GPU.
Data: scripts/make_data.py proposer.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.prompts import door_prompt  # noqa: E402
from lab.proposer import END  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--base", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", default="runs/sft")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=25)
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(l) for l in open(args.data)]
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"][:400]
    tok = AutoTokenizer.from_pretrained(args.base)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def encode(r):
        prompt = tok.encode(door_prompt(r["world_text"], r["history"]), add_special_tokens=False)
        comp = tok.encode(r["target"] + "\n" + END, add_special_tokens=False) + [tok.eos_token_id]
        prompt = prompt[-(args.max_len - len(comp)):]      # cut the start of long histories, keep the answer
        return prompt + comp, [-100] * len(prompt) + comp

    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
                                                 device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                             "up_proj", "down_proj"]))

    def loss_of(batch):
        enc = [encode(r) for r in batch]
        n = max(len(x) for x, _ in enc)
        ids = torch.tensor([x + [pad] * (n - len(x)) for x, _ in enc], device="cuda")
        labels = torch.tensor([y + [-100] * (n - len(y)) for _, y in enc], device="cuda")
        mask = torch.tensor([[1] * len(x) + [0] * (n - len(x)) for x, _ in enc], device="cuda")
        return model(input_ids=ids, attention_mask=mask, labels=labels).loss

    rng = random.Random(0)
    steps_per_epoch = len(train) // (args.bs * args.accum)
    total = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / args.warmup if s < args.warmup else
                                              0.5 * (1 + math.cos(math.pi * (s - args.warmup) / max(1, total - args.warmup))))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_config.json").write_text(json.dumps(vars(args), indent=1))
    log = open(out / "log.jsonl", "a")
    t0, s = time.time(), 0
    model.train()
    for epoch in range(args.epochs):
        rng.shuffle(train)
        for b in range(steps_per_epoch):
            for k in range(args.accum):
                i = (b * args.accum + k) * args.bs
                loss = loss_of(train[i:i + args.bs])
                (loss / args.accum).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            s += 1
            rec = {"step": s, "epoch": epoch, "loss": round(loss.item(), 4), "sec": round(time.time() - t0)}
            if s % args.eval_every == 0 or s == total:
                model.eval()
                with torch.no_grad():
                    rec["val_loss"] = round(sum(loss_of(val[j:j + args.bs]).item()
                                                for j in range(0, len(val), args.bs)) / max(1, len(val) // args.bs), 4)
                model.train()
                model.save_pretrained(out / f"adapter_step{s}")
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(rec)
    model.save_pretrained(out / "adapter")


if __name__ == "__main__":
    main()

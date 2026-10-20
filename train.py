#!/usr/bin/env python3
"""train.py -- QAT-LoRA on the shortest correct sampled answer per training problem.

  train.py <samples.jsonl> <out_dir> [--epochs E] [--lr LR] [--rank R] [--alpha A] [--batch B] [--max-len N]
           [--prompt-budget]

Selection is code: per problem, among samples whose answer is right and whose thinking closed, the
one with the fewest generated tokens. out_dir/selection.jsonl records, per problem, how many
candidates, how many right, and the chosen length. Loss is on the generated tokens only (thinking,
answer and the closing <turn|>). out_dir/lora.pt holds the adapter; export_gguf.py writes it into a
GGUF.

--prompt-budget: the prompt states a thinking budget (lazy_eval.PROMPT_BUDGET), N = the smallest of
BUCKETS that covers the sample's own thinking tokens. Per problem it trains on the shortest right
sample and, when there is one, the longest right sample in a different bucket, so the model sees
the same problem at two budgets. Prompt ids come from the HF chat template, checked against the
sampler's ids for the plain prompt.
"""

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict

import torch

import lazy_eval
import qat_lora

TURN_END = 106  # <turn|>, checked against the tokenizer in main()
END_THINK = 101  # <channel|>
BUCKETS = (128, 256, 512, 1024, 2048)


def think_len(r):
    return r["gen_ids"].index(END_THINK) + 1 if END_THINK in r["gen_ids"] else None


def bucket(n):
    return next((b for b in BUCKETS if n <= b), None)


def chat_ids(tok, text):
    ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                  enable_thinking=True, tokenize=True)
    return list(ids["input_ids"]) if hasattr(ids, "keys") else list(ids)


def select(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["id"]].append(r)
    chosen, log = [], []
    for pid, rs in sorted(by.items()):
        ok = [r for r in rs if r["right"] and "error" not in r and r.get("finish") != "length"]
        best = min(ok, key=lambda r: (r["n_gen"], r["sample"])) if ok else None
        log.append({"id": pid, "candidates": len(rs), "right": len(ok),
                    "chosen_len": best["n_gen"] if best else None,
                    "lens_right": sorted(r["n_gen"] for r in ok)})
        if best:
            chosen.append(best)
    return chosen, log


def select_budget(rows):
    """-> [(row, budget)]: shortest right per problem, plus the longest right in another bucket."""
    by = defaultdict(list)
    for r in rows:
        if r["right"] and "error" not in r and r.get("finish") != "length" and think_len(r):
            by[r["id"]].append(r)
    out = []
    for _, rs in sorted(by.items()):
        rs.sort(key=lambda r: (r["n_gen"], r["sample"]))
        b0 = bucket(think_len(rs[0]))
        if b0 is None:
            continue
        out.append((rs[0], b0))
        other = [r for r in rs[1:] if bucket(think_len(r)) not in (None, b0)]
        if other:
            out.append((other[-1], bucket(think_len(other[-1]))))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("samples")
    p.add_argument("out")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=32.0)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prompt-budget", action="store_true", help="state each sample's budget in the prompt")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(qat_lora.HF_DIR)
    assert tok.convert_tokens_to_ids("<turn|>") == TURN_END, tok.convert_tokens_to_ids("<turn|>")
    assert tok.convert_tokens_to_ids("<channel|>") == END_THINK, tok.convert_tokens_to_ids("<channel|>")

    rows = lazy_eval.load(a.samples)
    chosen, log = select(rows)
    with open(os.path.join(a.out, "selection.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in log)
    if a.prompt_budget:
        question = {r["id"]: r["question"] for r in lazy_eval.load("data/train.jsonl")}
        first = chosen[0]
        assert chat_ids(tok, lazy_eval.prompt(question[first["id"]])) == first["prompt_ids"], "template != sampler ids"
        pairs = select_budget(rows)
        with open(os.path.join(a.out, "selection-budget.jsonl"), "w") as f:
            f.writelines(json.dumps({"id": r["id"], "sample": r["sample"], "think": think_len(r), "budget": b}) + "\n"
                         for r, b in pairs)
        items = [(chat_ids(tok, lazy_eval.prompt(question[r["id"]], b)), r) for r, b in pairs]
    else:
        items = [(r["prompt_ids"], r) for r in chosen]
    seqs = []
    for prompt_ids, r in items:
        gen = r["gen_ids"] if r["gen_ids"][-1:] == [TURN_END] else [*r["gen_ids"], TURN_END]
        if len(prompt_ids) + len(gen) <= a.max_len:
            seqs.append((prompt_ids, gen))
    print(json.dumps({"problems": len(log), "with_right": len(chosen), "train_seqs": len(seqs),
                      "mean_gen_len": round(sum(len(g) for _, g in seqs) / max(len(seqs), 1), 1)}))

    torch.manual_seed(a.seed)
    model, cfg, lm_head, missing = qat_lora.load(rank=a.rank, alpha=a.alpha)
    assert not missing, missing
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps = a.epochs * math.ceil(len(seqs) / a.batch)
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))

    rng = random.Random(a.seed)
    logf = open(os.path.join(a.out, "train.log.jsonl"), "w")  # noqa: SIM115 -- open for the whole run
    step = 0
    t0 = time.time()
    for epoch in range(a.epochs):
        order = list(range(len(seqs)))
        rng.shuffle(order)
        for b in range(0, len(order), a.batch):
            batch = [seqs[i] for i in order[b : b + a.batch]]
            n_tok = sum(len(g) for _, g in batch)
            total = 0.0
            for prompt, gen in batch:  # one sequence at a time, loss weighted by its answer tokens
                ids = torch.tensor([prompt + gen], device="cuda")
                labels = ids.clone()
                labels[:, : len(prompt)] = -100
                loss = qat_lora.masked_loss(model, cfg, lm_head, ids, labels) * (len(gen) / n_tok)
                loss.backward()
                total += loss.item()
            gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            rec = {"step": step, "of": steps, "epoch": epoch, "loss": round(total, 4), "gnorm": round(gnorm, 3),
                   "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0),
                   "peak_mib": round(torch.cuda.max_memory_allocated() / 2**20)}
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            if step % 10 == 0 or step == steps:
                print(json.dumps(rec), flush=True)
    logf.close()
    torch.save({"lora": qat_lora.lora_state(model), "rank": a.rank, "alpha": a.alpha, "args": vars(a)},
               os.path.join(a.out, "lora.pt"))


if __name__ == "__main__":
    main()

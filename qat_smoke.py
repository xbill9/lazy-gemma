#!/usr/bin/env python3
"""qat_smoke.py -- SPEC checks 2 and 3 for the QAT-LoRA arm.

  qat_smoke.py run/ref-greedy.json [--rank R] [--seq N] [--batch B] [--steps S]

1. Load: peak VRAM, and any tensor the loader failed to fill.
2. Same model as served: teacher-force llama-server's greedy tokens (run/ref-greedy.json) and count
   how often this model's top-1 token is the one llama.cpp chose.
3. Train: S optimizer steps of B sequences of N tokens (gradient checkpointing, loss on the second
   half), peak VRAM and seconds per step.
"""

import argparse
import json
import time

import torch

import qat_lora


def mib(x):
    return round(x / 2**20)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ref")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--steps", type=int, default=2)
    a = p.parse_args()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model, cfg, lm_head, missing = qat_lora.load(rank=a.rank)
    torch.cuda.synchronize()
    n_lora = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
    print(json.dumps({"load_s": round(time.time() - t0, 1), "load_peak_mib": mib(torch.cuda.max_memory_allocated()),
                      "unfilled": missing, "lora_params": n_lora}))

    model.eval()
    with open(a.ref) as f:
        refs = json.load(f)
    agree = total = 0
    with torch.no_grad():
        for r in refs:
            ids = torch.tensor([r["prompt_ids"] + r["gen_ids"]], device="cuda")
            h = model(input_ids=ids, use_cache=False).last_hidden_state
            top = qat_lora.lm_logits(cfg, lm_head, h[0]).argmax(-1).tolist()
            n0 = len(r["prompt_ids"])
            pred = top[n0 - 1 : n0 - 1 + len(r["gen_ids"])]
            agree += sum(x == y for x, y in zip(pred, r["gen_ids"], strict=True))
            total += len(r["gen_ids"])
    print(json.dumps({"top1_agree": agree, "of": total, "rate": round(agree / total, 4)}))

    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    torch.cuda.reset_peak_memory_stats()
    g = torch.Generator().manual_seed(0)
    for step in range(a.steps):
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(a.batch):  # one sequence at a time, gradients accumulated over the batch
            ids = torch.randint(1000, 200000, (1, a.seq), generator=g).cuda()
            labels = ids.clone()
            labels[:, : a.seq // 2] = -100
            loss = qat_lora.masked_loss(model, cfg, lm_head, ids, labels) / a.batch
            loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        print(json.dumps({"step": step, "loss": round(loss.item() * a.batch, 4), "s_per_step": round(time.time() - t0, 2),
                          "peak_mib": mib(torch.cuda.max_memory_allocated()),
                          "reserved_mib": mib(torch.cuda.max_memory_reserved())}))


if __name__ == "__main__":
    main()

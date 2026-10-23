#!/usr/bin/env python3
"""Thinking and answer length per model on all 1,319 GSM8K test problems, in Gemma 4 tokens.

Counts the `reasoning` and `text` fields of each eval row with the model's tokenizer. Prints median and mean
thinking tokens, median answer tokens, and the share of replies whose thinking is over 64, 128, 256 tokens."""
import json
import os
import statistics
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)
import qat_lora

R = os.path.join(ROOT, "benchmarks", "runs")
REST = "2026-10-22-rest-e2b-3050oem"
MODELS = [
    ("base uncapped", f"{REST}/base-think.jsonl", "2026-10-20-baseline-e2b-3050oem/think-greedy.jsonl"),
    ("base + cap 128", f"{REST}/base-budget-128.jsonl", "2026-10-20-budget-e2b-3050oem/budget-128.jsonl"),
    ("base + cap 64", f"{REST}/base-budget-64.jsonl", "2026-10-23-qatlora-capped64-e2b-3050oem/base-budget-64.jsonl"),
    ("tuned 512", f"{REST}/capped512-think.jsonl", "2026-10-22-qatlora-capped512-e2b-3050oem/think-greedy.jsonl"),
    ("tuned 256", f"{REST}/capped256-think.jsonl", "2026-10-21-qatlora-capped-e2b-3050oem/think-greedy.jsonl"),
    ("tuned 128", f"{REST}/capped128-think.jsonl", "2026-10-22-qatlora-capped128-e2b-3050oem/think-greedy.jsonl"),
    ("tuned 64", f"{REST}/capped64-think.jsonl", "2026-10-23-qatlora-capped64-e2b-3050oem/think-greedy.jsonl"),
]


def main():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(qat_lora.HF_DIR)
    n = lambda s: len(tok(s or "", add_special_tokens=False)["input_ids"])
    for name, *paths in MODELS:
        rows = []
        for p in paths:
            with open(os.path.join(R, p)) as f:
                rows += [json.loads(line) for line in f]
        assert len(rows) == 1319, (name, len(rows))
        think = [n(r["reasoning"]) for r in rows]
        ans = [n(r["text"]) for r in rows]
        over = {c: sum(t > c for t in think) for c in (64, 128, 256)}
        print(f"{name}: thinking median {statistics.median(think)} mean {statistics.mean(think):.1f}; "
              f"answer median {statistics.median(ans)}; thinking over 64/128/256: "
              f"{over[64]}/{over[128]}/{over[256]} of 1319")


if __name__ == "__main__":
    main()

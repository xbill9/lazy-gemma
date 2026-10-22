#!/usr/bin/env python3
"""Paired comparison, 256-capped-samples model (uncapped) vs the base GGUF with the 256 cap: on the 1,069
rest problems, and on all 1,319 test problems (rest plus the 250 held-out set). Problems only one side got
right, exact two-sided McNemar p, tokens per correct answer, mean tokens on problems both got right."""
import json
import math
import os

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def load(p):
    return {r["id"]: r for r in map(json.loads, open(os.path.join(RUNS, p)))}


base = load("2026-10-22-rest-e2b-3050oem/base-budget-256.jsonl")
new = load("2026-10-22-rest-e2b-3050oem/capped256-think.jsonl")
sets = {"rest 1,069": (base, new),
        "all 1,319": ({**base, **load("2026-10-20-budget-e2b-3050oem/budget-256.jsonl")},
                      {**new, **load("2026-10-21-qatlora-capped-e2b-3050oem/think-greedy.jsonl")})}
for name, (old, cur) in sets.items():
    assert old.keys() == cur.keys()
    ids = list(old)
    b = sum(cur[i]["right"] and not old[i]["right"] for i in ids)
    c = sum(old[i]["right"] and not cur[i]["right"] for i in ids)
    n = b + c
    pv = min(1.0, 2 * sum(math.comb(n, j) for j in range(min(b, c) + 1)) / 2**n) if n else 1.0
    tpc = [sum(d[i]["completion_tokens"] for i in ids) / sum(d[i]["right"] for i in ids) for d in (old, cur)]
    both = [i for i in ids if old[i]["right"] and cur[i]["right"]]
    mo = sum(old[i]["completion_tokens"] for i in both) / len(both)
    mn = sum(cur[i]["completion_tokens"] for i in both) / len(both)
    print(f"{name} ({len(ids)}): right base+cap {sum(old[i]['right'] for i in ids)}, capped-data "
          f"{sum(cur[i]['right'] for i in ids)}; only capped-data right {b}, only base+cap right {c}, "
          f"McNemar p={pv:.3f}; tokens per correct {tpc[0]:.1f} vs {tpc[1]:.1f}; "
          f"{len(both)} both right: {mo:.1f} vs {mn:.1f} mean tokens")

#!/usr/bin/env python3
"""Paired comparison of the capped-data model (uncapped eval) against three earlier evals on the same 250 problems:
problems only one side got right, exact two-sided McNemar p, and mean output tokens on problems both got right."""
import json
import math
import os

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def load(p):
    return {r["id"]: r for r in map(json.loads, open(os.path.join(RUNS, p)))}


new = load("2026-10-21-qatlora-capped-e2b-3050oem/think-greedy.jsonl")
for name, p in [("base + cap 256", "2026-10-20-budget-e2b-3050oem/budget-256.jsonl"),
                ("base uncapped", "2026-10-20-baseline-e2b-3050oem/think-greedy.jsonl"),
                ("tuned lr 1e-3 + cap 256", "2026-10-21-qatlora-lr1e-3-budget256-e2b-3050oem/budget-256.jsonl")]:
    old = load(p)
    assert old.keys() == new.keys()
    b = sum(new[i]["right"] and not old[i]["right"] for i in old)
    c = sum(old[i]["right"] and not new[i]["right"] for i in old)
    n = b + c
    pv = min(1.0, 2 * sum(math.comb(n, j) for j in range(min(b, c) + 1)) / 2**n) if n else 1.0
    both = [i for i in old if old[i]["right"] and new[i]["right"]]
    to = sum(old[i]["completion_tokens"] for i in both) / len(both)
    tn = sum(new[i]["completion_tokens"] for i in both) / len(both)
    print(f"vs {name}: only capped-data right {b}, only {name} right {c}, McNemar p={pv:.3f}; "
          f"{len(both)} both right: {to:.1f} vs {tn:.1f} mean tokens")

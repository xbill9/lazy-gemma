#!/usr/bin/env python3
"""Paired comparison, each capped-samples model (uncapped) vs the base GGUF with the same cap: on the 1,069
rest problems, and on all 1,319 test problems (rest plus the 250 held-out set). Caps with no finished eval
are skipped. Problems only one side got
right, exact two-sided McNemar p, tokens per correct answer, mean tokens on problems both got right."""
import json
import math
import os

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def load(p):
    return {r["id"]: r for r in map(json.loads, open(os.path.join(RUNS, p)))}


TRAINED_250 = {256: "2026-10-21-qatlora-capped-e2b-3050oem", 128: "2026-10-22-qatlora-capped128-e2b-3050oem",
               512: "2026-10-22-qatlora-capped512-e2b-3050oem", 64: "2026-10-23-qatlora-capped64-e2b-3050oem"}
BASE_250 = {64: "2026-10-23-qatlora-capped64-e2b-3050oem/base-budget-64.jsonl"}  # others: 2026-10-20-budget
sets = {}
for cap, run in TRAINED_250.items():
    paths = [f"2026-10-22-rest-e2b-3050oem/base-budget-{cap}.jsonl", f"2026-10-22-rest-e2b-3050oem/capped{cap}-think.jsonl",
             BASE_250.get(cap, f"2026-10-20-budget-e2b-3050oem/budget-{cap}.jsonl"), f"{run}/think-greedy.jsonl"]
    if not all(os.path.exists(os.path.join(RUNS, p)) for p in paths):
        continue
    base, new, base250, new250 = map(load, paths)
    if len(new) < 1069:
        continue
    sets[f"cap {cap}, rest 1,069"] = (base, new)
    sets[f"cap {cap}, all 1,319"] = ({**base, **base250}, {**new, **new250})
UNCAPPED = ["2026-10-22-rest-e2b-3050oem/base-think.jsonl", "2026-10-20-baseline-e2b-3050oem/think-greedy.jsonl"]
if all(os.path.exists(os.path.join(RUNS, p)) for p in UNCAPPED):
    uncapped = {**load(UNCAPPED[0]), **load(UNCAPPED[1])}
    for name, (_, cur) in list(sets.items()):
        if name.endswith("all 1,319"):
            sets[name.replace("all 1,319", "vs base uncapped, all 1,319")] = (uncapped, cur)
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
    print(f"{name} ({len(ids)}): right base {sum(old[i]['right'] for i in ids)}, capped-data "
          f"{sum(cur[i]['right'] for i in ids)}; only capped-data right {b}, only base right {c}, "
          f"McNemar p={pv:.3f}; tokens per correct {tpc[0]:.1f} vs {tpc[1]:.1f}; "
          f"{len(both)} both right: {mo:.1f} vs {mn:.1f} mean tokens")

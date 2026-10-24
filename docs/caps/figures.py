#!/usr/bin/env python3
"""Every derived figure in devto-lazy-gemma-caps.md, computed from the run files. Output: evidence/figures.txt."""
import json
import math
import os
from datetime import datetime

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
R = os.path.join(ROOT, "benchmarks", "runs")
REST = "2026-10-22-rest-e2b-3050oem"
LEVELS = 1835532288


def rows(*paths):
    return {r["id"]: r for p in paths for r in map(json.loads, open(os.path.join(R, p)))}


def tpc(d):
    return sum(r["completion_tokens"] for r in d.values()) / sum(r["right"] for r in d.values())


def right(d):
    return sum(r["right"] for r in d.values())


def mcnemar(a, b):
    x = sum(b[i]["right"] and not a[i]["right"] for i in a)
    y = sum(a[i]["right"] and not b[i]["right"] for i in a)
    n = x + y
    return min(1.0, 2 * sum(math.comb(n, j) for j in range(min(x, y) + 1)) / 2**n)


def last(path):
    with open(os.path.join(R, path)) as f:
        return json.loads(f.readlines()[-1])


def text(path):
    with open(os.path.join(R, path)) as f:
        return f.read()


def stamp(path, needle):
    for line in text(path).splitlines():
        if needle in line:
            return datetime.strptime(line[:19] + " +0000", "%Y-%m-%d %H:%M:%S %z")
    raise KeyError(needle)


base = {"none": rows(f"{REST}/base-think.jsonl", "2026-10-20-baseline-e2b-3050oem/think-greedy.jsonl")}
for c in (512, 256, 128):
    base[c] = rows(f"{REST}/base-budget-{c}.jsonl", f"2026-10-20-budget-e2b-3050oem/budget-{c}.jsonl")
base[64] = rows(f"{REST}/base-budget-64.jsonl", "2026-10-23-qatlora-capped64-e2b-3050oem/base-budget-64.jsonl")
RUN = {512: "2026-10-22-qatlora-capped512-e2b-3050oem", 256: "2026-10-21-qatlora-capped-e2b-3050oem",
       128: "2026-10-22-qatlora-capped128-e2b-3050oem", 64: "2026-10-23-qatlora-capped64-e2b-3050oem"}
NAME = {512: "capped512", 256: "capped256", 128: "capped128", 64: "capped64"}
tuned = {c: rows(f"{REST}/{NAME[c]}-think.jsonl", f"{RUN[c]}/think-greedy.jsonl") for c in RUN}

for c, d in base.items():
    print(f"base cap {c}: right {right(d)} of {len(d)}, tokens per correct {tpc(d):.1f}")
for c, d in tuned.items():
    u, b = base["none"], base[c]
    print(f"tuned {c}: right {right(d)}, tokens per correct {tpc(d):.1f}; vs same cap: {right(d) - right(b):+d} right, "
          f"{100 * (1 - tpc(d) / tpc(b)):.1f}% fewer tokens, p={mcnemar(b, d):.2g}; vs uncapped: {right(d) - right(u):+d} right, "
          f"{100 * (1 - tpc(d) / tpc(u)):.1f}% fewer tokens, p={mcnemar(u, d):.3f}")
t64, b128 = tuned[64], base[128]
print(f"tuned 64 vs base cap 128: {right(t64) - right(b128):+d} right, {100 * (1 - tpc(t64) / tpc(b128)):.1f}% fewer tokens, "
      f"p={mcnemar(b128, t64):.3f}")
print(f"base cap 128 vs uncapped: {100 * (1 - tpc(b128) / tpc(base['none'])):.1f}% fewer tokens, "
      f"{right(b128) - right(base['none']):+d} right")
print(f"tuned 256 accuracy: {100 * right(tuned[256]) / len(tuned[256]):.1f}%")
for c, run in RUN.items():
    t = last(f"{run}/train.log.jsonl")
    lv = int(text(f"{run}/export.out").split("levels changed ")[1].split()[0])
    print(f"train {c}: {t['elapsed_s']} s = {t['elapsed_s'] // 3600} h {t['elapsed_s'] % 3600 // 60} min, "
          f"peak {t['peak_mib']} MiB; levels moved {100 * lv / LEVELS:.1f}%")
s = stamp(f"{REST}/driver2.log", "samples 64") - stamp(f"{REST}/driver2.log", "base-think")
print(f"sampling 64: {s.seconds // 3600} h {s.seconds % 3600 // 60} min")
print(f"64-cap targets vs uncapped targets: {376.1 / 723.8:.2f} of the length")

SEED1 = {64: "2026-10-23-qatlora-capped64-seed1-e2b-3050oem", 128: "2026-10-23-qatlora-capped128-seed1-e2b-3050oem"}
for c, run in SEED1.items():
    if not os.path.exists(os.path.join(R, run, "rest-think.summary.json")) or not text(f"{run}/rest-think.summary.json"):
        continue
    d = rows(f"{run}/rest-think.jsonl", f"{run}/think-greedy.jsonl")
    u, b, s0 = base["none"], base[c], tuned[c]
    print(f"tuned {c} seed 1: right {right(d)}, tokens per correct {tpc(d):.1f}; vs seed 0: {right(d) - right(s0):+d} right, "
          f"p={mcnemar(s0, d):.3f}; vs same cap: {right(d) - right(b):+d} right, {100 * (1 - tpc(d) / tpc(b)):.1f}% fewer "
          f"tokens, p={mcnemar(b, d):.2g}; vs uncapped: {right(d) - right(u):+d} right, "
          f"{100 * (1 - tpc(d) / tpc(u)):.1f}% fewer tokens, p={mcnemar(u, d):.3f}")
    print(f"tuned {c} seed 1 vs base cap 128: {right(d) - right(b128):+d} right, {100 * (1 - tpc(d) / tpc(b128)):.1f}% fewer "
          f"tokens, p={mcnemar(b128, d):.3f}")

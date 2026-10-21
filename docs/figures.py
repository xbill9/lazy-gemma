#!/usr/bin/env python3
"""figures.py -- every derived figure in docs/devto-lazy-gemma.md, computed from the run files.

  python3 docs/figures.py > docs/evidence/figures.txt
"""

import json
import os
import re
import statistics

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks", "runs")


def js(path):
    with open(os.path.join(R, path)) as f:
        return json.load(f)


def jl(path):
    with open(os.path.join(R, path)) as f:
        return [json.loads(line) for line in f]


def pct_fewer(new, old):
    return f"{100 * (1 - new / old):.1f}%"


def main():
    base = js("2026-10-20-baseline-e2b-3050oem/think-greedy.summary.json")
    cap = js("2026-10-20-budget-e2b-3050oem/budget-256.summary.json")
    a = js("2026-10-20-qatlora-e2b-3050oem/think-greedy.summary.json")
    b = js("2026-10-20-qatlora-budgetprompt-e2b-3050oem/plain.summary.json")
    a3 = js("2026-10-21-qatlora-lr1e-3-e2b-3050oem/think-greedy.summary.json")
    print("tokens per correct, fewer than uncapped:")
    print(f"  cap 256 {pct_fewer(cap['tokens_per_correct'], base['tokens_per_correct'])}")
    print(f"  run A   {pct_fewer(a['tokens_per_correct'], base['tokens_per_correct'])}")
    print(f"  run B   {pct_fewer(b['tokens_per_correct'], base['tokens_per_correct'])}")
    print(f"  run A at lr 1e-3 {pct_fewer(a3['tokens_per_correct'], base['tokens_per_correct'])}")
    saved = base["tokens_per_correct"] - cap["tokens_per_correct"]
    print(f"cap 256 saves {saved:.1f} tokens per correct answer; at 5.00 tok/s that is {saved / 5.00:.1f} s")

    for name in ("2026-10-20-qatlora-e2b-3050oem", "2026-10-20-qatlora-budgetprompt-e2b-3050oem",
                 "2026-10-21-qatlora-lr1e-3-e2b-3050oem"):
        log = jl(f"{name}/train.log.jsonl")
        s = log[-1]["elapsed_s"]
        print(f"{name}: {log[-1]['of']} steps, {s} s = {s // 3600} h {s % 3600 // 60} min, "
              f"{s / log[-1]['of']:.1f} s per step, peak {max(r['peak_mib'] for r in log)} MiB")
        with open(os.path.join(R, name, "export.out")) as f:
            n, tot = map(int, re.search(r"levels changed (\d+) of (\d+)", f.read()).groups())
        print(f"  4-bit weights changed: {n:,} of {tot:,} = {100 * n / tot:.3f}%")

    sel = jl("2026-10-20-qatlora-e2b-3050oem/selection.jsonl")
    chosen = [r["chosen_len"] for r in sel if r["chosen_len"]]
    every = [n for r in sel for n in r["lens_right"]]
    print(f"shortest right answer mean {statistics.mean(chosen):.1f}, all right answers mean "
          f"{statistics.mean(every):.1f}: {pct_fewer(statistics.mean(chosen), statistics.mean(every))} shorter")
    print(f"lr 1e-3 export moved {134965355 / 238829:.0f}x run A's export")
    print(f"overfit moved 24,385,598 levels in 12 steps = {24385598 / 238829:.0f}x run A's export")


if __name__ == "__main__":
    main()

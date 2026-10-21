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
    ql = js("2026-10-21-qlora-nf4-e2b-3050oem/think-greedy.summary.json")
    print("tokens per correct, fewer than uncapped:")
    print(f"  cap 256 {pct_fewer(cap['tokens_per_correct'], base['tokens_per_correct'])}")
    print(f"  run A   {pct_fewer(a['tokens_per_correct'], base['tokens_per_correct'])}")
    print(f"  run B   {pct_fewer(b['tokens_per_correct'], base['tokens_per_correct'])}")
    print(f"  run A at lr 1e-3 {pct_fewer(a3['tokens_per_correct'], base['tokens_per_correct'])}")
    print(f"  QLoRA at lr 1e-3 {pct_fewer(ql['tokens_per_correct'], base['tokens_per_correct'])}")
    print(f"exact grid vs QLoRA at lr 1e-3: {pct_fewer(a3['tokens_per_correct'], ql['tokens_per_correct'])} fewer "
          f"tokens per correct answer ({a3['tokens_per_correct']} vs {ql['tokens_per_correct']})")
    saved = base["tokens_per_correct"] - cap["tokens_per_correct"]
    print(f"cap 256 saves {saved:.1f} tokens per correct answer; at 5.00 tok/s that is {saved / 5.00:.1f} s")

    for name in ("2026-10-20-qatlora-e2b-3050oem", "2026-10-20-qatlora-budgetprompt-e2b-3050oem",
                 "2026-10-21-qatlora-lr1e-3-e2b-3050oem", "2026-10-21-qlora-nf4-e2b-3050oem"):
        log = jl(f"{name}/train.log.jsonl")
        s = log[-1]["elapsed_s"]
        print(f"{name}: {log[-1]['of']} steps, {s} s = {s // 3600} h {s % 3600 // 60} min, "
              f"{s / log[-1]['of']:.1f} s per step, peak {max(r['peak_mib'] for r in log)} MiB")
        with open(os.path.join(R, name, "export.out")) as f:
            n, tot = map(int, re.search(r"levels changed (\d+) of (\d+)", f.read()).groups())
        print(f"  4-bit weights changed: {n:,} of {tot:,} = {100 * n / tot:.3f}%")

    with open(os.path.join(R, "2026-10-21-qlora-nf4-e2b-3050oem", "train.out")) as f:
        print("QLoRA setup:", *[ln.strip() for ln in f if ln.startswith(('{"load_s"', '{"ple_device"'))])
    print(f"per-layer table in 16-bit: 262,144 x 8,960 x 2 bytes = {262144 * 8960 * 2 / 1e9:.1f} GB")
    from importlib.metadata import version
    print("versions:", *[f"{m} {version(m)}" for m in ("torch", "transformers", "peft", "bitsandbytes")])

    print("held-out runs, 250 problems each (right, share right, tokens per correct, median tokens, s per correct):")
    for label, path in (("uncapped", "2026-10-20-baseline-e2b-3050oem/think-greedy.summary.json"),
                        ("cap 128", "2026-10-20-budget-e2b-3050oem/budget-128.summary.json"),
                        ("cap 256", "2026-10-20-budget-e2b-3050oem/budget-256.summary.json"),
                        ("cap 512", "2026-10-20-budget-e2b-3050oem/budget-512.summary.json"),
                        ("run A", "2026-10-20-qatlora-e2b-3050oem/think-greedy.summary.json"),
                        ("run B plain", "2026-10-20-qatlora-budgetprompt-e2b-3050oem/plain.summary.json"),
                        ("run B prompt 256", "2026-10-20-qatlora-budgetprompt-e2b-3050oem/prompt-256.summary.json"),
                        ("run A lr 1e-3", "2026-10-21-qatlora-lr1e-3-e2b-3050oem/think-greedy.summary.json"),
                        ("QLoRA lr 1e-3", "2026-10-21-qlora-nf4-e2b-3050oem/think-greedy.summary.json")):
        r = js(path)
        print(f"  {label:17s} {r['right']} {100 * r['right'] / r['n']:.1f}% {r['tokens_per_correct']} "
              f"median {r['median_tokens']:.0f} tokens {r['seconds_per_correct']:.2f} s")
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "qat_smoke.py")) as f:
        defaults = re.findall(r'"--(seq|batch)", type=int, default=(\d+)', f.read())
    print("qat_smoke.py defaults (the log does not record arguments):", ", ".join(f"{k} {int(v):,}" for k, v in defaults))

    def server(name):
        """-> (median decode tokens/s per request, minutes from server start to last reply) from llama-server's log."""
        with open(os.path.join(R, name, "server.log")) as f:
            text = f.read()
        tps = [float(x) for x in re.findall(r"\|\s+eval time =.*?([\d.]+) tokens per second", text)]
        last = [ln.split()[0] for ln in text.splitlines() if "print_timing" in ln][-1]  # min.sec.ms.us since start
        m, sec, ms, _ = (int(x) for x in last.split("."))
        return statistics.median(tps), m + (sec + ms / 1000) / 60

    for name in ("2026-10-20-baseline-e2b-3050oem", "2026-10-20-samples-e2b-3050oem",
                 "2026-10-21-qatlora-lr1e-3-e2b-3050oem", "2026-10-21-qlora-nf4-e2b-3050oem"):
        tps, mins = server(name)
        print(f"{name}: median decode {tps:.1f} tok/s per request, {mins // 60:.0f} h {mins % 60:.0f} min to last reply")
    gen = sum(r["n_gen"] for r in jl("2026-10-20-samples-e2b-3050oem/samples.jsonl"))
    _, smin = server("2026-10-20-samples-e2b-3050oem")
    print(f"sampling: {gen:,} generated tokens in {smin:.0f} min = {gen / (smin * 60):.0f} tok/s across 4 slots")
    tmin = jl("2026-10-21-qatlora-lr1e-3-e2b-3050oem/train.log.jsonl")[-1]["elapsed_s"] / 60
    _, emin = server("2026-10-21-qatlora-lr1e-3-e2b-3050oem")
    loop = smin + tmin + emin
    print(f"loop, sample + train + eval: {smin:.0f} + {tmin:.0f} + {emin:.0f} min = {loop // 60:.0f} h {loop % 60:.0f} min")

    sel = jl("2026-10-20-qatlora-e2b-3050oem/selection.jsonl")
    chosen = [r["chosen_len"] for r in sel if r["chosen_len"]]
    every = [n for r in sel for n in r["lens_right"]]
    print(f"shortest right answer mean {statistics.mean(chosen):.1f}, all right answers mean "
          f"{statistics.mean(every):.1f}: {pct_fewer(statistics.mean(chosen), statistics.mean(every))} shorter")
    print(f"lr 1e-3 export moved {134965355 / 238829:.0f}x run A's export")
    print(f"overfit moved 24,385,598 levels in 12 steps = {24385598 / 238829:.0f}x run A's export")


if __name__ == "__main__":
    main()

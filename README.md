# lazy-gemma: teaching Gemma 4 E2B to think less, trained on an RTX 3050

Entry for the [Hacktoberfest Open-Source AI Challenge, Week 3: Slow and Steady](https://dev.to/challenges/hacktoberfest-week3-2026-10-19).
Started 2026-10-20. Deadline 2026-10-25, 11:59 PM PDT; any commit after it gets noted here.

Write-up: `docs/devto-lazy-gemma.md`.

## What it does

Fine-tunes Gemma 4 E2B on an 8 GB RTX 3050 OEM to reach the same answer in fewer thinking tokens. Training runs
through the exact Q4_0 weights of the served GGUF (`qat_lora.py`), and `export_gguf.py` writes the trained weights
back into a copy of that file, so the result serves with the same llama-server flags.

## Results (250 held-out GSM8K problems, greedy, thinking on, 3050)

| Model | Right | Tokens per correct answer | Run |
|---|---|---|---|
| Uncapped | 227 | 977.8 | `benchmarks/runs/2026-10-20-baseline-e2b-3050oem` |
| llama.cpp `thinking_budget_tokens` 256 | 228 | 628.0 | `benchmarks/runs/2026-10-20-budget-e2b-3050oem` |
| Tuned A, shortest right of 4, lr 1e-4 | 220 | 927.2 | `benchmarks/runs/2026-10-20-qatlora-e2b-3050oem` |
| Tuned B, budget in the prompt, plain prompt | 229 | 947.2 | `benchmarks/runs/2026-10-20-qatlora-budgetprompt-e2b-3050oem` |
| Tuned A at lr 1e-3 | 226 | 916.2 | `benchmarks/runs/2026-10-21-qatlora-lr1e-3-e2b-3050oem` |
| Tuned A data, standard QLoRA (NF4), lr 1e-3 | 228 | 963.2 | `benchmarks/runs/2026-10-21-qlora-nf4-e2b-3050oem` |
| Tuned A at lr 1e-3, with the 256 cap | 224 | 634.2 | `benchmarks/runs/2026-10-21-qatlora-lr1e-3-budget256-e2b-3050oem` |
| Tuned on 256-capped samples, lr 1e-3 | 222 | 602.6 | `benchmarks/runs/2026-10-21-qatlora-capped-e2b-3050oem` |

Training: 4,896 MiB peak, 33.4 s per step of 8 sequences. Derived figures: `docs/figures.py`
(output in `docs/evidence/figures.txt`). Check outputs: `benchmarks/runs/2026-10-20-checks-e2b-3050oem`.

## Run it

```
python3 make_splits.py
python3 sample_train.py http://127.0.0.1:8080 samples.jsonl --problems 1000 --samples 4
python3 train.py samples.jsonl out/ --epochs 2 --lr 1e-4
python3 export_gguf.py out/lora.pt lazy.gguf
python3 lazy_eval.py run http://127.0.0.1:8080 data/heldout.jsonl rows.jsonl
make test lint
```

The machine's own `python3` (3.14), torch 2.14.1+cu130, transformers 5.12.1, gguf-py from llama.cpp.

## Credits

- `lazy_eval.py`'s GSM8K prompt and grader are copied from `gen_eval.py` in my earlier
  `gemma4-dev/jev-tpu-v5e1` work.
- The served exact Q4_0 GGUF (`xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf`) was built by `gguf_exact.py` in
  `gemma4-dev/local-llamacpp-1650ti-4b-q4_0/benchmarks/runs/2026-09-30-exact-gguf-e4b-1650ti`.
- Open source: llama.cpp, PyTorch, transformers, gguf-py, Gemma 4 E2B (Gemma terms), GSM8K
  (`openai/grade-school-math`).

## Unmeasured

The Snapdragon 850 decode speed quoted in the write-up (5.00 tok/s) comes from
`gemma4-dev/local-llamacpp-sdm850-2b-q4_0`, measured 2026-10-11; the tuned models were not run there.

# Capped training data, and the 256 cap on the tuned model, 3050 OEM (2026-10-21/22)

Driver: `run.sh` (log `driver.log`). Covers three run directories:

- `2026-10-21-qatlora-lr1e-3-budget256-e2b-3050oem`: the lr 1e-3 tuned GGUF (`gguf.sha256`) with llama.cpp's
  `thinking_budget_tokens` 256.
- `2026-10-21-samples-budget256-e2b-3050oem`: 4 samples on each of the same 1,000 training problems (seed 0,
  T=1.0, top_p 0.95, top_k 64, 4-slot server), thinking capped at 256 tokens by `sample_train.py --budget 256`.
  3,458 of 4,000 right (uncapped samples: 3,524). 932 problems with a right sample; the shortest right per
  problem averages 516.2 generated tokens (uncapped set: 723.8).
- This directory: QAT-LoRA on those 932, lr 1e-3, rank 16, alpha 32, batch 8, 2 epochs, 234 steps in 6,846 s,
  peak 4,880 MiB. Export moved 172,302,863 of 1,835,532,288 Q4_0 levels (`export.out`).

Every eval: 250 held-out problems, greedy, thinking on, `max_tokens` 4096, one request at a time, the
baseline's server flags.

| Model | Cap | Right | Mean / median tokens | Tokens per correct answer |
|---|---|---|---|---|
| Base (`2026-10-20-budget-e2b-3050oem`) | 256 | 228 | 572.8 / 517 | 628.0 |
| Tuned lr 1e-3 | 256 | 224 | 568.2 / 517 | 634.2 |
| Capped-data tuned | none | 222 | 535.1 / 491.5 | 602.6 |
| Capped-data tuned | 256 | 222 | 535.4 / 491.5 | 603.0 |

The capped-data model writes fewer tokens than the base model under the cap, and gets 6 fewer problems right
(222 vs 228). The claim is fewer tokens at equal accuracy, so this run does not make it. Paired on the same
problems (`compare.py`, output `compare.txt`): it gets 6 right that the capped base misses and misses 12 the
capped base gets right (exact McNemar p = 0.238); on the 216 both get right it averages 509.0 tokens against
554.6. Adding the 256 cap to it changes almost nothing (603.0 vs 602.6), so it has learned to stop near the cap
on its own.

Sampler check (2026-10-21, base server, 4 held-out problems, greedy): `/completion` with the reasoning budget
fields and the chat endpoint with `thinking_budget_tokens` 256 both close thinking after 255 tokens.

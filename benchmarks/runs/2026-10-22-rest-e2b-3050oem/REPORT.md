# The rest of the GSM8K test split, 3050 OEM (2026-10-22)

`data/heldout-rest.jsonl` (`make_rest.py`): the 1,069 GSM8K test problems the 250-problem held-out set did not
draw, written before any eval here; none shares a question with the training split. Same eval as the 250 set:
greedy, thinking on, `max_tokens` 4096, one request at a time, the baseline's server flags. Driver: `run.sh`
(log `driver.log`), input hashes in `inputs.sha256`.

| Model | Cap | Set | Right | Tokens per correct answer |
|---|---|---|---|---|
| Base GGUF | 256 | 1,069 | 943 | 654.0 |
| Trained on 256-capped samples (`2026-10-21-qatlora-capped-e2b-3050oem`) | none | 1,069 | 948 | 612.0 |
| Base GGUF | 256 | all 1,319 | 1,171 | 648.9 |
| Trained on 256-capped samples | none | all 1,319 | 1,170 | 610.2 |

All 1,319 = these 1,069 plus the 250 held-out set (`2026-10-20-budget-e2b-3050oem`,
`2026-10-21-qatlora-capped-e2b-3050oem`).

Paired (`compare.py`, output `compare.txt`): on all 1,319, 51 problems only the trained model gets right and
52 only the capped base gets right (exact McNemar p = 1.000). Accuracy is equal, and the trained model uses 6.0%
fewer tokens per correct answer with no cap set. On the 1,119 problems both get right it averages 515.1 tokens
against 549.7. The 6-problem gap on the 250 set (222 vs 228) does not hold up: on the 1,069 the trained model
gets 5 more right.

#### Cap 128 (2026-10-22)

Same recipe at a 128 cap: 4 samples on the same 1,000 training problems with thinking capped at 128
(`2026-10-22-samples-budget128-e2b-3050oem`, 3,484 of 4,000 right), shortest right per problem (939 problems,
404.2 generated tokens on average), QAT-LoRA lr 1e-3, 2 epochs (`2026-10-22-qatlora-capped128-e2b-3050oem`,
154,373,515 Q4_0 levels changed). Base with the 128 cap on the 250: `2026-10-20-budget-e2b-3050oem/budget-128.jsonl`.

| Model | Cap | Set | Right | Tokens per correct answer |
|---|---|---|---|---|
| Base GGUF | 128 | 1,069 | 942 | 543.1 |
| Trained on 128-capped samples | none | 1,069 | 943 | 486.7 |
| Base GGUF | 128 | all 1,319 | 1,166 | 544.2 |
| Trained on 128-capped samples | none | all 1,319 | 1,161 | 492.2 |

On all 1,319: 63 problems only the trained model gets right, 68 only the capped base (McNemar p = 0.727). The
trained model gets 5 fewer right, a gap these 1,319 problems cannot tell from noise, and uses 9.6% fewer tokens
per correct answer than the base with the 128 cap. On the 1,098 both get right it averages 399.9 tokens against
437.5. The base with the 128 cap loses 5 problems against the 256 cap (1,166 vs 1,171) at 544.2 tokens per
correct answer against 648.9, so the 128 cap is the stronger no-training baseline.

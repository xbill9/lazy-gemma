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

#### Cap 512 (2026-10-23)

Same recipe at a 512 cap (`2026-10-22-samples-budget512-e2b-3050oem`, 3,497 of 4,000 right; 941 problems,
664.3 generated tokens on average; `2026-10-22-qatlora-capped512-e2b-3050oem`, 113,689,221 Q4_0 levels changed).
Base with the 512 cap on the 250: `2026-10-20-budget-e2b-3050oem/budget-512.jsonl`.

| Model | Cap | Set | Right | Tokens per correct answer |
|---|---|---|---|---|
| Base GGUF | 512 | 1,069 | 957 | 855.6 |
| Trained on 512-capped samples | none | 1,069 | 958 | 763.7 |
| Base GGUF | 512 | all 1,319 | 1,185 | 855.9 |
| Trained on 512-capped samples | none | all 1,319 | 1,179 | 764.3 |

On all 1,319: 42 only the trained model gets right, 48 only the capped base (McNemar p = 0.598). 6 fewer right,
10.7% fewer tokens per correct answer; on the 1,137 both get right, 656.7 tokens against 739.6.

#### Against the uncapped base (2026-10-23)

Base GGUF, no cap, on the 1,069 (`base-think`): 960 right at 998.9 tokens per correct answer. With the 250-set
baseline (`2026-10-20-baseline-e2b-3050oem`), all 1,319: 1,187 right at 994.9, 4 replies hit the 4,096 limit.

| Trained on samples capped at | Right of 1,319 | Tokens per correct answer | Fewer than uncapped base | McNemar p |
|---|---|---|---|---|
| 512 | 1,179 | 764.3 | 23.2% | 0.461 |
| 256 | 1,170 | 610.2 | 38.7% | 0.100 |
| 128 | 1,161 | 492.2 | 50.5% | 0.030 |

Against the uncapped base, the 128-trained model gets 26 fewer right (80 only the base gets right, 54 only the
trained model), and at p = 0.030 that loss is measurable. The 512-trained model gets 8 fewer right, within noise.

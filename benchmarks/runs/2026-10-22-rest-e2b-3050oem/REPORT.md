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

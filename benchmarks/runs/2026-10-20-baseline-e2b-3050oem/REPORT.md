# Baseline: E2B Q4_0, thinking on, 3050 OEM (2026-10-20)

Served file `gemma-4-E2B-it-q4_0-exact.gguf` (`xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf`), llama.cpp
`b11696-3-g7182ef71e`, command in `server.cmd`. `lazy_eval.py`, 250 held-out problems
(`data/heldout.jsonl`, seed 0), greedy, thinking on per request, `max_tokens` 4096, one request at a time,
prompt cache off.

| | value |
|---|---|
| right | 227 / 250 (0.908) |
| mean / median output tokens | 887.8 / 788 |
| output tokens per correct answer | 977.8 |
| seconds per correct answer (3050) | 9.06 |
| hit the 4,096 limit | 2 |

Output tokens include the thinking block. Seconds on the 3050 are inflated for some early rows: a CPU-side
export test ran beside them. The speed claim is measured on the Snapdragon 850, not here.

Rows: `think-greedy.jsonl` (raw thinking and answer per problem). Summary: `think-greedy.summary.json`.

---
library_name: gguf
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
pipeline_tag: text-generation
base_model:
- xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf
base_model_relation: finetune
tags:
- gemma4
- gguf
- llama.cpp
- q4_0
- qat
- lora
- reasoning
- gsm8k
- text-only
---

# Gemma 4 E2B-it, trained to think in about 256 tokens, Q4_0 GGUF (unofficial, experimental)

**This is an unofficial research build, made and published independently of Google.** It is
[`xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf`](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf)
after a LoRA fine-tune that teaches it to close its thinking block after about 256 tokens, written back
into the same Q4_0 grid. Same size, same tensors, same llama-server flags.

It is part of [lazy-gemma](https://github.com/xbill9/lazy-gemma), a Hacktoberfest 2026 Week 3 entry.

**Newer model:** [`xbill9/gemma-4-E2B-it-lazy-capped64-q4_0-gguf`](https://huggingface.co/xbill9/gemma-4-E2B-it-lazy-capped64-q4_0-gguf),
trained the same way on 64-capped samples, gets more right with fewer tokens: 1,175 of 1,319 at 454.0 tokens
per correct answer.

#### Results

All 1,319 GSM8K test problems, greedy, thinking on, `max_tokens` 4096, llama.cpp `b11696-3-g7182ef71e`
on an RTX 3050 OEM (8 GB). Correctness is graded by code.

| Model | Thinking cap | Right | Tokens per correct answer |
|---|---|---|---|
| Base GGUF | none | 1,187 | 994.9 |
| Base GGUF | llama.cpp `thinking_budget_tokens` 256 | 1,171 | 648.9 |
| **This GGUF** | none | **1,170** | **610.2** |

Against the base with llama.cpp's 256 cap it gets the same accuracy, 1,170 against 1,171 (51 problems only
this model gets right, 52 only the capped base; exact McNemar p = 1.000), at 6.0% fewer tokens per correct
answer and with no cap set. Against the uncapped base it gets 17 fewer right (p = 0.100) at 38.7% fewer
tokens. Adding the 256 cap to this model changes almost nothing: on the first 250 problems, 603.0 tokens per
correct answer with the cap against 602.6 without.

Tokens include the thinking block, counted by the server with the model's tokenizer. Rows, logs and the
comparison script are in the repo under
[`benchmarks/runs/2026-10-22-rest-e2b-3050oem`](https://github.com/xbill9/lazy-gemma/tree/main/benchmarks/runs/2026-10-22-rest-e2b-3050oem)
and [`benchmarks/runs/2026-10-21-qatlora-capped-e2b-3050oem`](https://github.com/xbill9/lazy-gemma/tree/main/benchmarks/runs/2026-10-21-qatlora-capped-e2b-3050oem).

#### How it was trained

- **Data:** 4 samples on each of 1,000 GSM8K training problems (seed 0, T=1.0, top_p 0.95, top_k 64), with
  thinking capped at 256 tokens by llama.cpp's reasoning budget. 3,458 of 4,000 samples were right. Per
  problem, code kept the shortest right sample: 932 problems, 516.2 generated tokens on average. Train and
  test problems are disjoint.
- **Training:** LoRA rank 16, alpha 32, on q/k/v/o/gate/up/down, lr 1e-3, batch 8, 2 epochs (234 steps),
  1h54m on the RTX 3050 at a 4,880 MiB peak. The forward pass runs through the exact Q4_0 weights of the
  base GGUF (fake-quant of weight plus LoRA delta, straight-through gradient), so the adapter is learned
  for the 4-bit weights that ship.
- **Export:** the trained weights are rounded back onto each block's Q4_0 grid and written into a copy of
  the base GGUF. 172,302,863 of 1,835,532,288 Q4_0 levels changed. Metadata, tokenizer and chat template are
  unchanged.

#### Running it

```bash
llama-server -m gemma-4-E2B-it-lazy-capped-q4_0.gguf -ngl 99 -fa 1 -c 8192 --jinja
```

Thinking is on by default; the reply's thinking lands in `reasoning_content` and the answer in `content`.

SHA-256: `a2527137e332e9b80acee264dcc3aa596c2bd8cc16bfd8ed592ba99fd49f97db`

#### Limitations

- Trained and measured on GSM8K only. Other tasks were not measured.
- Text only. Tested with llama.cpp CUDA on one RTX 3050 OEM.
- Unofficial and experimental. Report problems on the GitHub repo, not to Google.

#### License and attribution

Gemma 4 is released by Google DeepMind under the
[Apache 2.0 license](https://ai.google.dev/gemma/docs/gemma_4_license). This repository redistributes
fine-tuned Gemma 4 weights under the same license. The base weights, tokenizer and chat template are
Google DeepMind's; this build is not affiliated with or endorsed by Google. Training data: GSM8K
(`openai/grade-school-math`).

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

# Gemma 4 E2B-it, trained on 64-token thinking, Q4_0 GGUF (unofficial, experimental)

**This is an unofficial research build, made and published independently of Google.** It is
[`xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf`](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf)
after a LoRA fine-tune that trains it on its own shortest right answers with thinking cut at 64 tokens, written back
into the same Q4_0 grid. Same size, same tensors, same llama-server flags.

It is part of [lazy-gemma](https://github.com/xbill9/lazy-gemma), a Hacktoberfest 2026 Week 3 entry.

#### Results

All 1,319 GSM8K test problems, greedy, thinking on, `max_tokens` 4096, llama.cpp `b11696-3-g7182ef71e`
on an RTX 3050 OEM (8 GB). Correctness is graded by code. This model runs with no cap.

| Model | Thinking cap | Right | Tokens per correct answer |
|---|---|---|---|
| Base GGUF | none | 1,187 | 994.9 |
| Base GGUF | llama.cpp `thinking_budget_tokens` 512 | 1,185 | 855.9 |
| Base GGUF | 256 | 1,171 | 648.9 |
| Base GGUF | 128 | 1,166 | 544.2 |
| Base GGUF | 64 | 1,117 | 580.9 |
| **This GGUF** | none | **1,175** | **454.0** |

Against the uncapped base it uses 54.4% fewer tokens per correct answer and gets 12 fewer right (50 problems
only this model gets right, 62 only the base; exact McNemar p = 0.299), a gap 1,319 problems cannot tell from
noise. Against the base with the 128 cap, the best setting without training, it gets 9 more right at 16.6%
fewer tokens. A 64 cap on the base model costs 70 problems against no cap; training on 64-capped samples gets
58 of them back (p = 2.1e-05 against the base with the 64 cap).

The same recipe at caps 128, 256 and 512 gave 1,161, 1,170 and 1,179 right at 492.2, 610.2 and 764.3 tokens per
correct answer: accuracy does not fall steadily as the cap tightens. Each model is one training run with one
seed.

Tokens include the thinking block, counted by the server with the model's tokenizer. Rows, logs and the
comparison script are in the repo under
[`benchmarks/runs/2026-10-22-rest-e2b-3050oem`](https://github.com/xbill9/lazy-gemma/tree/main/benchmarks/runs/2026-10-22-rest-e2b-3050oem)
and [`benchmarks/runs/2026-10-23-qatlora-capped64-e2b-3050oem`](https://github.com/xbill9/lazy-gemma/tree/main/benchmarks/runs/2026-10-23-qatlora-capped64-e2b-3050oem).

#### How it was trained

- **Data:** 4 samples on each of 1,000 GSM8K training problems (seed 0, T=1.0, top_p 0.95, top_k 64), with
  thinking capped at 64 tokens by llama.cpp's reasoning budget. 3,495 of 4,000 samples were right. Per
  problem, code kept the shortest right sample: 943 problems, 376.1 generated tokens on average (939 fit the 2,048-token training length). Train and
  test problems are disjoint.
- **Training:** LoRA rank 16, alpha 32, on q/k/v/o/gate/up/down, lr 1e-3, batch 8, 2 epochs (236 steps),
  1h44m on the RTX 3050 at a 4,895 MiB peak. The forward pass runs through the exact Q4_0 weights of the
  base GGUF (fake-quant of weight plus LoRA delta, straight-through gradient), so the adapter is learned
  for the 4-bit weights that ship.
- **Export:** the trained weights are rounded back onto each block's Q4_0 grid and written into a copy of
  the base GGUF. 175,425,067 of 1,835,532,288 Q4_0 levels changed. Metadata, tokenizer and chat template are
  unchanged.

#### Running it

```bash
llama-server -m gemma-4-E2B-it-lazy-capped64-q4_0.gguf -ngl 99 -fa 1 -c 8192 --jinja
```

Thinking is on by default; the reply's thinking lands in `reasoning_content` and the answer in `content`.

SHA-256: `750e778349575310e5a45df422eefde7d6b6fa66e33a15d55d7ddb9296f697a7`

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

---
title: "Train Gemma 4 on Its Own Cut-Short Answers: Half the Tokens on an 8 GB RTX 3050"
published: false
description: "llama.cpp's thinking cap trades accuracy for tokens. Fine-tuning Gemma 4 E2B on its own answers sampled under a 64-token cap halves tokens per correct answer on all 1,319 GSM8K test problems, with no measurable accuracy loss and no cap at inference."
tags: gemma, llamacpp, machinelearning, finetuning
cover_image: https://raw.githubusercontent.com/xbill9/lazy-gemma/main/docs/caps/devto-cover.2ca1e7c8.jpg
---

This article provides a step by step guide to fine-tuning Gemma 4 E2B on an 8 GB NVIDIA RTX 3050 so that it thinks in 64 tokens, using answers the model wrote itself under llama.cpp's thinking cap. Every log, script and number is committed.

https://github.com/xbill9/lazy-gemma

**On all 1,319 GSM8K test problems, Gemma 4 E2B uses 994.9 tokens per correct answer and gets 1,187 right. Trained on its own shortest right answers sampled with thinking cut at 64 tokens, it uses 454.0 and gets 1,175 right, with no cap set at inference: 54.4% fewer tokens, and an accuracy gap of 12 problems that 1,319 problems cannot tell from chance. The same 64-token cap applied to the untrained model costs it 70 problems. Training on the cut-short answers wins 58 of them back.**

---

#### Where Part 1 Left Off

[Part 1](https://dev.to/xbill/fine-tuning-gemma-4-on-an-8-gb-rtx-3050-slow-low-end-and-still-viable-291f) built the loop on one RTX 3050: sample four answers per training problem, keep the shortest right one, train through the exact 4-bit weights of the served GGUF, and write the result back into the same file. The tuned models cut tokens per correct answer by at most 6.3%, while llama.cpp's `thinking_budget_tokens` cut them by 35.8% with no training.

The reason was the data. The shortest of four uncapped answers was only 11.8% shorter than a typical right answer, so the model had little short thinking to learn from. This part changes one thing: the training answers are sampled with the thinking cap on.

---

#### At This Point You Should Have…

- An NVIDIA GeForce RTX 3050 OEM, 8,192 MiB
- llama.cpp built with CUDA, at `b11696-3-g7182ef71e`
- The served model: `xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf`
- The machine's own Python 3.14 with torch 2.14.1 (CUDA 13.0) and transformers 5.12.1
- A clone: `git clone https://github.com/xbill9/lazy-gemma`, with Part 1's `data/train.jsonl` and `data/heldout.jsonl`

---

#### Step 1 — Use the Whole Test Split

Part 1 graded on 250 of GSM8K's 1,319 test problems. `make_rest.py` writes the other 1,069 to `data/heldout-rest.jsonl`, after checking the downloaded test file against the sha256 recorded by `make_splits.py` and that no question appears in the training split:

```text
$ python3 make_rest.py
{"heldout_rest": 1069}
```

Every result below is on all 1,319, as the 250 plus the 1,069, with the same eval: greedy, thinking on, `max_tokens` 4096, one request at a time, graded by code.

---

#### Step 2 — Measure the Cap at Five Settings

The untrained model, uncapped and with `thinking_budget_tokens` at 512, 256, 128 and 64:

| Thinking cap | Right of 1,319 | Tokens per correct answer |
| :--- | :--- | :--- |
| none | 🥇 1,187 | 994.9 |
| 512 | 🥈 1,185 | 855.9 |
| 256 | 1,171 | 648.9 |
| 128 | 1,166 | 🥇 544.2 |
| 64 | 1,117 | 580.9 |

The cap is a dial. Each halving costs a few right answers down to 128, and at 64 it costs 49 more and saves nothing: the model, cut off mid-thought, writes a longer answer and gets more of them wrong. The best setting with no training is 128.

---

#### Step 3 — Sample With the Cap On

`sample_train.py --budget B` sends llama.cpp's reasoning budget fields with each `/completion` request, the same cap the eval's `thinking_budget_tokens` applies on the chat endpoint. The forced end-of-thinking token lands in the sampled token ids like a sampled one, so the trainer sees exactly what the server produced.

On the same 1,000 training problems as Part 1, four samples each at temperature 1.0, on a 4-slot llama-server:

```text
samples 64  {"samples": 4000, "right": 3495, "problems": 1000}
samples 128 {"samples": 4000, "right": 3484, "problems": 1000}
samples 256 {"samples": 4000, "right": 3458, "problems": 1000}
samples 512 {"samples": 4000, "right": 3497, "problems": 1000}
```

Uncapped, Part 1 got 3,524 right. The cap barely changes how many training answers are right; it changes how long they are.

---

#### Step 4 — Train on the Shortest Right Capped Answer

`train.py` is unchanged from Part 1: the shortest right answer per problem, chosen in code, QAT-LoRA rank 16 at lr 1e-3, batch 8, two passes.

```text
cap 64   {"problems": 1000, "with_right": 943, "train_seqs": 939, "mean_gen_len": 376.1}
cap 128  {"problems": 1000, "with_right": 939, "train_seqs": 936, "mean_gen_len": 404.2}
cap 256  {"problems": 1000, "with_right": 932, "train_seqs": 932, "mean_gen_len": 516.2}
cap 512  {"problems": 1000, "with_right": 941, "train_seqs": 938, "mean_gen_len": 664.3}
```

Part 1's uncapped targets averaged 723.8 tokens. The 64-cap targets average 376.1, about half.

Each run fits the card the same way as Part 1, under 4.9 GB:

```text
cap 64   {"step": 236, "of": 236, "loss": 0.0885, "elapsed_s": 6248, "peak_mib": 4895}
cap 128  {"step": 234, "of": 234, "loss": 0.145,  "elapsed_s": 6341, "peak_mib": 4869}
cap 256  {"step": 234, "of": 234, "loss": 0.1089, "elapsed_s": 6846, "peak_mib": 4880}
cap 512  {"step": 236, "of": 236, "loss": 0.0834, "elapsed_s": 7568, "peak_mib": 4883}
```

Shorter targets train faster: 1 h 44 min at 64, 2 h 6 min at 512. Each export moves 6.2% to 9.6% of the 4-bit weights:

```text
cap 64   levels changed 175425067 of 1835532288 in trained tensors
cap 512  levels changed 113689221 of 1835532288 in trained tensors
```

---

#### Step 5 — Measure the Tuned Models on All 1,319

Each tuned model runs with no cap. Against the untrained model with the same cap, and against the untrained model uncapped:

| Trained on answers capped at | Right of 1,319 | Tokens per correct answer | vs same cap, untrained | vs uncapped, untrained |
| :--- | :--- | :--- | :--- | :--- |
| 512 | 🥇 1,179 | 764.3 | 6 fewer right, 10.7% fewer tokens | 8 fewer right, 23.2% fewer tokens |
| 256 | 1,170 | 610.2 | 1 fewer right, 6.0% fewer tokens | 17 fewer right, 38.7% fewer tokens |
| 128 | 1,161 | 492.2 | 5 fewer right, 9.5% fewer tokens | 26 fewer right, 50.5% fewer tokens |
| 64 | 🥈 1,175 | 🥇 454.0 | 58 more right, 21.8% fewer tokens | 12 fewer right, 54.4% fewer tokens |

Whether an accuracy gap is real is checked per problem with an exact McNemar test, which counts the problems only one model gets right:

```text
cap 64, vs base uncapped, all 1,319 (1319): right base 1187, capped-data 1175; only capped-data right 50, only base right 62, McNemar p=0.299
cap 64, all 1,319 (1319): right base 1117, capped-data 1175; only capped-data right 120, only base right 62, McNemar p=0.000
cap 128, vs base uncapped, all 1,319 (1319): right base 1187, capped-data 1161; only capped-data right 54, only base right 80, McNemar p=0.030
```

The 64-trained model's 12-problem gap to the uncapped model (p = 0.299) is within chance. Its 58-problem lead over the untrained model at the same cap is far outside it (p = 2.1e-05). The 128-trained model is the only one with a measurable loss against the uncapped model: 26 fewer right, p = 0.030.

Against the best untrained setting, the 128 cap, the 64-trained model gets 9 more right at 16.6% fewer tokens per correct answer.

🔎 **Tip: 250 problems are too few to judge accuracy.** On Part 1's 250 held-out problems, the 256-trained model got 222 right against the capped base's 228, and the per-problem test gave p = 0.238. On all 1,319 the same two models get 1,170 and 1,171. Both models are at about 89%, so a few hundred problems leave a gap of several right answers inside chance. Grade on the whole test split before calling an accuracy loss.

---

#### Step 6 — Where the Tokens Go

The eval rows keep the thinking and the answer as separate text. `docs/caps/think_lengths.py` counts both with Gemma 4's tokenizer, over all 1,319:

```text
base uncapped: thinking median 512 mean 607.6; answer median 264; thinking over 64/128/256: 1319/1319/1309 of 1319
base + cap 64: thinking median 63 mean 63.0; answer median 321; thinking over 64/128/256: 0/0/0 of 1319
tuned 512: thinking median 425 mean 410.1; answer median 245; thinking over 64/128/256: 1319/1314/1184 of 1319
tuned 128: thinking median 127 mean 126.9; answer median 252; thinking over 64/128/256: 1319/0/0 of 1319
tuned 64: thinking median 63 mean 63.0; answer median 274; thinking over 64/128/256: 2/0/0 of 1319
```

The tuned models stop thinking where their training cap stopped them, with no cap set: the 64-trained model's thinking runs over 64 tokens on 2 problems of 1,319. The 512-trained model stops earlier than its cap, at a median of 425.

The 64-trained model and the capped untrained model think for the same 63 tokens. The difference is the answer: the untrained model, cut off, writes a median of 321 answer tokens and gets 1,117 right; the trained model writes 274 and gets 1,175 right. After the same short thinking, it writes a shorter answer and gets more of them right.

---

#### Step 7 — Repeat With a Second Seed

SEED-PENDING

---

#### Compare and Contrast

| | Untrained, uncapped | Untrained, cap 128 | Tuned on 64-capped answers |
| :--- | :--- | :--- | :--- |
| Right of 1,319 | 🥇 1,187 | 🥉 1,166 | 🥈 1,175 |
| Tokens per correct answer | 994.9 | 🥈 544.2 | 🥇 454.0 |
| Median thinking tokens | 512 | 127 | 63 |
| Training needed | 🟢 none | 🟢 none | ⚠️ 2 h 18 min sampling, 1 h 44 min training |
| Works with any client | 🟢 yes | ⚠️ needs the request field | 🟢 yes |

---

#### So, Which One?

To make Gemma 4 E2B think less with no training, set `thinking_budget_tokens` to 128: it cuts tokens per correct answer by 45.3% for 21 fewer right out of 1,319. To halve the tokens and keep the accuracy, fine-tune on the model's own answers sampled under a 64-token cap: one night on the 3050, and the result serves with no request field, from any client. Avoid a 64-token cap on the untrained model: it costs 70 right answers and saves fewer tokens than 128.

The 64-trained model is on Hugging Face as [`xbill9/gemma-4-E2B-it-lazy-capped64-q4_0-gguf`](https://huggingface.co/xbill9/gemma-4-E2B-it-lazy-capped64-q4_0-gguf), the same size and format as the file it was trained from.

---

#### Summary

The goal of this article was to make Gemma 4 E2B reach the same answer in fewer tokens with a fine-tune that beats llama.cpp's thinking cap. The key to the solution was the training data: answers the model writes under the cap, so the shortest right answer is half as long as the model's uncapped ones. The results on all 1,319 GSM8K test problems were:

- 🟢 Tuned on 64-capped answers: 454.0 tokens per correct answer against 994.9 uncapped, 54.4% fewer, with 1,175 right against 1,187 (McNemar p = 0.299).
- 🟢 It beats the best untrained setting, the 128 cap: 9 more right, 16.6% fewer tokens, and no cap needed at inference.
- 🟢 Every tuned model stops thinking near its training cap with no cap set; the 64-trained model passes 64 thinking tokens on 2 of 1,319 problems.
- 🟢 The whole loop runs on one 8 GB RTX 3050, under 4.9 GB while training.
- ⚠️ The 512-, 256- and 128-trained models save 23% to 51% of tokens against the uncapped model, and the 128-trained model loses 26 right answers (p = 0.030).
- ❌ The 64-token cap on the untrained model costs 70 right answers and saves fewer tokens than the 128 cap.
- ❌ 250 problems were too few: a 6-problem gap there became 1 problem on all 1,319.

Scope: one RTX 3050 OEM, llama.cpp `b11696-3-g7182ef71e`, GSM8K only, one greedy eval per model on all 1,319 test problems, four samples per training problem on the same 1,000 training problems at every cap, one training run per cap. Accuracy differences are tested per problem with an exact two-sided McNemar test.

The strategy for training Gemma 4 to think less on a low-end 8 GB GPU was validated with an incremental step by step approach.

---

#### References

- [Part 1: Fine-Tuning Gemma 4 on an 8 GB RTX 3050](https://dev.to/xbill/fine-tuning-gemma-4-on-an-8-gb-rtx-3050-slow-low-end-and-still-viable-291f)
- [lazy-gemma on GitHub](https://github.com/xbill9/lazy-gemma)
- [Results report for all 1,319 problems](https://github.com/xbill9/lazy-gemma/blob/main/benchmarks/runs/2026-10-22-rest-e2b-3050oem/REPORT.md)
- [Tuned on 64-capped answers, Q4_0 GGUF](https://huggingface.co/xbill9/gemma-4-E2B-it-lazy-capped64-q4_0-gguf)
- [Gemma 4 E2B, exact Q4_0 GGUF used as the base](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [GSM8K](https://github.com/openai/grade-school-math)

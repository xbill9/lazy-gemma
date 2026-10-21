---
title: "Fine-Tuning Gemma 4 on an 8 GB RTX 3050: Slow, Low End and Still Viable"
published: false
description: "An 8 GB RTX 3050 serves Gemma 4 E2B at 109 tokens per second, and that speed makes it a training card too. One card samples its own training data, fine-tunes the model to think less and grades it, in 6 h 44 min."
tags: devchallenge, hf26challenge, gemma, llamacpp
cover_image: https://raw.githubusercontent.com/xbill9/lazy-gemma/main/docs/devto-cover.5597ecd9.jpg
---

*This is a submission for the [Hacktoberfest Open-Source AI Challenge Week 3: Slow and Steady](https://dev.to/challenges/hacktoberfest-week3-2026-10-19)*

This article provides a step by step guide to fine-tuning Gemma 4 E2B on an 8 GB NVIDIA RTX 3050, the low end of a desktop GPU, so that the model thinks less before it answers. Every log, script and number is committed.

https://github.com/xbill9/lazy-gemma

**The RTX 3050 is an 8 GB inference card: it serves Gemma 4 E2B at 109 tokens per second. That speed makes it a training card too, because cheap fine-tuning starts with inference. On one 3050 the whole loop runs in 6 h 44 min: the model writes its own training data, 3.7 million tokens in 4 h 1 min; trains on it in 2 h 10 min at a 4,896 MiB peak, or 1 h 6 min with the standard QLoRA stack; and is graded on 250 held-out problems in 32 min. The trained file drops straight back into llama.cpp, byte for byte the format it started in. The lazy model itself is not there yet: the best tuned run cuts tokens per correct answer by 6.3% with one fewer right answer out of 250, while llama.cpp's built-in thinking cap of 256 tokens cuts them by 35.8% with no training at all.**

---

## What I Built

Gemma 4 thinks before it answers. On grade-school maths problems the E2B model writes a median of 788 tokens per reply on the 3050, and most of that is thinking. On a fast GPU that costs a few seconds. On a slow device it is most of the wait.

The lazy idea: teach the model to reach the same answer in fewer words. Ask it the same problem four times, keep the shortest answer that is right, and fine-tune it on those.

The 3050 question: a used 8 GB card is good for inference, so can it fine-tune too, or does training need rented hardware? This kind of fine-tuning starts with inference: the model has to answer thousands of problems before there is anything to train on, and that took longer than the training itself. A card that serves 109 tokens per second already does that part. The answer here is that the same card runs the whole loop, overnight, and the trained model ships in the same 4-bit file it serves.

What the project delivers:

- a trainer that runs on an 8 GB card and trains through the exact 4-bit weights of the served model
- an exporter that writes the trained weights back into that file, with nothing else changed
- the standard QLoRA recipe (transformers, PEFT, bitsandbytes) fitted to the same card, as a comparison
- an evaluation on 250 held-out problems, graded by code, against the untrained model and against llama.cpp's thinking cap

---

## Demo

The baseline and each tuned model answer the same 250 held-out GSM8K problems, greedy, thinking on, through llama-server on the 3050:

| Model | Right | Tokens per correct answer |
| :--- | :--- | :--- |
| Gemma 4 E2B, uncapped | 227 (90.8%) | 977.8 |
| Same, llama.cpp thinking cap 128 | 224 (89.6%) | 548.8 |
| Same, llama.cpp thinking cap 256 | 228 (91.2%) | 628.0 |
| Same, llama.cpp thinking cap 512 | 228 (91.2%) | 856.8 |
| Tuned A: shortest right answer, lr 1e-4 | 220 (88.0%) | 927.2 |
| Tuned B: budget in the prompt, plain prompt | 229 (91.6%) | 947.2 |
| Tuned B, prompt says "at most 256 tokens" | 224 (89.6%) | 903.2 |
| Tuned A at lr 1e-3 | 226 (90.4%) | 916.2 |
| QLoRA at lr 1e-3 | 228 (91.2%) | 963.2 |

Tokens include the thinking. A tuned model only counts as a win if it uses fewer tokens per correct answer at the same accuracy.

---

## Code

https://github.com/xbill9/lazy-gemma

| Path | What |
| :--- | :--- |
| `make_splits.py` | Fixed train and held-out sets from GSM8K, written before any training |
| `lazy_eval.py` | Runs and grades the held-out set against llama-server |
| `sample_train.py` | Four answers per training problem, with the exact token ids |
| `qat_lora.py` | The model, built on the served 4-bit weights, with trainable adapters |
| `train.py` | Picks the shortest right answer per problem and trains |
| `qlora_train.py` | The same data and settings on the standard QLoRA stack |
| `export_gguf.py` | Writes the trained weights back into a copy of the served file |
| `benchmarks/runs/` | Every run: script, logs, per-problem rows, summaries |

---

## How I Built It

#### At This Point You Should Have…

- An NVIDIA GeForce RTX 3050 OEM, 8,192 MiB, driver 615.78.08, in a desktop with 46 GB of RAM
- llama.cpp built with CUDA, at `b11696-3-g7182ef71e`
- The served model: `xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf`, Gemma 4 E2B with Google's quantization-aware weights written exactly as 4-bit Q4_0
- `google/gemma-4-E2B-it-qat-q4_0-unquantized` from Hugging Face, for the model's layout and tokenizer
- The machine's own Python 3.14 with torch 2.14.1 (CUDA 13.0) and transformers 5.12.1; peft 0.21.2 and bitsandbytes 0.50.2 for the QLoRA comparison
- A clone: `git clone https://github.com/xbill9/lazy-gemma`

---

#### Step 1 — Fix the Problems Before Anything Trains

`make_splits.py` takes the 7,473 GSM8K training problems for training and draws 250 of the 1,319 test problems, by seed 0, for the held-out set. Both are written to `data/` with the source files' sha256 in `data/manifest.json`. The trainer never sees a held-out problem.

Grading is code: the number after `Answer:` in the reply, compared with the reference. The prompt and grader come from my earlier GSM8K work (see Prior Work).

---

#### Step 2 — Measure the Baseline and the Free Fix

The untrained model first, then llama.cpp's own way to think less: `thinking_budget_tokens` on each request, which ends the thinking block at that many tokens.

```text
uncapped    {"n": 250, "right": 227, "accuracy": 0.908, "truncated": 2, "mean_tokens": 887.8, "median_tokens": 788,   "tokens_per_correct": 977.8, "seconds_per_correct": 9.06}
budget 256  {"n": 250, "right": 228, "accuracy": 0.912, "truncated": 0, "mean_tokens": 572.8, "median_tokens": 517.0, "tokens_per_correct": 628.0, "seconds_per_correct": 5.8}
```

The cap at 256 keeps accuracy and cuts tokens per correct answer by 35.8%. That is the number a fine-tune has to beat.

---

#### Step 3 — Train Through the Weights That Ship

The deliverable is a 4-bit file for small cards. A model trained in 16-bit and rounded to 4-bit afterwards ends up with weights it never trained on. So `qat_lora.py` trains through the 4-bit weights themselves:

- Every weight is read straight out of the served GGUF as a whole number from -8 to 7 plus a scale per group of 32.
- A small trainable adapter (LoRA, rank 16, 24.2M values) adds a correction to each weight, and the sum is rounded back onto the same 4-bit grid on every forward pass.
- Gradients pass through the rounding as if it were not there, the standard trick for training through a step function.

With the adapter at zero, the model's weights equal the file's:

```text
blk.0.ffn_gate.weight    exact=True mismatched=0/9437184
blk.20.attn_q.weight     exact=True mismatched=0/3145728
blk.34.ffn_down.weight   exact=True mismatched=0/18874368
blk.7.inp_gate.weight    exact=True mismatched=0/393216
token_embd.weight        exact=True mismatched=0/402653184
```

```text
{"top1_agree": 510, "of": 512, "rate": 0.9961}
```

Its next-token choice matches llama-server's on 510 of 512 tokens of a real reply.

🔎 **Tip: the lazy route is a few hundred lines of PyTorch.** The usual recipe for a 4-bit fine-tune is QLoRA: transformers, PEFT and bitsandbytes. It trains against bitsandbytes' own 4-bit format, NF4, so the adapter learns weights that get rounded a second time on the way into a Q4_0 GGUF. llama.cpp's trainer reads the GGUF directly, and at the commit used here it stops on Gemma 4 with `unsupported unary op for backward pass: TANH`; its `llama-finetune` trains full weights only. Reading the weights out of the GGUF and writing the loop around them took `qat_lora.py` (216 lines) and `train.py` (179 lines), and the two checks above show the model it trains is the shipped file. The loop itself, AdamW with warmup, a cosine schedule and gradient clipping, is what TRL's trainer does in one call. The 4-bit grid is the part none of those libraries offered. Step 8 runs the standard recipe on the same data and measures the difference.

---

#### Step 4 — Fit It in 8 GB

Three choices make training fit on the 3050:

- The 4-bit weights stay 4-bit in memory, stored as int8 with fp16 scales.
- Gemma 4's per-layer embedding table (2.24 GiB as int8) stays on the CPU. Only the rows a sequence uses are looked up.
- Gradient checkpointing, one sequence at a time, and the loss computed in chunks so the 262,144-word output layer never holds a whole sequence at once.

One step of 8 sequences at 1,024 tokens:

```text
{"load_s": 26.5, "load_peak_mib": 2434, "unfilled": [], "lora_params": 24158208}
{"step": 1, "loss": 16.508, "s_per_step": 35.68, "peak_mib": 4725, "reserved_mib": 5120}
```

That uses 4.7 GB of the card's 8 GB. Overfitting one reply at lr 1e-3 brings the loss from 0.1831 to 0.0235 in 12 steps and moves 24,385,598 of the 1,835,532,288 4-bit weights, so the gradients reach the grid.

The standard QLoRA stack meets the same two limits on this card. Gemma 4's per-layer table is 4.7 GB in 16-bit, and Hugging Face's Trainer and accelerate place the whole model on the GPU, table included, so the first backward pass runs out of memory. `qlora_train.py` moves the table to the CPU, marks the model as split across CPU and GPU so neither library moves it back, and computes the loss in the same chunks. Then it fits:

```text
{"load_s": 3.9, "load_peak_mib": 6201, "lora_params": 24158208}
{"ple_device": "cpu", "gpu_mib": 1785}
```

The peak is the table passing through the GPU while loading; training starts from 1,785 MiB. Same adapter size, 24,158,208 values, at 17.1 s per step.

---

#### Step 5 — Sample Four Answers per Problem

`sample_train.py` asks the served model each of 1,000 training problems four times, at temperature 1.0, on a 4-slot llama-server. It keeps the exact prompt and reply token ids, so the trainer sees what the server saw.

```text
{"samples": 4000, "right": 3524, "problems": 1000}
```

4,000 samples, 3,707,725 generated tokens, took 4 h 1 min on the 3050: 256 tokens per second across the four slots. This is the longest step of the loop, and it is pure inference. 948 problems got at least one right answer. The shortest right answer averages 757.2 tokens against 858.3 for all right answers, so the targets are 12% shorter than the model's typical right answer.

---

#### Step 6 — Train on the Shortest Right Answer

`train.py` picks, in code, the shortest right answer per problem whose thinking finished, then trains two passes at lr 1e-4, batch 8:

```text
{"problems": 1000, "with_right": 948, "train_seqs": 931, "mean_gen_len": 723.8}
{"step": 234, "of": 234, "epoch": 1, "loss": 0.2107, "gnorm": 1.525, "lr": 0.0, "elapsed_s": 7822, "peak_mib": 4896}
```

234 steps in 2 h 10 min, 33.4 s per step, 4,896 MiB peak.

A second run, B, writes a thinking budget into the prompt ("Think for at most 256 tokens") set to each answer's own length, and pairs each problem's shortest answer with a longer one at a bigger budget. 1,411 sequences, 354 steps, 3 h 28 min, 4,899 MiB peak.

---

#### Step 7 — Export Back Into the Same File

`export_gguf.py` copies the served GGUF and overwrites only the 4-bit weights of the trained layers, under the original scales. A zero adapter writes back the identical file:

```text
levels changed 0 of 1835532288 in trained tensors -> zero.gguf
zero adapter: byte-identical
```

For run A:

```text
levels changed 238829 of 1835532288 in trained tensors -> gemma-4-E2B-it-lazy-q4_0.gguf
```

The result loads in the same llama-server with the same flags, at the same size, on any card that ran the original.

---

#### Step 8 — Measure the Tuned Models

The same 250 problems, the same server flags, greedy:

```text
run A                      {"right": 220, "accuracy": 0.88,  "mean_tokens": 816.0, "median_tokens": 707.0, "tokens_per_correct": 927.2}
run B, plain prompt        {"right": 229, "accuracy": 0.916, "mean_tokens": 867.6, "median_tokens": 780.0, "tokens_per_correct": 947.2}
run B, prompt budget 128   {"right": 221, "accuracy": 0.884, "mean_tokens": 801.7, "median_tokens": 721.0, "tokens_per_correct": 906.9}
run B, prompt budget 256   {"right": 224, "accuracy": 0.896, "mean_tokens": 809.3, "median_tokens": 740.5, "tokens_per_correct": 903.2}
run B, prompt budget 512   {"right": 225, "accuracy": 0.9,   "mean_tokens": 822.3, "median_tokens": 755.0, "tokens_per_correct": 913.7}
run A at lr 1e-3           {"right": 226, "accuracy": 0.904, "mean_tokens": 828.3, "median_tokens": 706.0, "tokens_per_correct": 916.2}
QLoRA at lr 1e-3           {"right": 228, "accuracy": 0.912, "mean_tokens": 878.4, "median_tokens": 785.5, "tokens_per_correct": 963.2}
```

Run A cuts tokens per correct answer by 5.2% and loses 7 right answers, so it fails the equal-accuracy rule. Run B's plain prompt cuts them by 3.1% with accuracy held. A budget written in the prompt moves tokens by less than 11 between 128 and 512. Run A at lr 1e-3 cuts them by 6.3% and gets 226 right, one fewer than the untrained model.

The QLoRA row is the standard recipe from Step 4: transformers' Trainer, PEFT and bitsandbytes' NF4 format, on run A's data and settings at lr 1e-3, exported by the same script. It trains in 1 h 6 min, half the time of run A, and gets 228 right and cuts tokens per correct answer by 1.5%. Run A at the same rate cuts them by 6.3%, or 4.9% fewer tokens than QLoRA: 916.2 against 963.2. Its export moved 52,531,066 of the 4-bit weights, 2.862%, against run A's 7.353%. An adapter trained on NF4 loses part of its correction when the weights are rounded into Q4_0.

🔎 **Tip: count the 4-bit weights that moved.** Run A's export changed 238,829 of 1,835,532,288 weights, 0.013%; run B's changed 0.041%. A correction smaller than half a 4-bit step rounds away, so an adapter can train for hours and leave the shipped file nearly unchanged. At lr 1e-3 the same training moved 134,965,355, or 7.35%, 565 times as many as at lr 1e-4. Tokens per correct answer fell from 927.2 to 916.2. Once the weights move, the limit is the data: the shortest of four right answers is only 11.8% shorter than a typical right answer, so the model has little shorter thinking to learn from.

---

#### Compare and Contrast

| | llama.cpp thinking cap 256 | Tuned A | Tuned A, lr 1e-3 | QLoRA, lr 1e-3 | Tuned B, plain prompt |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Right of 250 | 🥈 228 | 220 | 226 | 🥈 228 | 🥇 229 |
| Tokens per correct answer | 🥇 628.0 | 927.2 | 🥈 916.2 | 963.2 | 947.2 |
| Training needed | 🟢 none | 2 h 10 min on the 3050 | 2 h 10 min on the 3050 | 1 h 6 min on the 3050 | 3 h 28 min on the 3050 |
| Works with any client | ⚠️ needs the request field | 🟢 yes | 🟢 yes | 🟢 yes | 🟢 yes |
| Fewer tokens at equal accuracy | 🟢 yes | ❌ no, accuracy fell | ⚠️ 6.3%, one fewer right | ⚠️ 1.5%, small | ⚠️ 3.1%, small |

---

#### So, Which One?

To make Gemma 4 E2B think less today, set `thinking_budget_tokens` to 256 on the request: it costs nothing and cuts tokens per correct answer by 35.8% on these problems. To fine-tune Gemma 4 E2B cheaply, the 8 GB card that serves it is enough: sample, train and grade on the 3050 in under 7 hours. For the training step, the standard QLoRA stack is the quick route, an hour for two passes, once the per-layer table goes to the CPU and the loss is chunked. Training through the shipped 4-bit weights takes twice as long, fits in under 5 GB, and here cut more tokens per correct answer.

---

#### Prior Work

- **GSM8K prompt and grader**: copied from `gen_eval.py` in my earlier Gemma 4 TPU work (`gemma4-dev/jev-tpu-v5e1`).
- **The exact Q4_0 GGUF**: built by `gguf_exact.py` in my earlier Gemma 4 llama.cpp work (`gemma4-dev/local-llamacpp-1650ti-4b-q4_0`), published as `xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf`.
- **Open source used**: [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT), [PyTorch](https://github.com/pytorch/pytorch), [transformers](https://github.com/huggingface/transformers), [PEFT](https://github.com/huggingface/peft) and [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) (QLoRA comparison only), [gguf-py](https://github.com/ggml-org/llama.cpp/tree/master/gguf-py), Gemma 4 E2B (Gemma terms), [GSM8K](https://github.com/openai/grade-school-math).

---

## Why Does Open Innovation Matter?

Every step of this project needs the weights on the desk.

- **Training through the shipped weights needs the shipped file.** The trainer reads the 4-bit weights straight out of the GGUF that llama.cpp serves. A hosted fine-tuning API trains its own copy in its own format.
- **The trained model runs where the old one ran.** The export writes into the same open file format, so the lazy model goes back onto the 3050, or a smaller card, with nothing to install.
- **Open source shows where the gaps are.** llama.cpp's own trainer stops on Gemma 4 with `ggml_compute_backward: unsupported unary op for backward pass: TANH`, from the model's output softcap. Because the code is open, the line is findable (`src/models/gemma4.cpp:436`), and so is the workaround: train in PyTorch on the same weights.
- **No per-token bill.** 4 hours of sampling and 5 hours of training cost the electricity of one desktop.

---

## The Lazy Payoff

The payoff available today comes from llama.cpp, with no training: the 256-token thinking cap takes the 3050 from 9.06 s to 5.80 s per correct answer, and from 977.8 to 628.0 tokens.

On a slow device the tokens matter more. The Snapdragon 850 laptop from my earlier work decodes E2B at 5.00 tokens per second (measured 2026-10-11, `gemma4-dev/local-llamacpp-sdm850-2b-q4_0`). By arithmetic, 349.8 fewer tokens per correct answer is about 70 s less waiting per answer there. Not measured on that laptop in this project.

---

## Prize Categories

**Gemma**: Gemma 4 E2B, fine-tuned locally on an 8 GB RTX 3050 through its quantization-aware 4-bit weights and served with llama.cpp.

---

## Summary

The goal of this project was to show that a low-end 8 GB GPU that serves Gemma 4 E2B can also fine-tune it, by teaching the model to think less. The 3050's inference speed covers the sampling; keeping Gemma 4's large tables off the GPU makes the training fit, with either trainer. The results were:

- 🟢 One RTX 3050 runs the whole loop in 6 h 44 min: sampling at 256 tokens per second across four slots, training, and a 250-problem evaluation.
- 🟢 Training fits on the RTX 3050: 4,896 MiB peak, 33.4 s per step, 2 h 10 min for two passes over 931 examples.
- 🟢 The standard QLoRA stack fits too, at 17.1 s per step, with the per-layer table on the CPU and the loss in chunks.
- 🟢 The trained model is the served file with only its trained weights changed; a zero adapter writes back an identical file.
- 🟢 llama.cpp's thinking cap of 256 tokens cuts tokens per correct answer by 35.8% at equal accuracy, with no training.
- ⚠️ The best tuned run, lr 1e-3, cuts them by 6.3% with one fewer right answer. At lr 1e-4 only 0.013% to 0.041% of the 4-bit weights moved; at lr 1e-3, 7.35% moved and the tokens barely followed.
- ⚠️ QLoRA holds accuracy at 228 with a 1.5% cut in tokens per correct answer. Training through the shipped weights at the same rate cuts 6.3%.
- ❌ Training on the shortest right answer at lr 1e-4 cost 7 right answers out of 250.
- ❌ llama.cpp's own trainer stops on Gemma 4's output softcap.

Scope: one RTX 3050 OEM, llama.cpp `b11696-3-g7182ef71e`, one greedy run of 250 held-out GSM8K problems per model, four samples per training problem on 1,000 problems. Seconds per correct answer on the 3050 are inflated for some early baseline rows, which ran beside a CPU-side export test; token counts are unaffected. The Snapdragon 850 figure comes from another project and is not re-measured here.

The strategy for fine-tuning Gemma 4 on a low-end 8 GB GPU was validated with an incremental step by step approach.

---

#### References

- [Hacktoberfest Open-Source AI Challenge, Week 3](https://dev.to/challenges/hacktoberfest-week3-2026-10-19)
- [lazy-gemma on GitHub](https://github.com/xbill9/lazy-gemma)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [Gemma 4 E2B, exact Q4_0 GGUF used here](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf)
- [google/gemma-4-E2B-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-unquantized)
- [GSM8K](https://github.com/openai/grade-school-math)

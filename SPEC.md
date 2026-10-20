# SPEC — lazy-gemma

Handoff for the session on the 3050 desktop (hostname `gpu`, GA106 8192 MiB, i7-5820K host, per
`~/gemma4-dev/local-llamacpp-3050oem-2b-q4_0`). Nothing below has been run.

## Decisions taken (2026-10-20)

- Model: Gemma 4 E2B instruct. Hardware: the 3050. 4-bit base + LoRA.
- Two arms, A/B on the training stack: QLoRA (transformers + PEFT + bitsandbytes) and llama.cpp training.
- Arm A is QAT-aware LoRA, not QLoRA: base `google/gemma-4-E2B-it-qat-q4_0-unquantized` in bf16, forward
  through Q4_0 fake-quant of (W + LoRA delta) with a straight-through gradient, so the adapter is learned
  for the exact Q4_0 weights that ship. Reason: the deliverable is a Q4_0 GGUF for small cards, and
  NF4 training followed by Q4_0 rounding would train against the wrong weights.
- Arm B blocked at llama.cpp b11696-3-g7182ef71e: `ggml_compute_backward: unsupported unary op for
  backward pass: TANH` (Gemma 4 final-logit softcap, `src/models/gemma4.cpp:436`); `llama-finetune` is also
  full-weight only, no LoRA. Log: `run/armB-q4_0-try.log`.
- Extra baseline: E2B Q4_0 with llama.cpp's thinking budget cap (`thinking_budget_tokens`), the
  no-training way to think less. The tuned model has to beat it at equal accuracy.
- Prize category: Best Use of Gemma. Tinker is out: its models page (checked 2026-10-20) lists no Gemma.
- Python: the machine's own `python3`, no virtualenvs.
- Naming: `~/gemma4-dev/NAMING.md` defines slot 2 as the *serving* stack and says a fine-tuned rig "needs
  its own answer" (lines 366–367). A training rig needs a spec decision from the user before it gets a
  rig name. Until then this is a plain repo; fold it into gemma4-dev after the deadline.

## First checks, in order (each is a go/no-go)

1. `nvidia-smi`; `python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"`;
   versions of transformers, peft, bitsandbytes, accelerate, datasets. Is the GPU free (stop llama-server)?
2. Arm A smoke test: load `google/gemma-4-E2B-it` with `BitsAndBytesConfig(load_in_4bit=True,
   bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)`. Record peak VRAM and which modules
   bitsandbytes refuses or skips. The per-layer embedding table is the open question: frozen, and
   whether it stays on CPU.
3. One LoRA step on 8 examples at 1k tokens with gradient checkpointing. Record peak VRAM and s/step.
4. Arm B: find llama.cpp's current training entry point at the rig's commit and whether it accepts the
   Gemma 4 graph and a Q4_0 base. Record the exact error if not.
5. Baseline eval harness: reuse the 12B rig's GSM8K grader if it fits; grading is code, never a model.

## Check results (2026-10-20, host `gpu`)

1. Pass. torch 2.14.1+cu130 (CUDA 13.0, bf16), transformers 5.12.1, peft 0.21.2, bitsandbytes 0.50.2,
   trl 1.16.0, datasets 5.1.0, pip `--user --break-system-packages` into `/usr/bin/python3` 3.14.8.
   Debian's torchvision segfaults under the newer torch; `torchvision==0.29.1` from the cu130 index fixes it.
2. Pass (QAT arm, `qat_lora.py`). Every Q4_0 tensor is read from the served GGUF as int8 levels and fp16
   steps; with no LoRA the fake-quant equals the GGUF dequantization in every value (5 tensors checked,
   0 mismatches). Top-1 agreement with llama-server greedy tokens: 510 of 512. Load peak 2,434 MiB,
   per-layer table (2.24 GiB int8) on the CPU.
3. Pass. 8 × 1,024 tokens, rank 16 on q/k/v/o/gate/up/down (24.2M LoRA values), gradient checkpointing,
   one sequence at a time: 35.7 s per step, peak 4,725 MiB allocated, 5,120 MiB reserved. Overfitting one
   reply moves loss 0.18 → 0.02 in 12 steps and changes 24.4M of 1.84B levels, so gradients reach the grid.
4. Blocked, see decisions above.
5. Pass. `lazy_eval.py` copies the prompt and grader of `jev-tpu-v5e1/gen_eval.py`. Note the 12B's
   GSM8K 0.960 was measured with `--reasoning off`.

## Data rules

- Train and held-out sets are disjoint, fixed by seed, and written to disk before any training.
- The shortest *correct* answer is chosen by code. Record per problem: candidates, how many correct,
  chosen length.
- Report tokens with the model's tokenizer, counted by code.

## What gets reported

Baseline vs arm A vs arm B, same held-out problems, same sampling: accuracy, mean and median output
tokens, tokens per correct answer, and seconds per correct answer on the Snapdragon 850 rig.

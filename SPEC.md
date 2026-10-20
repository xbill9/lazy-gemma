# SPEC — lazy-gemma

Handoff for the session on the 3050 desktop (hostname `gpu`, GA106 8192 MiB, i7-5820K host, per
`~/gemma4-dev/local-llamacpp-3050oem-2b-q4_0`). Nothing below has been run.

## Decisions taken (2026-10-20)

- Model: Gemma 4 E2B instruct. Hardware: the 3050. 4-bit base + LoRA.
- Two arms, A/B on the training stack: QLoRA (transformers + PEFT + bitsandbytes) and llama.cpp training.
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

## Data rules

- Train and held-out sets are disjoint, fixed by seed, and written to disk before any training.
- The shortest *correct* answer is chosen by code. Record per problem: candidates, how many correct,
  chosen length.
- Report tokens with the model's tokenizer, counted by code.

## What gets reported

Baseline vs arm A vs arm B, same held-out problems, same sampling: accuracy, mean and median output
tokens, tokens per correct answer, and seconds per correct answer on the Snapdragon 850 rig.

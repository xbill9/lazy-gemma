# lazy-gemma: teaching Gemma 4 E2B to think less

Entry for the [Hacktoberfest Open-Source AI Challenge, Week 3: Slow and Steady](https://dev.to/challenges/hacktoberfest-week3-2026-10-19).
Started 2026-10-20. Deadline 2026-10-25, 11:59 PM PDT; any commit after it gets noted here.

**Status: nothing trained, nothing measured.** This file states the plan; every figure below is a
measurement from another project, cited to it.

## The idea

Gemma 4 thinks before it answers. On a fast GPU that costs little; on a slow port it is most of the wait.
The Snapdragon 850 rig decodes E2B at 5.00 tok/s and prefills at 9.22 tok/s
(`~/gemma4-dev/local-llamacpp-sdm850-2b-q4_0`, measured 2026-10-11), so every 500 thinking tokens is about
100 seconds before the answer starts.

This project fine-tunes E2B to reach the same answer in fewer tokens, on an 8 GB RTX 3050, with a 4-bit
base and a LoRA, two ways:

| Arm | Stack | Base it trains on |
|---|---|---|
| A | transformers + PEFT + bitsandbytes (QLoRA) | E2B loaded 4-bit NF4 |
| B | llama.cpp's own training code | `xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf`, the file the 3050 rig already serves |

Arm A is the result by Sunday. Arm B is the port: training moved onto the llama.cpp rig itself, with no
conversion step. Neither has been tried with Gemma 4.

## Method

1. **Baseline.** E2B Q4_0 on the 3050, thinking on: accuracy and tokens per answer on a held-out set.
2. **Data.** Sample several answers per training problem (E2B, and the 12B rig: GSM8K 0.960 measured
   2026-10-16 in `~/gemma4-dev/local-llamacpp-3050oem-12b-q4_0`). Code grades every answer and keeps the
   shortest correct one per problem. No model judges correctness.
3. **Train** a LoRA on those answers, arm A and arm B.
4. **Measure** the tuned model the same way as the baseline, same problems, same sampling. The claim is
   tokens per correct answer at equal accuracy, and it is false if accuracy drops.
5. **The lazy payoff.** Convert to Q4_0 GGUF and time baseline vs tuned on the Snapdragon 850: seconds
   per correct answer.

## Layout

| Path | What |
|---|---|
| `RULES-week3.md`, `TEMPLATE-week3.md` | Challenge rules and submission template, saved for offline use |
| `SPEC.md` | The working spec and the first checks to run on the 3050 |

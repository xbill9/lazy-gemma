# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Hacktoberfest Week 3 entry: fine-tune Gemma 4 E2B to reach the same answer in fewer thinking tokens.
`SPEC.md` is the working plan and its first checks are go/no-go gates, run in order. `RULES-week3.md` is the challenge rules.

## Deadline

2026-10-25, 11:59 PM PDT. Any commit after it must be noted in `README.md`, or the entry can be disqualified.
The project must stay new work: credit any non-trivial reused code (e.g. the 12B rig's GSM8K grader) in the README and the post.

## Hardware

- Sessions run on host `gpu`: RTX 3050 OEM, 8192 MiB. Training needs the whole card.
- A `llama-server` from `~/llama.cpp/build/bin/` is often holding the GPU (`pgrep -a llama-server`). You may stop it without asking. Note its command line first so it can be restarted.
- Training uses the machine's `python3` (3.14). No virtualenvs.

## Measurement rules

- Correctness is graded by code, never by a model. The shortest *correct* candidate per problem is chosen by code.
- Count tokens with the model's tokenizer, in code. Never estimate them.
- Train and held-out sets are disjoint, fixed by seed, and written to disk before any training.
- The claim is fewer tokens per correct answer at equal accuracy. If accuracy drops, the claim is false; report it that way.
- Baseline and tuned models run on the same held-out problems with the same sampling.
- README figures from other rigs cite their source directory and date. Anything not measured here is marked as unmeasured.

## Gemma 4 + llama-server

Gemma 4 thinks first: on `/v1/chat/completions` the thinking goes to `reasoning_content` and `content` stays empty until the thinking block closes. A small `max_tokens` returns `content: ""` with `finish_reason: "length"` on a healthy server. Use `/v1/chat/completions` (raw `/v1/completions` returns empty on `-it` checkpoints) and count reasoning tokens as output.

## Layout

Follow the sibling rig `~/gemma4-dev/local-llamacpp-3050oem-12b-q4_0`: Python scripts at the root, a `Makefile` with `test` (`python3 -m unittest discover -s tests -v`) and `lint` (`ruff check` on an explicit file list, so add each new module to it), and benchmark runs under `benchmarks/runs/<date>-<name>/` with a `REPORT.md`.

Don't rename this into a `<platform>-<runtime>-<hardware>-<model>` rig directory. `~/gemma4-dev/NAMING.md` has no slot value for a training rig yet, and that needs the user's decision.

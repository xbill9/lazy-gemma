#!/bin/bash
# Standard QLoRA (transformers Trainer + PEFT + bitsandbytes NF4) on the same data and settings as
# 2026-10-21-qatlora-lr1e-3-e2b-3050oem, exported into the served GGUF's Q4_0 grid by the same
# export_gguf.py, and the baseline's exact eval (250 held-out, greedy, thinking on, max_tokens 4096).
# A two-step smoke run first, so a load or memory failure stops the run in minutes.
set -u
cd "$(dirname "$0")"/../../..
R=benchmarks/runs/2026-10-21-qlora-nf4-e2b-3050oem
S=benchmarks/runs/2026-10-20-samples-e2b-3050oem/samples.jsonl
OUT=$HOME/models/lazy-gemma/gemma-4-E2B-it-lazy-qlora-nf4-q4_0.gguf
mkdir -p "$(dirname $OUT)"
pkill -x llama-server; while pgrep -x llama-server >/dev/null; do sleep 1; done
ARGS="--epochs 2 --lr 1e-3 --rank 16 --alpha 32 --batch 8 --max-len 2048"
python3 qlora_train.py $S $R/smoke $ARGS --max-steps 2 > $R/smoke.out 2>&1 || exit 1
python3 qlora_train.py $S $R $ARGS > $R/train.out 2>&1 || exit 1
python3 export_gguf.py $R/lora.pt $OUT > $R/export.out 2>&1 || exit 1
sha256sum $OUT > $R/gguf.sha256
setsid $HOME/llama.cpp/build/bin/llama-server -m $OUT --host 127.0.0.1 --port 8080 -ngl 99 -c 8192 --parallel 1 -fa 1 -t 6 --metrics > $R/server.log 2>&1 < /dev/null &
for i in $(seq 120); do curl -fsS localhost:8080/health >/dev/null 2>&1 && break; sleep 1; done
python3 lazy_eval.py run http://127.0.0.1:8080 data/heldout.jsonl $R/think-greedy.jsonl --max-tokens 4096 > $R/think-greedy.summary.json
cat $R/think-greedy.summary.json
pkill -x llama-server

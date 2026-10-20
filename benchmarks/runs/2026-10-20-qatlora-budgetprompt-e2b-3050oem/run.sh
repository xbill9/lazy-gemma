#!/bin/bash
# QAT-LoRA, budget in the prompt: same samples as 2026-10-20-qatlora-e2b-3050oem, prompts state
# "Think for at most N tokens" with N the bucket of each sample's own thinking length (train.py
# --prompt-budget). Eval: the baseline's 250 held-out at prompt budgets 128/256/512 and the plain prompt.
set -u
cd "$(dirname "$0")"/../../..
R=benchmarks/runs/2026-10-20-qatlora-budgetprompt-e2b-3050oem
S=benchmarks/runs/2026-10-20-samples-e2b-3050oem/samples.jsonl
OUT=$HOME/models/lazy-gemma/gemma-4-E2B-it-lazy-budgetprompt-q4_0.gguf
mkdir -p "$(dirname $OUT)"
pkill -x llama-server; while pgrep -x llama-server >/dev/null; do sleep 1; done
python3 train.py $S $R --prompt-budget --epochs 2 --lr 1e-4 --rank 16 --alpha 32 --batch 8 --max-len 2048 > $R/train.out 2>&1 || exit 1
python3 export_gguf.py $R/lora.pt $OUT > $R/export.out 2>&1 || exit 1
sha256sum $OUT > $R/gguf.sha256
setsid $HOME/llama.cpp/build/bin/llama-server -m $OUT --host 127.0.0.1 --port 8080 -ngl 99 -c 8192 --parallel 1 -fa 1 -t 6 --metrics > $R/server.log 2>&1 < /dev/null &
for i in $(seq 120); do curl -fsS localhost:8080/health >/dev/null 2>&1 && break; sleep 1; done
for b in 128 256 512; do
  python3 lazy_eval.py run http://127.0.0.1:8080 data/heldout.jsonl $R/prompt-$b.jsonl --max-tokens 4096 --prompt-budget $b > $R/prompt-$b.summary.json
  echo "$(date '+%F %T') prompt budget $b $(cat $R/prompt-$b.summary.json)"
done
python3 lazy_eval.py run http://127.0.0.1:8080 data/heldout.jsonl $R/plain.jsonl --max-tokens 4096 > $R/plain.summary.json
echo "$(date '+%F %T') plain $(cat $R/plain.summary.json)"
pkill -x llama-server

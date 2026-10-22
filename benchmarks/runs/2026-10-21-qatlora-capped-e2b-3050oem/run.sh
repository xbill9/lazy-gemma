#!/bin/bash
# Two runs, one after the other, on the 3050.
# 1. 2026-10-21-qatlora-lr1e-3-budget256-e2b-3050oem: the lr 1e-3 tuned GGUF with llama.cpp's 256-token
#    thinking cap on top, the budget baseline's exact eval (250 held-out, greedy, thinking on, max_tokens 4096).
# 2. Capped training data: 4 samples on the same 1,000 training problems (seed 0, T=1.0, top_p 0.95, top_k 64,
#    4-slot server), thinking capped at 256 tokens; shortest right per problem; QAT-LoRA with the lr 1e-3
#    run's settings; export; eval uncapped and with the 256 cap.
set -u
cd "$(dirname "$0")"/../../..
R1=benchmarks/runs/2026-10-21-qatlora-lr1e-3-budget256-e2b-3050oem
S=benchmarks/runs/2026-10-21-samples-budget256-e2b-3050oem
R2=benchmarks/runs/2026-10-21-qatlora-capped-e2b-3050oem
BIN=$HOME/llama.cpp/build/bin/llama-server
BASE=$HOME/models/gemma-4-E2B-it-qat-q4_0-exact-v2/gemma-4-E2B-it-q4_0-exact.gguf
LR3=$HOME/models/lazy-gemma/gemma-4-E2B-it-lazy-lr1e-3-q4_0.gguf
OUT=$HOME/models/lazy-gemma/gemma-4-E2B-it-lazy-capped-q4_0.gguf
stop() { pkill -x llama-server; while pgrep -x llama-server >/dev/null; do sleep 1; done; }
serve() { local m=$1 log=$2; shift 2; stop
  setsid $BIN -m $m --host 127.0.0.1 --port 8080 -ngl 99 "$@" -fa 1 -t 6 --metrics > $log 2>&1 < /dev/null &
  for i in $(seq 120); do curl -fsS localhost:8080/health >/dev/null 2>&1 && return 0; sleep 1; done; return 1; }
log() { echo "$(date '+%F %T') $*"; }

sha256sum $LR3 > $R1/gguf.sha256
serve $LR3 $R1/server.log -c 8192 --parallel 1 || exit 1
python3 lazy_eval.py run http://127.0.0.1:8080 data/heldout.jsonl $R1/budget-256.jsonl --max-tokens 4096 --budget 256 > $R1/budget-256.summary.json
log "run 1 budget 256 $(cat $R1/budget-256.summary.json)"

serve $BASE $S/server.log -c 24576 --parallel 4 || exit 1
python3 sample_train.py http://127.0.0.1:8080 $S/samples.jsonl --problems 1000 --samples 4 --concurrency 4 --budget 256 > $S/summary.json || exit 1
log "samples $(cat $S/summary.json)"
stop

python3 train.py $S/samples.jsonl $R2 --epochs 2 --lr 1e-3 --rank 16 --alpha 32 --batch 8 --max-len 2048 > $R2/train.out 2>&1 || exit 1
log "trained $(head -1 $R2/train.out)"
python3 export_gguf.py $R2/lora.pt $OUT > $R2/export.out 2>&1 || exit 1
sha256sum $OUT > $R2/gguf.sha256
serve $OUT $R2/server.log -c 8192 --parallel 1 || exit 1
python3 lazy_eval.py run http://127.0.0.1:8080 data/heldout.jsonl $R2/think-greedy.jsonl --max-tokens 4096 > $R2/think-greedy.summary.json
log "run 2 uncapped $(cat $R2/think-greedy.summary.json)"
python3 lazy_eval.py run http://127.0.0.1:8080 data/heldout.jsonl $R2/budget-256.jsonl --max-tokens 4096 --budget 256 > $R2/budget-256.summary.json
log "run 2 budget 256 $(cat $R2/budget-256.summary.json)"
stop

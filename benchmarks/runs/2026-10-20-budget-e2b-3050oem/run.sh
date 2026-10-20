#!/bin/bash
# Capped-thinking baselines: E2B Q4_0, greedy, thinking on, llama.cpp thinking_budget_tokens 128/256/512,
# the same 250 held-out problems and server as 2026-10-20-baseline-e2b-3050oem. Then 4 samples on each of
# 1,000 training problems (T=1.0, top_p 0.95, top_k 64) on a 4-slot server, for train.py.
set -u
cd "$(dirname "$0")"/../../..; ROOT=$PWD
B=benchmarks/runs/2026-10-20-budget-e2b-3050oem; S=benchmarks/runs/2026-10-20-samples-e2b-3050oem
BIN=$HOME/llama.cpp/build/bin/llama-server
G=$HOME/models/gemma-4-E2B-it-qat-q4_0-exact-v2/gemma-4-E2B-it-q4_0-exact.gguf
serve() { pkill -x llama-server; while pgrep -x llama-server >/dev/null; do sleep 1; done
  setsid $BIN -m $G --host 127.0.0.1 --port 8080 -ngl 99 "$@" -fa 1 -t 6 --metrics > "$LOG" 2>&1 < /dev/null &
  for i in $(seq 120); do curl -fsS localhost:8080/health >/dev/null 2>&1 && return 0; sleep 1; done; return 1; }
LOG=$B/server.log serve -c 8192 --parallel 1
for b in 128 256 512; do
  python3 lazy_eval.py run http://127.0.0.1:8080 data/heldout.jsonl $B/budget-$b.jsonl --max-tokens 4096 --budget $b > $B/budget-$b.summary.json
  echo "$(date '+%F %T') budget $b $(cat $B/budget-$b.summary.json)"
done
LOG=$S/server.log serve -c 24576 --parallel 4
python3 sample_train.py http://127.0.0.1:8080 $S/samples.jsonl --problems 1000 --samples 4 --concurrency 4 > $S/summary.json
echo "$(date '+%F %T') samples $(cat $S/summary.json)"
pkill -x llama-server

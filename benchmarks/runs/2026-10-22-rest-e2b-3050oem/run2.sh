#!/bin/bash
# Follows run.sh (waits for it to exit). Same eval settings.
# 1. Base GGUF uncapped on the 1,069, so the uncapped baseline covers all 1,319 with the 250-set baseline run.
# 2. The capped-samples recipe at cap 64: sample, base with the 64 cap on the 250 and the 1,069, train,
#    export, the trained model uncapped on the 250 and the 1,069.
set -u
cd "$(dirname "$0")"/../../..
E=benchmarks/runs/2026-10-22-rest-e2b-3050oem
while pgrep -f "$E/run.sh" >/dev/null; do sleep 60; done
BIN=$HOME/llama.cpp/build/bin/llama-server
BASE=$HOME/models/gemma-4-E2B-it-qat-q4_0-exact-v2/gemma-4-E2B-it-q4_0-exact.gguf
stop() { pkill -x llama-server; while pgrep -x llama-server >/dev/null; do sleep 1; done; }
serve() { local m=$1 log=$2; shift 2; stop
  setsid $BIN -m $m --host 127.0.0.1 --port 8080 -ngl 99 "$@" -fa 1 -t 6 --metrics > $log 2>&1 < /dev/null &
  for i in $(seq 120); do curl -fsS localhost:8080/health >/dev/null 2>&1 && return 0; sleep 1; done; return 1; }
log() { echo "$(date '+%F %T') $*"; }
ev() { local data=$1 out=$2; shift 2
  python3 lazy_eval.py run http://127.0.0.1:8080 $data $out.jsonl --max-tokens 4096 "$@" > $out.summary.json
  log "$out $(cat $out.summary.json)"; }
REST=data/heldout-rest.jsonl

serve $BASE $E/server-base.log -c 8192 --parallel 1 || exit 1
ev $REST $E/base-think

B=64
S=benchmarks/runs/2026-10-23-samples-budget$B-e2b-3050oem
R=benchmarks/runs/2026-10-23-qatlora-capped$B-e2b-3050oem
OUT=$HOME/models/lazy-gemma/gemma-4-E2B-it-lazy-capped$B-q4_0.gguf
serve $BASE $S/server.log -c 24576 --parallel 4 || exit 1
python3 sample_train.py http://127.0.0.1:8080 $S/samples.jsonl --problems 1000 --samples 4 --concurrency 4 --budget $B > $S/summary.json || exit 1
log "samples $B $(cat $S/summary.json)"
serve $BASE $E/server-base.log -c 8192 --parallel 1 || exit 1
ev data/heldout.jsonl $R/base-budget-$B --budget $B
ev $REST $E/base-budget-$B --budget $B
stop
python3 train.py $S/samples.jsonl $R --epochs 2 --lr 1e-3 --rank 16 --alpha 32 --batch 8 --max-len 2048 > $R/train.out 2>&1 || exit 1
log "trained $B $(head -1 $R/train.out)"
python3 export_gguf.py $R/lora.pt $OUT > $R/export.out 2>&1 || exit 1
sha256sum $OUT > $R/gguf.sha256
serve $OUT $R/server.log -c 8192 --parallel 1 || exit 1
ev data/heldout.jsonl $R/think-greedy
ev $REST $E/capped$B-think
stop
log done2

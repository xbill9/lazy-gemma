#!/bin/bash
# The 1,069 GSM8K test problems the 250 held-out set did not draw (data/heldout-rest.jsonl, make_rest.py),
# with the 250-set eval settings: greedy, thinking on, max_tokens 4096, one request at a time.
# 1. Base GGUF with the 256 cap, and the 256-capped-samples model uncapped, on the 1,069.
# 2. The capped-samples recipe at caps 128 and 512: 4 samples on the same 1,000 training problems (seed 0),
#    shortest right per problem, QAT-LoRA lr 1e-3 for 2 epochs, export; each model uncapped on the 250 and
#    the 1,069, and the base GGUF with the same cap on the 1,069.
set -u
cd "$(dirname "$0")"/../../..
E=benchmarks/runs/2026-10-22-rest-e2b-3050oem
BIN=$HOME/llama.cpp/build/bin/llama-server
BASE=$HOME/models/gemma-4-E2B-it-qat-q4_0-exact-v2/gemma-4-E2B-it-q4_0-exact.gguf
C256=$HOME/models/lazy-gemma/gemma-4-E2B-it-lazy-capped-q4_0.gguf
stop() { pkill -x llama-server; while pgrep -x llama-server >/dev/null; do sleep 1; done; }
serve() { local m=$1 log=$2; shift 2; stop
  setsid $BIN -m $m --host 127.0.0.1 --port 8080 -ngl 99 "$@" -fa 1 -t 6 --metrics > $log 2>&1 < /dev/null &
  for i in $(seq 120); do curl -fsS localhost:8080/health >/dev/null 2>&1 && return 0; sleep 1; done; return 1; }
log() { echo "$(date '+%F %T') $*"; }
ev() { local data=$1 out=$2; shift 2
  python3 lazy_eval.py run http://127.0.0.1:8080 $data $out.jsonl --max-tokens 4096 "$@" > $out.summary.json
  log "$out $(cat $out.summary.json)"; }
REST=data/heldout-rest.jsonl

sha256sum data/heldout-rest.jsonl $BASE $C256 > $E/inputs.sha256
serve $BASE $E/server-base.log -c 8192 --parallel 1 || exit 1
ev $REST $E/base-budget-256 --budget 256
serve $C256 $E/server-capped256.log -c 8192 --parallel 1 || exit 1
ev $REST $E/capped256-think

for B in 128 512; do
  S=benchmarks/runs/2026-10-22-samples-budget$B-e2b-3050oem
  R=benchmarks/runs/2026-10-22-qatlora-capped$B-e2b-3050oem
  OUT=$HOME/models/lazy-gemma/gemma-4-E2B-it-lazy-capped$B-q4_0.gguf
  serve $BASE $S/server.log -c 24576 --parallel 4 || exit 1
  python3 sample_train.py http://127.0.0.1:8080 $S/samples.jsonl --problems 1000 --samples 4 --concurrency 4 --budget $B > $S/summary.json || exit 1
  log "samples $B $(cat $S/summary.json)"
  serve $BASE $E/server-base.log -c 8192 --parallel 1 || exit 1
  ev $REST $E/base-budget-$B --budget $B
  stop
  python3 train.py $S/samples.jsonl $R --epochs 2 --lr 1e-3 --rank 16 --alpha 32 --batch 8 --max-len 2048 > $R/train.out 2>&1 || exit 1
  log "trained $B $(head -1 $R/train.out)"
  python3 export_gguf.py $R/lora.pt $OUT > $R/export.out 2>&1 || exit 1
  sha256sum $OUT > $R/gguf.sha256
  serve $OUT $R/server.log -c 8192 --parallel 1 || exit 1
  ev data/heldout.jsonl $R/think-greedy
  ev $REST $E/capped$B-think
done
stop
log done

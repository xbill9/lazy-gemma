#!/bin/bash
# Training-seed repeat: the 64 and 128 capped-samples models retrained with --seed 1 (LoRA init and data order)
# on the same samples and settings, exported, and run uncapped on the 250 and the 1,069.
set -u
cd "$(dirname "$0")"/../../..
E=benchmarks/runs/2026-10-22-rest-e2b-3050oem
BIN=$HOME/llama.cpp/build/bin/llama-server
stop() { pkill -x llama-server; while pgrep -x llama-server >/dev/null; do sleep 1; done; }
serve() { local m=$1 log=$2; shift 2; stop
  setsid $BIN -m $m --host 127.0.0.1 --port 8080 -ngl 99 "$@" -fa 1 -t 6 --metrics > $log 2>&1 < /dev/null &
  for i in $(seq 120); do curl -fsS localhost:8080/health >/dev/null 2>&1 && return 0; sleep 1; done; return 1; }
log() { echo "$(date '+%F %T') $*"; }
ev() { local data=$1 out=$2; shift 2
  python3 lazy_eval.py run http://127.0.0.1:8080 $data $out.jsonl --max-tokens 4096 "$@" > $out.summary.json
  log "$out $(cat $out.summary.json)"; }
declare -A SAMPLES=([64]=benchmarks/runs/2026-10-23-samples-budget64-e2b-3050oem
                    [128]=benchmarks/runs/2026-10-22-samples-budget128-e2b-3050oem)
stop
for B in 64 128; do
  R=benchmarks/runs/2026-10-23-qatlora-capped$B-seed1-e2b-3050oem
  OUT=$HOME/models/lazy-gemma/gemma-4-E2B-it-lazy-capped$B-seed1-q4_0.gguf
  python3 train.py ${SAMPLES[$B]}/samples.jsonl $R --epochs 2 --lr 1e-3 --rank 16 --alpha 32 --batch 8 --max-len 2048 --seed 1 > $R/train.out 2>&1 || exit 1
  log "trained $B seed 1 $(head -1 $R/train.out)"
  python3 export_gguf.py $R/lora.pt $OUT > $R/export.out 2>&1 || exit 1
  sha256sum $OUT > $R/gguf.sha256
  serve $OUT $R/server.log -c 8192 --parallel 1 || exit 1
  ev data/heldout.jsonl $R/think-greedy
  ev data/heldout-rest.jsonl $R/rest-think
  stop
done
log done3

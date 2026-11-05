#!/usr/bin/env bash
# One-command replay for tracing (drrun needs a single ELF command): starts the
# recorded-response server and the SSE adapter, runs Open Interpreter on the
# task list, stops the servers. Usage: replay_oi_trace.sh <task_ids> <rec_dir> <out_dir>
set -u
export USER=${USER:-agentcpu} LOGNAME=${LOGNAME:-agentcpu}   # Open Interpreter calls getpass.getuser(); trace containers run as an unmapped uid
cd "$(dirname "$0")"; T=$1; REC=$2; OUT=$3; mkdir -p "$OUT"
REF=${REF_HASH:-$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))[\"content_hash_recording\"])" "$REC/../../meta.json" 2>/dev/null)}
wait_port() { for _ in $(seq 1 300); do python3 -c "import socket,sys; s=socket.socket(); s.settimeout(0.2); sys.exit(s.connect_ex(('127.0.0.1', int(sys.argv[1]))))" "$1" && return 0; sleep 0.2; done; echo "port $1 not ready"; return 1; }
hash_check() { H=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))[\"content_hash\"])" "$1/summary.json" 2>/dev/null); if [ -n "$REF" ]; then [ "$H" = "$REF" ] && echo "REPLAY_HASH_MATCH $H" || echo "REPLAY_HASH_MISMATCH replay=$H recording=$REF"; else echo "REPLAY_HASH $H (no reference)"; fi; }
RP=$((11500 + RANDOM % 200)); PP=$((RP+1000))
python3 replay_server.py $RP "$REC" > "$OUT/replay.log" 2>&1 & RS=$!
python3 oi_llm_proxy.py $PP http://127.0.0.1:$RP/v1 > "$OUT/proxy.log" 2>&1 & PX=$!; wait_port $RP; wait_port $PP
export AGENTCPU_RECORDING_ROOT=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get(\"recording_root\",\"\"))" "$REC/../../meta.json" 2>/dev/null)   # run_oi.py hashes paths under it as under this bundle
PYTHONHASHSEED=0 venv/bin/python run_oi.py "$T" http://127.0.0.1:$PP/v1 "$OUT/run" > "$OUT/run.log" 2> "$OUT/run.err"; RC=$?
kill $PX $RS 2>/dev/null; echo "RC=$RC MISSES=$(grep -c MISS "$OUT/replay.log")"; hash_check "$OUT/run"; exit $RC

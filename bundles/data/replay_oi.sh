#!/usr/bin/env bash
# Replay a recorded Open Interpreter session with no network: replay_server
# (recorded iterations in order) -> SSE proxy -> Open Interpreter -> its
# Jupyter kernel. Usage: ./replay_oi.sh <task_ids> <rec_dir(iter_*.json)> <out_dir>
set -uo pipefail
wait_port() { for _ in $(seq 1 300); do python3 -c "import socket,sys; s=socket.socket(); s.settimeout(0.2); sys.exit(s.connect_ex(('127.0.0.1', int(sys.argv[1]))))" "$1" && return 0; sleep 0.2; done; echo "port $1 not ready"; return 1; }
cd "$(dirname "$0")"; T=$1; REC=$(realpath "$2"); OUT=$(realpath -m "$3"); mkdir -p "$OUT"
RP=$((11500 + RANDOM % 200)); PP=$((RP+1000))
python3 replay_server.py $RP "$REC" > "$OUT/replay.log" 2>&1 & RS=$!
python3 oi_llm_proxy.py $PP http://127.0.0.1:$RP/v1 > "$OUT/proxy.log" 2>&1 & PX=$!; wait_port $RP; wait_port $PP
export AGENTCPU_RECORDING_ROOT=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get(\"recording_root\",\"\"))" "$REC/../../meta.json" 2>/dev/null)   # run_oi.py hashes paths under it as under this bundle
PYTHONHASHSEED=0 timeout 1800 ${OI_PREFIX:-} venv/bin/python run_oi.py "$T" http://127.0.0.1:$PP/v1 "$OUT/run" > "$OUT/run.log" 2> "$OUT/run.err"; echo "RC=$?"
kill $PX $RS 2>/dev/null
echo "MISSES=$(grep -c MISS "$OUT/replay.log")"; tail -1 "$OUT/run.log"

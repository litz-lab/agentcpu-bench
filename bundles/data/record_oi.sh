#!/usr/bin/env bash
# Record one Open Interpreter session on a DABStep task with Opus 5 through the
# agentcpu recorder (claude CLI backend, one short-lived recorder per run).
# Usage: RECORD_MODEL=claude-opus-5 ./record_oi.sh <task_id> <out_dir>
set -uo pipefail
cd "$(dirname "$0")"; T=$1; OUT=$(realpath -m "$2"); mkdir -p "$OUT"
export RECORD_BACKEND=claude-cli RECORD_MODEL=${RECORD_MODEL:-claude-opus-5} SAMPLES=1 PORT=$((10500 + RANDOM % 200))
export RECORD_KEY=${RECORD_KEY_OVERRIDE:-dabstep_$(echo $T | tr , _)_${OI_PROMPT_VARIANT:-files}} RECORD_ROOT=$OUT/recorder PYTHONHASHSEED=0
PP=$((PORT+1000))
(cd ../.. && python3 -m agentcpu.llm.recorder > "$OUT/recorder.log" 2>&1) & REC=$!
python3 oi_llm_proxy.py $PP http://127.0.0.1:$PORT/v1 > "$OUT/proxy.log" 2>&1 & PX=$!; sleep 2
timeout ${RECORD_TIMEOUT:-1800} venv/bin/python run_oi.py "$T" http://127.0.0.1:$PP/v1 "$OUT/run" > "$OUT/run.log" 2> "$OUT/run.err"; echo "RC=$?"
kill $PX $REC 2>/dev/null
tail -1 "$OUT/run.log"; grep -c "proxy req" "$OUT/proxy.log"; ls "$OUT/recorder/$RECORD_KEY" 2>/dev/null | wc -l

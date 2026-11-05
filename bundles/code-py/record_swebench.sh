#!/usr/bin/env bash
# Record a new code-py session: SWE-agent with a real model through the agentcpu
# recorder (short-lived, one per run; never a long-lived shim).
# Usage: RECORD_MODEL=claude-opus-5 ./record_swebench.sh <instance.json> <out_dir>
set -euo pipefail
cd "$(dirname "$0")"
I=$(realpath "$1"); OUT=$(realpath -m "$2"); mkdir -p "$OUT"
IID=$(python3 -c "import json;print(json.load(open('$I'))['instance_id'])")
BASE=$(python3 -c "import json;print(json.load(open('$I'))['base_commit'])")
IMG=docker.io/swebench/sweb.eval.x86_64.$(echo "$IID" | sed 's/__/_1776_/' | tr 'A-Z' 'a-z'):latest
python3 -c "import json;open('$OUT/issue.txt','w').write(json.load(open('$I'))['problem_statement'])"
export RECORD_BACKEND=claude-cli RECORD_MODEL=${RECORD_MODEL:-claude-opus-5} SAMPLES=1 PORT=$((10200 + RANDOM % 200))
export RECORD_KEY=$IID RECORD_ROOT=$OUT/recorder PYTHONHASHSEED=0
(cd ../.. && python3 -m agentcpu.llm.recorder > "$OUT/recorder.log" 2>&1) &
REC=$!; sleep 2
venv/bin/sweagent run --config SWE-agent/config/sweagent_0_7/07.yaml \
  --agent.model.name openai/$RECORD_MODEL --agent.model.api_base http://127.0.0.1:$PORT/v1 --agent.model.api_key x \
  --agent.model.per_instance_cost_limit 0 --agent.model.total_cost_limit 0 --agent.model.per_instance_call_limit 40 \
  --env.deployment.type docker --env.deployment.image "$IMG" --env.deployment.python_standalone_dir /root \
  --env.repo.type preexisting --env.repo.repo_name testbed --env.repo.base_commit "$BASE" \
  --problem_statement.type text_file --problem_statement.path "$OUT/issue.txt" --problem_statement.id "$IID" \
  --output_dir "$OUT/run" || true
kill $REC 2>/dev/null || true
F=$(ls "$OUT"/run/*/*.traj | head -1)
# Self-contained bundle: embed the problem statement (the run referenced a file path).
python3 - "$F" <<'PY'
import json, sys
p = sys.argv[1]; t = json.load(open(p)); rc = json.loads(t["replay_config"])
ps = rc["problem_statement"]
if ps.get("type") == "text_file":
    rc["problem_statement"] = {"type": "text", "text": open(ps["path"]).read(), "id": ps.get("id")}
    t["replay_config"] = json.dumps(rc); json.dump(t, open(p, "w"))
PY
./content_hash.py "$F"

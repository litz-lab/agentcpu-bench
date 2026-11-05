#!/usr/bin/env bash
# One entry point over the five scenarios: replay the recorded session of a
# scenario and print one verdict line, comparing the replay with the recording
# (misses, content or tool hash, patch or task verdict). Exit status 0 only
# when everything matches.
#
#   ./replay.sh api          LangChain runner on two AppWorld tasks from 287e338_1
#   ./replay.sh code-py      SWE-agent on SWE-bench psf__requests-5414
#   ./replay.sh data         Open Interpreter on seven DABStep tasks (hard7)
#   ./replay.sh code-rust    Codex CLI on SWE-bench Multilingual sharkdp__bat-2393
#   ./replay.sh code-node    Claude Code on SWE-bench psf__requests-5414
#   ./replay.sh all          all five, in this order
#
# Each scenario needs its environment first (see bundles/README.md):
# bundles/{api,code-py,data,code-rust}/setup.sh and
# agentcpu/anchors/claude_code/tasks/requests_5414/setup.sh, or run inside the
# scarab-infra agentcpu image. code-py and code-rust replay in the official
# SWE-bench images under docker. Output directories go under
# ${AGENTCPU_OUT:-/tmp/agentcpu_replay}.
set -u
cd "$(dirname "$0")"; ROOT=$PWD
OUTROOT=${AGENTCPU_OUT:-/tmp/agentcpu_replay}; mkdir -p "$OUTROOT"
SC=${1:-all}
FAIL=0
verdict() {  # scenario bundle ok|FAIL detail...
  local sc=$1 b=$2 st=$3; shift 3
  printf "%-10s %-16s %-4s %s\n" "$sc" "$b" "$st" "$*"
  [ "$st" = ok ] || FAIL=1
}

run_api() {
  local fw=langchain key=langchain_n2_287e338_1 out=$OUTROOT/api
  mkdir -p "$out"
  # outside the capture image, use the environment bundles/api/setup.sh built
  [ -f bundles/api/venv/bin/python ] && . bundles/api/env.sh
  AGENT_MOCK_REPLAY_KEY=$key AGENT_MOCK_REPLAY_ROOT=$ROOT/bundles/api AGENT_DETERMINISTIC_TRANSPORT=1 \
  PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LITELLM_LOCAL_MODEL_COST_MAP=True HF_HUB_OFFLINE=1 \
  python3 -m agentcpu "$fw" --agentic-mode --tool-backend appworld --appworld-task 287e338_1 --task-id replay \
    --seed 42 --context-size short --iterations 2 --trace-output "$out/trace.json" --summary-output "$out/summary.json" \
    > "$out/run.out" 2> "$out/run.err"
  python3 - "$fw" "$out/summary.json" <<'PY' || FAIL=1
import json, sys
fw, p = sys.argv[1:]
try: s = json.load(open(p))
except Exception as e: print(f"{'api':<10} {fw:<16} FAIL no summary ({e})"); sys.exit(1)
r = s.get("replay") or {}; tasks = s.get("tasks", [])
ok = r.get("misses") == 0 and r.get("requests") == r.get("recorded")
print(f"{'api':<10} {fw:<16} {'ok' if ok else 'FAIL':<4} requests={r.get('requests')}/{r.get('recorded')} misses={r.get('misses')} "
      f"rounds={[t.get('rounds_used') for t in tasks]} verdicts={[t.get('success') for t in tasks]}")
sys.exit(0 if ok else 1)
PY
}

run_codepy() {
  local t=requests out=$OUTROOT/code-py; rm -rf "$out"
  local log; log=$(cd bundles/code-py && ./run_$t.sh "$out" 2>&1)
  local got exp
  got=$(echo "$log" | grep -o "content_hash=[0-9a-f]*" | head -1 | cut -d= -f2)
  exp=$(echo "$log" | grep "^expected" | grep -o "content_hash=[0-9a-f]*" | cut -d= -f2)
  [ -z "$exp" ] && exp=$(cd bundles/code-py && ./content_hash.py trajectories/${t}_*/*.traj | grep -o "content_hash=[0-9a-f]*" | cut -d= -f2)
  if [ -n "$got" ] && [ "$got" = "$exp" ]; then verdict code-py "$t" ok "content_hash=$got (as recorded) patch=$out/produced.patch"
  else verdict code-py "$t" FAIL "content_hash=${got:-none} expected=$exp (log: $out)"; echo "$log" | tail -5 > "$out.err"; fi
}

run_data() {
  local set=hard7 out=$OUTROOT/data; rm -rf "$out"
  local rec=bundles/data/rec/$set ids iters
  ids=$(python3 -c "import json;print(json.load(open('$rec/meta.json'))['task_ids'])")
  iters=$(python3 -c "import json;print(json.load(open('$rec/meta.json'))['iterations_dir'])")
  local log; log=$(cd bundles/data && ./replay_oi.sh "$ids" "$ROOT/$rec/$iters" "$out" 2>&1)
  local misses got exp
  misses=$(echo "$log" | grep -o "MISSES=[0-9]*" | cut -d= -f2)
  got=$(echo "$log" | grep -o '"content_hash": "[0-9a-f]*"' | tail -1 | grep -o "[0-9a-f]\{16\}")
  exp=$(python3 -c "import json;print(json.load(open('$rec/meta.json')).get('content_hash_recording',''))")
  if [ "$misses" = 0 ] && [ -n "$got" ] && [ "$got" = "$exp" ]; then verdict data "$set" ok "misses=0 content_hash=$got (as recorded) verdicts=$(echo "$log" | grep -o '"verdicts": \[[^]]*\]' | tail -1 | cut -c13-)"
  else verdict data "$set" FAIL "misses=${misses:-?} content_hash=${got:-none} expected=$exp (log: $out)"; fi
}

run_coderust() {
  local b=bat-2393 iid=sharkdp__bat-2393 out=$OUTROOT/code-rust; rm -rf "$out"
  local meta=bundles/code-rust/bundles/$b/meta.json
  local exp expp; exp=$(python3 -c "import json;print(json.load(open('$meta'))['content_hash_recording'])"); expp=$(python3 -c "import json;print(json.load(open('$meta')).get('patch_hash_recording',''))")
  (cd bundles/code-rust && MODE=replay ./run_codex_task.sh "$iid" "$out" bundles/$b/responses > "$out.log" 2>&1)
  python3 - code-rust "$b" "$out/summary.json" "$exp" "$expp" <<'PY' || FAIL=1
import json, sys
sc, b, p, exp, expp = sys.argv[1:]
try: s = json.load(open(p))
except Exception as e: print(f"{sc:<10} {b:<16} FAIL no summary ({e})"); sys.exit(1)
ph = s.get("patch_hash") or ""
ok = s.get("misses") == 0 and s.get("content_hash") == exp and (not expp or ph == expp)
print(f"{sc:<10} {b:<16} {'ok' if ok else 'FAIL':<4} misses={s.get('misses')} content_hash={s.get('content_hash')} "
      f"{'(as recorded)' if s.get('content_hash') == exp else 'expected=' + exp} patch_hash={ph} {'(as recorded)' if ph == expp else 'expected=' + expp}")
sys.exit(0 if ok else 1)
PY
}

run_codenode() {
  # a clean claude home: a user's ~/.claude/settings.json can redirect the product's traffic
  export CC_HOME=${CC_HOME:-$OUTROOT/cc_home}; mkdir -p "$CC_HOME"
  local t=requests_5414 out=$OUTROOT/code-node; rm -rf "$out"
  local d=agentcpu/anchors/claude_code/tasks/$t log
  [ -d "$d/repo" ] || { verdict code-node "$t" FAIL "not set up: run $d/setup.sh first"; return; }
  log=$(cd "$d" && ./replay.sh "$out" 2>&1)
  local misses th
  misses=$(echo "$log" | grep -o "misses[=:] *[0-9]*" | tail -1 | grep -o "[0-9]*$")
  th=$(echo "$log" | grep "tool_hash=" | tail -1)
  local patch_ok=yes
  echo "$log" | grep -q "DIFFERS" && patch_ok=no
  if echo "$log" | grep -q "produced.patch sha256=\([0-9a-f]*\)  expected=(recording) \1"; then :; elif echo "$log" | grep -q "produced.patch"; then patch_ok=no; fi
  if [ "$misses" = 0 ] && echo "$th" | grep -q "match=yes" && [ $patch_ok = yes ]; then verdict code-node "$t" ok "misses=0 $th patch=as recorded"
  else verdict code-node "$t" FAIL "misses=${misses:-?} ${th:-no tool hash} patch_ok=$patch_ok (log: $out)"; echo "$log" | tail -8 > "$out.err"; fi
}

case "$SC" in
  api)       run_api;;
  code-py)   run_codepy;;
  data)      run_data;;
  code-rust) run_coderust;;
  code-node) run_codenode;;
  all)       run_api; run_codepy; run_data; run_coderust; run_codenode;;
  *) echo "usage: $0 {api|code-py|data|code-rust|code-node|all}"; exit 2;;
esac
exit $FAIL

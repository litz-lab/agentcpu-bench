#!/usr/bin/env bash
# code-rust bundle driver: the Codex CLI (Rust, static musl binary from the
# npm package) solving one SWE-bench Multilingual task inside the task's own
# evaluation image, with the LLM served by codex_responses_proxy.py on the
# host (record: -> agentcpu recorder -> claude CLI; replay: recorded responses).
#
# Usage: MODE=record|replay ./run_codex_task.sh <instance_id> <out_dir> [rec_dir]
#   record: rec_dir defaults to <out_dir>/responses
#   replay: rec_dir required (a recorded responses dir)
# Outputs in out_dir: events.jsonl (Codex --json), produced.patch, proxy.log,
# summary.json (turns, commands, hash of the agent-visible content).
set -u
cd "$(dirname "$0")"; IID=$1; OUT=$(realpath -m "$2"); mkdir -p "$OUT"
MODE=${MODE:-replay}; REC=${3:-$OUT/responses}
LOCAL=${CODEX_LOCAL:-$PWD/.local}    # scratch for container outputs; set CODEX_LOCAL to a local disk when the bundle sits on root-squashed NFS
mkdir -p "$LOCAL/out"; OUTL=$(mktemp -d "$LOCAL/out/${IID}_XXXX"); chmod a+rwx "$OUTL"
IMG=$(venv/bin/python -c "import pandas as pd,sys; df=pd.read_parquet('swebench_multilingual/data/test-00000-of-00001.parquet'); print(df[df.instance_id==sys.argv[1]].iloc[0].image)" "$IID")
venv/bin/python - "$IID" "$OUTL/prompt.txt" <<'PY'
import pandas as pd, sys
df = pd.read_parquet("swebench_multilingual/data/test-00000-of-00001.parquet"); r = df[df.instance_id == sys.argv[1]].iloc[0]
repo = r.repo.split("/")[1]
open(sys.argv[2], "w").write(
    f"You are working in a checkout of the {repo} repository at /testbed (the current directory); the language toolchain is installed. "
    "Solve the following GitHub issue: make the code change, verify it with the project's own build or test command where practical, "
    "do not modify or add tests, and do not commit. When done, summarize the change in two sentences.\n\nISSUE:\n" + r.problem_statement)
PY
PORT=$((18800 + RANDOM % 150)); RP=$((PORT + 1))
if [ "$MODE" = record ]; then
  export RECORD_BACKEND=claude-cli RECORD_MODEL=${RECORD_MODEL:-claude-opus-5} SAMPLES=1 PORT RECORD_KEY=codex_$IID RECORD_ROOT=$OUT/recorder PYTHONHASHSEED=0
  (cd "$(git rev-parse --show-toplevel 2>/dev/null || echo ../..)" && nohup python3 -m agentcpu.llm.recorder > "$OUT/recorder.log" 2>&1 < /dev/null & echo $! > "$OUT/rec.pid")
  nohup python3 codex_responses_proxy.py record $RP http://127.0.0.1:$PORT/v1 "$REC" > "$OUT/proxy.log" 2>&1 < /dev/null & echo $! > "$OUT/proxy.pid"
else
  nohup python3 codex_responses_proxy.py replay $RP "$REC" > "$OUT/proxy.log" 2>&1 < /dev/null & echo $! > "$OUT/proxy.pid"
fi
for _ in $(seq 1 100); do python3 -c "import socket,sys; s=socket.socket(); s.settimeout(0.2); sys.exit(s.connect_ex(('127.0.0.1', int(sys.argv[1]))))" $RP && break; sleep 0.2; done
T0=$(date +%s.%N)
timeout ${CODEX_TIMEOUT:-3600} docker run --rm --network host ${DOCKER_EXTRA:-} \
  -v "$(realpath vendor)":/opt/codex:ro -v "$PWD/home/.codex":/opt/codex_home:ro -v "$OUTL":/out \
  -e CODEX_HOME=/root/.codex -e AGENTCPU_KEY=x -e PYTHONHASHSEED=0 \
  -e npm_config_offline=true -e npm_config_update_notifier=false -e NO_UPDATE_NOTIFIER=1 -e GOPROXY=off -e GOFLAGS=-mod=mod -e CARGO_NET_OFFLINE=true -e GIT_TERMINAL_PROMPT=0 \
  "$IMG" bash -c "export PATH=/opt/codex/bin:/opt/codex/codex-path:/usr/local/cargo/bin:\$PATH; mkdir -p /root/.codex && cp /opt/codex_home/config.toml /root/.codex/; cd /testbed && git remote remove origin 2>/dev/null; git status --short > /out/pre_status.txt; \
    codex exec --json --skip-git-repo-check -c 'model_providers.agentcpu.base_url=\"http://127.0.0.1:$RP/v1\"' -C /testbed \"\$(cat /out/prompt.txt)\" < /dev/null > /out/events.jsonl 2> /out/codex.err; echo \$? > /out/rc.txt; \
    for f in \$(git ls-files --others --exclude-standard); do grep -Iq . \"\$f\" && git add -N \"\$f\"; done; git diff > /out/produced.patch; git status --short > /out/post_status.txt"
WALL=$(python3 -c "print(round($(date +%s.%N)-$T0,1))")
kill $(cat "$OUT/proxy.pid") 2>/dev/null; [ -f "$OUT/rec.pid" ] && kill $(cat "$OUT/rec.pid") 2>/dev/null
cp -r "$OUTL"/. "$OUT"/; rm -rf "$OUTL"
python3 - "$OUT" "$IID" "$MODE" "$WALL" <<'PY'
import json, sys, hashlib
sys.path.insert(0, ".")
from codex_hash import items_of, canon_of, content_hash
o, iid, mode, wall = sys.argv[1:]
items = items_of(f"{o}/events.jsonl")
cmds = [i for i in items if i["type"] == "command_execution"]; msgs = [i for i in items if i["type"] == "agent_message"]
patches = [i for i in items if i["type"] in ("file_change", "patch_apply")]
h = content_hash(canon_of(items))
patch = open(f"{o}/produced.patch").read(); ph = hashlib.sha256(patch.encode()).hexdigest()[:16]
llm_calls = sum(1 for l in open(f"{o}/proxy.log") if l.startswith("shape")); misses = sum(1 for l in open(f"{o}/proxy.log") if l.startswith("MISS"))
s = {"instance_id": iid, "mode": mode, "codex": "0.156.0", "llm_calls": llm_calls, "misses": misses, "commands": len(cmds), "agent_messages": len(msgs), "file_changes": len(patches),
     "rc": open(f"{o}/rc.txt").read().strip(), "wall_s": float(wall), "content_hash": h, "patch_hash": ph, "patch_bytes": len(patch),
     "final_message": (msgs[-1]["text"][:400] if msgs else "")}
json.dump(s, open(f"{o}/summary.json", "w"), indent=1); print(json.dumps(s))
PY

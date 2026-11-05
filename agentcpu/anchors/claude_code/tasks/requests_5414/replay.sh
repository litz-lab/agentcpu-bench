#!/usr/bin/env bash
# code-node bundle: Claude Code (the shipped product) on SWE-bench Verified
# requests-5414, LLM responses replayed from session_opus5.jsonl, real Bash /
# Read / Edit / Write / Grep / Glob tools against ./repo. The recorded tool
# calls carry the task directory as __TASK_DIR__; a copy of the session with
# this directory filled in is written to the output dir and served from there.
#
# Usage: ./replay.sh [out_dir]      prints turns, replay misses, diff identity
set -u
cd "$(dirname "$0")"
# default product: the pinned 2.1.274 from ../../install_products.sh, never a `claude` on PATH
# CLAUDE_BUILD=2.1.112 selects the Node build (run as `node cli.js`), for tools that pass env as KEY=VALUE words
[ "${CLAUDE_BUILD:-}" = 2.1.112 ] && [ -z "${CLAUDE_BIN:-}" ] && CLAUDE_BIN="node $(cd ../../products/2.1.112/node_modules/@anthropic-ai/claude-code && pwd)/cli.js"
[ -z "${CLAUDE_BIN:-}" ] && [ -x ../../products/2.1.274/node_modules/.bin/claude ] && CLAUDE_BIN=$(cd ../../products/2.1.274/node_modules/.bin && pwd)/claude
OUT=${1:-replay_out}; mkdir -p "$OUT"
./reset.sh > /dev/null
PORT=$((9300 + RANDOM % 200))
../../rebase_session.py session_opus5.jsonl "$OUT/session.jsonl" __TASK_DIR__ "$PWD" > /dev/null
rm -rf "$OUT/reqs"; CCPROXY_DUMP="$OUT/reqs" python3 ../../ccproxy.py replay $PORT "$OUT/session.jsonl" > "$OUT/proxy.log" 2>&1 &
PP=$!; sleep 1
# Replay needs no login: a dummy API key satisfies the CLI's auth check and
# keeps any account setting from redirecting traffic. CC_HOME=<clean dir>
# additionally hides a machine's ~/.claude/settings.json (a settings env can
# send the replay to a different endpoint).
(cd repo && env -u ANTHROPIC_AUTH_TOKEN HOME="${CC_HOME:-$HOME}" \
  ANTHROPIC_API_KEY=replay-dummy ANTHROPIC_BASE_URL=http://127.0.0.1:$PORT \
  CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 DISABLE_TELEMETRY=1 \
  ${CLAUDE_BIN:-claude} -p "$(cat ../prompt.txt)" --model claude-opus-5 \
  --allowedTools "Bash,Read,Edit,Write,Grep,Glob" --max-turns 40 --strict-mcp-config \
  --output-format json > "$OUT/out.json" 2> "$OUT/err.log")
kill $PP 2>/dev/null
(cd repo && git diff > "$OUT/produced.patch")
python3 - "$OUT" <<'PY'
import json, sys, hashlib
o = sys.argv[1]
d = json.load(open(f"{o}/out.json"))
misses = sum(1 for l in open(f"{o}/proxy.log") if l.startswith("MISS"))
h = hashlib.sha256(open(f"{o}/produced.patch", "rb").read()).hexdigest()[:16]
print(f"turns={d.get('num_turns')} is_error={d.get('is_error')} replay_misses={misses}")
print(f"produced.patch sha256={h}  expected=(recording) {hashlib.sha256(open('recorded.patch','rb').read()).hexdigest()[:16]}")
PY
python3 ../../tool_hash.py "$OUT/reqs" "$PWD"
echo "product $(grep -oh 'cc_version=[0-9.]*' "$OUT"/reqs/* 2>/dev/null | sort -u | head -1)"
./reset.sh > /dev/null

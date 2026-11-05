#!/usr/bin/env bash
# One-command replay of a code-rust bundle for tracing: loopback replay proxy, Codex on the baked checkout, offline toolchain.
# The checkout must be at /testbed (the recorded tool calls carry that path),
# with the task toolchain at /usr/local/cargo; the agentcpu_rust image provides both.
# Usage: replay_codex_trace.sh <bundle_dir> <out_dir>   (bundle_dir has responses/, prompt.txt)
set -u
CB=$(cd "$(dirname "$0")" && pwd)   # this bundle directory (vendor/ from setup.sh, config.toml)
B=$1; OUT=$2; mkdir -p "$OUT"; RP=$((18800 + RANDOM % 150))
export HOME=/tmp CODEX_HOME=/tmp/codex_home_$$; mkdir -p $CODEX_HOME; cp "$CB/config.toml" $CODEX_HOME/
export PATH=$CB/vendor/bin:$CB/vendor/codex-path:/usr/local/cargo/bin:$PATH RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/usr/local/cargo
export CARGO_NET_OFFLINE=true GIT_TERMINAL_PROMPT=0 npm_config_offline=true npm_config_update_notifier=false NO_UPDATE_NOTIFIER=1 GOPROXY=off AGENTCPU_KEY=x
python3 "$CB/codex_responses_proxy.py" replay $RP "$B/responses" > "$OUT/proxy.log" 2>&1 & PX=$!
for _ in $(seq 1 300); do python3 -c "import socket,sys; s=socket.socket(); s.settimeout(0.2); sys.exit(s.connect_ex((\"127.0.0.1\", int(sys.argv[1]))))" $RP && break; sleep 0.2; done
cd /testbed && git checkout -q -f . && git clean -qfd
codex exec --json --skip-git-repo-check -c "model_providers.agentcpu.base_url=\"http://127.0.0.1:$RP/v1\"" -C /testbed "$(cat $B/prompt.txt)" < /dev/null > "$OUT/events.jsonl" 2> "$OUT/codex.err"; RC=$?
git diff > "$OUT/produced.patch"; kill $PX 2>/dev/null
echo "RC=$RC MISSES=$(grep -c MISS $OUT/proxy.log) CALLS=$(grep -c shape $OUT/proxy.log) PATCH=$(sha256sum $OUT/produced.patch | cut -c1-16)"; exit $RC

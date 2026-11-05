#!/usr/bin/env bash
# Install the two Claude Code builds the code-node bundles use, pinned, into
# ./products (not on PATH, so a user's own `claude` is never picked up):
#   products/2.1.274  the Bun single-file binary the sessions were recorded on
#   products/2.1.112  the last Node-based release, used for the trace
# The replay scripts default to 2.1.274; select the Node build with
#   CLAUDE_BIN="node $(pwd)/products/2.1.112/node_modules/@anthropic-ai/claude-code/cli.js"
# (call cli.js through node directly: a wrapper named `claude` can be handed
# over to another installed build).
set -euo pipefail
cd "$(dirname "$0")"
for v in 2.1.274 2.1.112; do
  [ -d "products/$v/node_modules/@anthropic-ai/claude-code" ] || npm install --silent --no-audit --no-fund --prefix "products/$v" "@anthropic-ai/claude-code@$v"
done
echo "code-node: $(products/2.1.274/node_modules/.bin/claude --version), node build $(node products/2.1.112/node_modules/@anthropic-ai/claude-code/cli.js --version)"

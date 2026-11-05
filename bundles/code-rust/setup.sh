#!/usr/bin/env bash
# One-time environment for the code-rust bundle:
#  - venv (CPython 3.12.14, requirements.lock): pandas + pyarrow to read the
#    SWE-bench Multilingual table, used by the drivers for image names and prompts;
#  - the Codex CLI 0.156.0 from npm (package-lock.json pins it), exposed as
#    ./vendor (bin/codex, codex-path/rg, codex-resources) for the drivers;
#  - ./home/.codex/config.toml from config.toml.
# Needs: uv, npm, network for the first run. The task images
# (swebench/sweb.eval.x86_64.<instance>) are pulled by docker on first use.
set -euo pipefail
cd "$(dirname "$0")"
[ -x venv/bin/python ] || uv venv -q --python 3.12.14 venv
uv pip install -q --python venv/bin/python -r requirements.lock
[ -x node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/bin/codex ] || npm ci --silent --ignore-scripts
ln -sfn node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl vendor
mkdir -p home/.codex && cp config.toml home/.codex/config.toml
vendor/bin/codex --version && echo "code-rust: venv, Codex and config ready"

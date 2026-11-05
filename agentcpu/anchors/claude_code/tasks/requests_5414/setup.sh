#!/usr/bin/env bash
# Recreate the task checkout and environment (not committed: ./repo, ./venv).
# psf/requests at the SWE-bench Verified base commit for psf__requests-5414.
set -euo pipefail
cd "$(dirname "$0")"
BASE=$(python3 -c "import json;print(json.load(open('instance.json'))['base_commit'])")
test -d repo || git clone -q https://github.com/psf/requests.git repo
(cd repo && git checkout -q "$BASE")
# Portable venv: uv's standalone CPython 3.12 works on every node (Ubuntu 22.04
# and 24.04 alike); the system python differs per node.
test -d venv || { command -v uv >/dev/null && uv venv -q venv --python 3.12; } || python3 -m venv venv
PIP="./venv/bin/python -m pip -q install"; ./venv/bin/python -m pip --version >/dev/null 2>&1 || PIP="uv pip install -q --python venv/bin/python"
$PIP -e repo pytest pytest-httpbin pytest-mock
(cd repo && ../venv/bin/python -m pytest -q -p no:cacheprovider tests/test_requests.py -k "invalid_url" | tail -1)
../../install_products.sh
echo "setup done"

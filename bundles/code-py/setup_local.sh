#!/usr/bin/env bash
# Local-deployment variant of the code-py bundle, used for tracing (DynamoRIO
# must see the tool commands as children of one process tree, so SWE-ReX runs
# them locally instead of in a docker sandbox). Needs the task checkout at
# /testbed (the recorded commands carry that path); builds ./testbed_venv with
# the task's test dependencies, which provides the `python` the commands call.
set -euo pipefail
cd "$(dirname "$0")"
INST=trajectories/requests_5414/instance.json
BASE=$(python3 -c "import json;print(json.load(open('$INST'))['base_commit'])")
if [ ! -d /testbed/.git ]; then git clone -q https://github.com/psf/requests.git /testbed; fi
git -C /testbed checkout -q -f "$BASE"; git -C /testbed remote remove origin 2>/dev/null || true
[ -x testbed_venv/bin/python ] || uv venv -q --python 3.12 testbed_venv
uv pip install -q --python testbed_venv/bin/python -e /testbed pytest pytest-httpbin pytest-mock pip flake8
echo "code-py local: /testbed at $BASE, testbed_venv ready"

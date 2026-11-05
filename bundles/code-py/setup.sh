#!/usr/bin/env bash
# One-time environment for the code-py bundle: a CPython 3.12.14 venv with the
# exact package set of the recording (requirements.lock) and SWE-agent at the
# recorded commit, installed editable (its tool bundles are read from the
# checkout at replay time; rebase_traj.py points the trajectory at it).
# Needs: uv (https://docs.astral.sh/uv/), git, network for the first run.
set -euo pipefail
cd "$(dirname "$0")"
SWE_COMMIT=3ea751c087f32b16e039a2233dd6eefecef325d5    # SWE-agent v1.1.0
[ -d SWE-agent ] || git clone -q https://github.com/SWE-agent/SWE-agent.git SWE-agent
(cd SWE-agent && git checkout -q "$SWE_COMMIT")
[ -x venv/bin/python ] || uv venv -q --python 3.12.14 venv
uv pip install -q --python venv/bin/python -r requirements.lock
uv pip install -q --python venv/bin/python --no-deps -e SWE-agent
venv/bin/sweagent --help > /dev/null && echo "code-py: venv ready ($(venv/bin/python -V), SWE-agent $SWE_COMMIT)"

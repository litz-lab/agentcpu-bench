#!/usr/bin/env bash
# Replay the requests-5414 session with SWE-ReX's local deployment (tools run as
# child processes of this tree, against /testbed). Used by the trace descriptor.
# Usage: ./replay_local.sh [output_dir]
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 LITELLM_LOCAL_MODEL_COST_MAP=True
export PATH="$PWD/testbed_venv/bin:$PATH"
OUT=${1:-/tmp/codepy_out}; mkdir -p "$OUT"
./rebase_traj.py trajectories/requests_5414/psf__requests-5414.traj "$OUT/input.traj"
venv/bin/python -m sweagent.run.run run-replay --traj_path "$OUT/input.traj" --deployment.type local --output_dir "$OUT"

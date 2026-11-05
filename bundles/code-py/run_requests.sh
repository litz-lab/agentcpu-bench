#!/usr/bin/env bash
# code-py bundle: SWE-agent 1.1.0 (config/sweagent_0_7/07.yaml: windowed tools,
# thought_action parser) on SWE-bench Verified psf__requests-5414, LLM responses
# recorded 2026-09-17 from claude-opus-5 through the agentcpu recorder
# (RECORD_BACKEND=claude-cli, one sample per call, --strict-mcp-config), replayed
# by the native ReplayModel on the official SWE-bench image.
# Usage: ./run_requests.sh [output_dir]
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1
OUT=${1:-requests_out}
# On a host whose docker has no bridge network (bridge none, no DNS in
# containers), SWE-ReX cannot build its derived image or publish a
# port. Set SWE_IMAGE to a pre-built derived image (docker save/load from a
# host that could build it) and the deployment runs it on the host network.
DEPLOY=""
if [ -n "${SWE_IMAGE:-}" ]; then
  DEPLOY="--deployment.type docker --deployment.image $SWE_IMAGE --deployment.python_standalone_dir '' --deployment.port 8000 --deployment.docker_args [--network=host] --deployment.pull never"
fi
mkdir -p "$OUT"; ./rebase_traj.py trajectories/requests_5414/psf__requests-5414.traj "$OUT/input.traj"
eval venv/bin/sweagent run-replay --traj_path "$OUT/input.traj" --output_dir "$OUT" $DEPLOY
F=$(ls "$OUT"/*/*.traj | head -1)
./content_hash.py "$F"
echo "expected: content_hash=$(./content_hash.py trajectories/requests_5414/psf__requests-5414.traj | sed 's/.*content_hash=//')  (recording; 2/2 replays matched on 2026-09-17)"
python3 -c "import json,sys; open('$OUT/produced.patch','w').write(json.load(open('$F'))['info'].get('submission',''))"

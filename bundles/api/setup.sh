#!/usr/bin/env bash
# One-time environment for the api scenario outside the capture image:
#  - venv (CPython 3.10.12) with the capture image's package set (requirements.lock);
#  - AppWorld 0.1.3.post1 in its own CPython 3.12 venv (it needs pydantic v1,
#    the runners need v2), its apps unpacked and its data downloaded into ./appworld;
#  - the Hugging Face file the runner loads offline (gpt2 tokenizer) into ./hf;
#  - libagent_markers.so (roi phase markers) built from agentcpu/markers/marker.c.
# Then `source bundles/api/env.sh` (replay.sh does this itself).
# Needs: uv, gcc, network for the first run.
set -euo pipefail
cd "$(dirname "$0")"; HERE=$PWD; ROOT=$(cd ../.. && pwd)
[ -x venv/bin/python ] || uv venv -q --python 3.10.12 venv
grep -v '^#' requirements.lock > .req.txt
uv pip install -q --python venv/bin/python -r .req.txt
[ -x appworld_venv/bin/python ] || uv venv -q --python 3.12 appworld_venv
uv pip install -q --python appworld_venv/bin/python appworld==0.1.3.post1
mkdir -p appworld
[ -d appworld/data ] || (cd appworld && ../appworld_venv/bin/appworld install && ../appworld_venv/bin/appworld download data --root .)
appworld_venv/bin/python -c "import appworld.apps.admin.models" && echo "api: AppWorld ready"
HF_HOME=$HERE/hf venv/bin/python - <<'PY'
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained("gpt2")
print("api: tokenizer cached")
PY
gcc -shared -fPIC -O2 -o libagent_markers.so "$ROOT/agentcpu/markers/marker.c"
echo "api: venv ready ($(venv/bin/python -V))"

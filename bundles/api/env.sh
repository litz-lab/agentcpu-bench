# Environment for replaying api bundles outside the capture image (after setup.sh).
_API=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PATH="$_API/venv/bin:$PATH"
export PYTHONPATH="$(cd "$_API/../.." && pwd)${PYTHONPATH:+:$PYTHONPATH}"
export APPWORLD_ROOT="$_API/appworld" APPWORLD_PYTHON="$_API/appworld_venv/bin/python"
export HF_HOME="$_API/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export AGENT_MARKERS_LIB="$_API/libagent_markers.so"

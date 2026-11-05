#!/usr/bin/env bash
# One-time environment for the data bundle: the CPython 3.11.16 venv of the
# recording (requirements.lock, includes Open Interpreter 0.4.3 and ipykernel)
# and the DABStep files the recorded code cells read. The dataset is Adyen's
# adyen/DABstep on Hugging Face; only data/context/* and data/tasks/*.jsonl are
# needed, placed under ./dabstep/data. Needs: uv, network for the first run.
set -euo pipefail
cd "$(dirname "$0")"
[ -x venv/bin/python ] || uv venv -q --python 3.11.16 venv
uv pip install -q --python venv/bin/python -r requirements.lock
venv/bin/python -c "import interpreter, pandas, ipykernel, jupyter_client" 2>/dev/null && echo "data: venv ready ($(venv/bin/python -V))"
# The kernel runs as python3-kernel, a copy of the venv interpreter, so the
# kernel process has its own executable name in traces and samples.
cp "$(readlink -f venv/bin/python)" venv/bin/python3-kernel
KS=venv/share/jupyter/kernels/python3; mkdir -p "$KS"
printf '{"argv": ["%s", "-m", "ipykernel_launcher", "-f", "{connection_file}"], "display_name": "Python 3 (ipykernel)", "language": "python"}\n' "$PWD/venv/bin/python3-kernel" > "$KS/kernel.json"
if [ ! -f dabstep/data/tasks/all.jsonl ]; then
  echo "data: DABStep files missing; fetching data/context and data/tasks from adyen/DABstep"
  venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("adyen/DABstep", repo_type="dataset", local_dir="dabstep",
                  allow_patterns=["data/context/*", "data/tasks/*"])
PY
fi
ls dabstep/data/context | head -3 && echo "data: DABStep files present"

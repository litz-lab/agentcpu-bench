#!/usr/bin/env bash
# SWE-bench completion check for any instance: apply the instance's test patch
# in the official evaluation image, run FAIL_TO_PASS without and with the given
# fix. Both directions are printed so a check that cannot fail (or cannot pass)
# is visible. Test-file hunks in the fix are excluded, as an agent's edits to
# tests must not override the benchmark's test patch.
# Usage: ./check_swebench.sh <instance.json> <fix.patch>
set -u
I=$(realpath "$1"); P=$(realpath "$2")
IID=$(python3 -c "import json;print(json.load(open('$I'))['instance_id'])")
IMG=docker.io/swebench/sweb.eval.x86_64.$(echo "$IID" | sed 's/__/_1776_/' | tr 'A-Z' 'a-z'):latest
D=$(mktemp -d /tmp/swe_check_XXXX)
python3 - "$I" "$D" <<'PY'
import json, sys
i = json.load(open(sys.argv[1])); d = sys.argv[2]
open(f"{d}/test.patch", "w").write(i["test_patch"])
open(f"{d}/f2p.txt", "w").write("\n".join(json.loads(i["FAIL_TO_PASS"])) + "\n")
PY
cp "$P" "$D/fix.patch"
RUN='cd /testbed && git apply /tmp/c/test.patch && source /opt/miniconda3/bin/activate testbed && mapfile -t IDS < /tmp/c/f2p.txt'
PT='python -m pytest -q -p no:cacheprovider "${IDS[@]}" 2>&1 | grep -E "passed|failed|error|ERROR" | tail -2'
echo "=== $IID FAIL_TO_PASS without the fix (must fail)"
docker run --rm -v "$D":/tmp/c:ro $IMG bash -lc "$RUN && $PT"
echo "=== $IID FAIL_TO_PASS with the fix"
docker run --rm -v "$D":/tmp/c:ro $IMG bash -lc "$RUN && git apply --exclude='*/tests/*' --exclude='tests/*' --exclude='*/test_*.py' --exclude='*/testing/*' /tmp/c/fix.patch && echo fix_applied && $PT"
rm -rf "$D"

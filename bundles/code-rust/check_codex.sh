#!/usr/bin/env bash
# SWE-bench Multilingual completion check for a produced patch: in a fresh
# task container apply the patch, then the instance's test_patch, run the
# eval script's test command, and report FAIL_TO_PASS results. The check must
# be able to fail (run with an empty patch to see the tests fail) and to pass
# (run with the gold patch). Usage: ./check_codex.sh <instance_id> <patch_file|gold|none> <out_dir>
set -u
cd "$(dirname "$0")"; IID=$1; PATCH=$2; OUT=$(realpath -m "$3"); mkdir -p "$OUT"
LOCAL=$(mktemp -d "${TMPDIR:-/tmp}/codex_check_${IID}_XXXX"); chmod a+rwx "$LOCAL"
venv/bin/python - "$IID" "$LOCAL" <<'PY'
import pandas as pd, sys, json
df = pd.read_parquet("swebench_multilingual/data/test-00000-of-00001.parquet"); r = df[df.instance_id == sys.argv[1]].iloc[0]; d = sys.argv[2]
open(f"{d}/test.patch", "w").write(r.test_patch); open(f"{d}/gold.patch", "w").write(r.patch)
open(f"{d}/eval.sh", "w").write(r.eval_script); open(f"{d}/image.txt", "w").write(r.image)
json.dump({"FAIL_TO_PASS": list(r.FAIL_TO_PASS), "PASS_TO_PASS": list(r.PASS_TO_PASS), "log_parser": r.log_parser, "eval_type": r.eval_type}, open(f"{d}/meta.json", "w"))
PY
case "$PATCH" in gold) cp "$LOCAL/gold.patch" "$LOCAL/fix.patch";; none) : > "$LOCAL/fix.patch";; *) cp "$PATCH" "$LOCAL/fix.patch";; esac
IMG=$(cat "$LOCAL/image.txt")
timeout 3600 docker run --rm --network none -v "$LOCAL":/chk "$IMG" bash -c "cd /testbed && { if [ -s /chk/fix.patch ]; then git apply /chk/fix.patch > /chk/apply.log 2>&1; else echo empty patch > /chk/apply.log; fi; }; echo apply_rc=\$? >> /chk/apply.log; bash /chk/eval.sh > /chk/eval.log 2>&1; echo eval_rc=\$? >> /chk/apply.log" > "$OUT/docker.log" 2>&1
cp "$LOCAL"/{apply.log,eval.log,meta.json} "$OUT"/ 2>/dev/null; rm -rf "$LOCAL"
python3 - "$OUT" <<'PY'
import json, re, sys
o = sys.argv[1]; m = json.load(open(f"{o}/meta.json")); log = open(f"{o}/eval.log", errors="replace").read()
seg = log.split(">>>>> Start Test Output")[-1].split(">>>>> End Test Output")[0]
res = {}
for t in m["FAIL_TO_PASS"]:
    leaf = re.escape(t.split(" > ")[-1].strip())   # mocha/karma names are "describe > it"; the log prints the leaf under the describe block
    # cargo: "test <name> ... ok"; go: "--- PASS: <name>"; mocha/karma: "✔ <leaf>" / "✖ <leaf>"; pytest/jest lines handled loosely
    ok = (re.search(rf"test .*{leaf}\b.* \.\.\. ok", seg) or re.search(rf"--- PASS: {leaf}\b", seg) or re.search(rf"[✔✓] {leaf}\s*$", seg, re.M)
          or re.search(rf"(PASS|ok|passed).*{leaf}\b", seg) or re.search(rf"{leaf}\b.*(PASS|ok\b|passed)", seg))
    fail = (re.search(rf"test .*{leaf}\b.* \.\.\. FAILED", seg) or re.search(rf"--- FAIL: {leaf}\b", seg) or re.search(rf"[✖✗x] {leaf}\s*$", seg, re.M)
            or re.search(rf"(FAIL|failed).*{leaf}\b", seg))
    res[t] = "pass" if ok and not fail else ("fail" if fail else "not found")
verdict = all(v == "pass" for v in res.values())
print(json.dumps({"FAIL_TO_PASS": res, "verdict": verdict, "apply": open(f"{o}/apply.log").read().strip().replace("\n", " | ")[:200]}))
json.dump({"FAIL_TO_PASS": res, "verdict": verdict}, open(f"{o}/verdict.json", "w"))
PY

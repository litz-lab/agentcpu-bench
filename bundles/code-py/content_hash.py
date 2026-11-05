#!/usr/bin/env python3
"""Content hash of a SWE-agent trajectory: per-step action|observation plus
the submission, with run-dependent tokens normalized (pytest durations, git
object hashes). Usage: content_hash.py <file.traj>"""
import hashlib, json, re, sys
def norm(s):
    s = re.sub(r"\b\d+\.\d+s\b", "Xs", s)
    s = re.sub(r"\b[0-9a-f]{40}\b", "<sha>", s)
    s = re.sub(r"\[\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+\]", "[ts]", s)  # pytest-httpbin server log stamps
    s = re.sub(r"\[\d\d/[A-Z][a-z]{2}/\d{4} \d\d:\d\d:\d\d\]", "[ts]", s)  # werkzeug access-log stamps
    return s
t = json.load(open(sys.argv[1]))
m = hashlib.sha256()
for step in t.get("trajectory", []):
    m.update((str(step.get("action", "")) + "|" + norm(str(step.get("observation", "")))).encode())
m.update(str(t.get("info", {}).get("submission", "")).encode())
print(f"steps={len(t['trajectory'])} exit={t['info'].get('exit_status')} content_hash={m.hexdigest()[:16]}")

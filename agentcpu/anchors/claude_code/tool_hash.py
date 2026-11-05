"""Content hash of what the tools returned in a Claude Code replay.

Reads the request bodies ccproxy dumped (CCPROXY_DUMP), takes the longest
/v1/messages request (the final history), extracts every tool_result block in
order and hashes their text after removing run-dependent tokens (durations,
git object ids, the checkout path) and line order. A replay whose
tools failed (missing interpreter, missing file) gives a different hash even when
the model's recorded answers, the turn count and the final patch all match.
Usage: tool_hash.py <dump_dir> [task_dir]  -> prints 'tool_results=<n> tool_hash=<16 hex>'"""
import json, os, re, sys, hashlib
d = sys.argv[1]; task_dir = sys.argv[2] if len(sys.argv) > 2 else None; best = None
for f in sorted(os.listdir(d)):
    try: j = json.load(open(os.path.join(d, f)))
    except Exception: continue
    msgs = j.get("messages")
    if isinstance(msgs, list) and (best is None or len(msgs) > len(best)): best = msgs
out = []
for m in best or []:
    c = m.get("content")
    if not isinstance(c, list): continue
    group = []
    for b in c:
        if isinstance(b, dict) and b.get("type") == "tool_result":
            t = b.get("content")
            if isinstance(t, list): t = "".join(x.get("text", "") for x in t if isinstance(x, dict))
            t = str(t)
            # pytest runs whose output is cut by `| tail -N` interleave the test
            # server's threaded access log and tracebacks run-dependently; for those
            # the agent-visible fact is the outcome line, so keep only that.
            summ = re.findall(r"^=*\s*(\d+ (?:failed|passed|error)[^\n]*?)(?: in [\d.]+s)?\s*=*$", t, re.M)
            if summ: t = "PYTEST " + " | ".join(summ)
            t = re.sub(r"(?m)^127\.0\.0\.1 - - \[.*\n?", "", t)  # werkzeug access log
            t = re.sub(r"\b\d+(\.\d+)?s\b", "Ts", t)           # pytest "in 0.35s", timings
            t = re.sub(r"\b\d+\.\d+ seconds\b", "T seconds", t)
            if task_dir: t = t.replace(task_dir, "__TASK_DIR__")   # the checkout location is not content
            t = re.sub(r"\b[0-9a-f]{40}\b", "<sha>", t)            # git object ids (stash entries, commits made by the tool)
            # the Grep tool (ripgrep) and parallel test runners emit lines in a
            # run-dependent order; the set of lines is the agent-visible content
            t = "\n".join(sorted(t.splitlines()))
            group.append(t)
    out += sorted(group)   # parallel tool calls in one turn complete in timing order
h = hashlib.sha256("\x00".join(out).encode()).hexdigest()[:16]
line = f"tool_results={len(out)} tool_hash={h}"
if task_dir and os.path.exists(os.path.join(task_dir, "meta.json")):
    refd = json.load(open(os.path.join(task_dir, "meta.json"))).get("reference", {})
    byb = refd.get("tool_hash_by_build") or ({"recorded build": refd["tool_hash"]} if refd.get("tool_hash") else {})
    hit = [b for b, r in byb.items() if r == h]
    if byb: line += f" match={'yes (' + hit[0] + ')' if hit else 'NO'} references={','.join(r[:8] for r in byb.values())}"
print(line)

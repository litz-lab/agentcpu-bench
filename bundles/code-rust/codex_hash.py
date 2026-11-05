#!/usr/bin/env python3
"""Content hash of a Codex session from its `codex exec --json` event log: the
agent-visible content of every completed item (command, exit code, output,
message text), with run-dependent tokens normalised. One definition shared by
run_codex_task.sh and tools_replay.sh.

Normalisations (only what the model could not have depended on):
  - Codex's own chunk ids and wall-time lines, durations like 1.23s, 19m 16s
    or 1h 2m (cargo prints minutes once a build runs over a minute, as it does
    under DynamoRIO), /tmp paths;
  - npm's update notice;
  - cargo's per-crate progress lines (Compiling/Checking/Fresh/Downloading):
    cargo schedules crates in parallel, so a different subset survives a
    `| tail -N` in every run;
  - cargo test's per-test lines ("test a::b ... ok"): tests run in parallel,
    so a `| tail -N` keeps a different subset each run; the summary line
    ("test result: ok. N passed") is kept;
  - file dates in `ls -l` output;
  - line order within one output (rg and cargo emit in parallel) and the
    completion order of parallel tool calls within one turn;
  - commands that Codex yielded on (exec_command's yield_time_ms elapsed, the
    model moved on) and later killed at session end: they appear as exit 130
    with empty output only when the kill is logged, and the model never saw
    that result, so they are not content.
Usage: codex_hash.py <events.jsonl> [...]   prints the hash per file."""
import hashlib, json, re, sys

def scrub(s: str) -> str:
    s = re.sub(r"Chunk ID: \w+", "Chunk ID: X", s)
    s = re.sub(r"Wall time: [\d.]+ seconds", "Wall time: T", s)
    s = re.sub(r"\b(?:\d+h )?\d+m \d+s\b|\b\d+h \d+m\b", "Ts", s)   # cargo's "1m 32s" for runs over a minute
    s = re.sub(r"\d+\.\d+s", "Ts", s)
    s = re.sub(r"/tmp/[\w\-/\.]+", "/tmp/X", s)
    s = re.sub(r"(?m)^npm notice.*\n?", "", s)
    s = re.sub(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) +\d{1,2} +(\d{2}:\d{2}|\d{4})\b", "DATE", s)
    lines = [l for l in s.splitlines()
             if not re.match(r"\s*(Compiling|Checking|Downloaded|Downloading|Fresh)\s", l)
             and not re.match(r"test \S+ \.\.\. (ok|ignored|FAILED)$", l)]
    return "\n".join(sorted(lines))

def items_of(events_path: str) -> list:
    ev = [json.loads(l) for l in open(events_path) if l.strip().startswith("{")]
    items = [e["item"] for e in ev if e.get("type") == "item.completed" and e["item"]["type"] != "error"]
    return [i for i in items if not (i["type"] == "command_execution" and i.get("exit_code") == 130
                                     and not (i.get("aggregated_output") or "").strip())]

def canon_of(items: list) -> list:
    c = [{"type": i["type"], "command": i.get("command"), "exit": i.get("exit_code"),
          "out": scrub(i.get("aggregated_output", "") or ""), "text": i.get("text")} for i in items]
    c.sort(key=lambda x: json.dumps(x, sort_keys=True))
    return c

def content_hash(canon: list) -> str:
    return hashlib.sha256(json.dumps(canon, sort_keys=True).encode()).hexdigest()[:16]

if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p, content_hash(canon_of(items_of(p))))

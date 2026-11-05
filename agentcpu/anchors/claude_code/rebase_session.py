#!/usr/bin/env python3
"""Rewrite the absolute task directory inside a recorded Claude Code session.

The model's tool calls carry absolute paths (Write/Read file_path, Bash cd),
so a recording made at one checkout location names that location. Recorded
sessions in this repository carry the placeholder __TASK_DIR__ instead;
replay.sh writes a copy with the real directory filled in before starting
ccproxy. The same tool converts a fresh recording to the placeholder form.

Paths live inside base64-encoded SSE bodies, split across input_json_delta
chunks, so each response body is decoded, the chunks of every tool_use block
are joined, the substitution made, and the text re-split into the same number
of chunks (so the product parses the same number of events).
Usage: rebase_session.py <in.jsonl> <out.jsonl> <from_string> <to_string>"""
import base64, json, sys

src, dst, old, new = sys.argv[1:5]

def rewrite_body(text: str) -> str:
    lines = text.split("\n")
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("data: ") and '"input_json_delta"' in line:
            # collect the run of consecutive input_json_delta events (same block)
            run = []
            while i < len(lines) and lines[i].startswith("data: ") and '"input_json_delta"' in lines[i]:
                run.append(i); i += 1
                # skip the blank line / event line between SSE events
                while i < len(lines) and (lines[i] == "" or lines[i].startswith("event: ")):
                    run.append(i); i += 1
            idx = [k for k in run if lines[k].startswith("data: ")]
            evs = [json.loads(lines[k][6:]) for k in idx]
            joined = "".join(e["delta"]["partial_json"] for e in evs)
            if old in joined:
                joined = joined.replace(old, new)
                n = len(evs); base, rem = divmod(len(joined), n)
                pos = 0
                for j, e in enumerate(evs):
                    size = base + (1 if j < rem else 0)
                    e["delta"]["partial_json"] = joined[pos:pos + size]; pos += size
                for k, e in zip(idx, evs):
                    lines[k] = "data: " + json.dumps(e, separators=(",", ":"), ensure_ascii=False)
            out.extend(lines[k] for k in run)
            continue
        out.append(line.replace(old, new) if old in line else line)
        i += 1
    return "\n".join(out)

n_sub = 0
with open(dst, "w") as f:
    for line in open(src):
        d = json.loads(line)
        if "body_b64" in d:
            body = base64.b64decode(d["body_b64"]).decode("utf-8")
            if old in body or old in rewrite_body(body):
                pass
            nb = rewrite_body(body)
            if nb != body:
                n_sub += 1
                d["body_b64"] = base64.b64encode(nb.encode("utf-8")).decode("ascii")
        f.write(json.dumps(d) + "\n")
print(f"{src} -> {dst}: {n_sub} responses rewritten")

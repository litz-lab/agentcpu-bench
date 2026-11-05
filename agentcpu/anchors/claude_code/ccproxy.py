#!/usr/bin/env python3
"""Record/replay proxy for the Anthropic Messages API, for the Claude Code
deployment anchor. Record mode forwards to api.anthropic.com and logs every
exchange; replay mode serves the recorded responses with no network, matching
requests by (path, model) in recorded order so side-channel calls (e.g. a
haiku helper) replay independently of interleaving.

Usage:
  python3 ccproxy.py record <port> <session.jsonl>
  python3 ccproxy.py replay <port> <session.jsonl>
Content determinism note: responses are served as full bodies (the client
parses SSE from the body; pacing is not part of the contract).
"""
import sys, os, time, json, base64, hashlib, threading, http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODE, PORT, LOGF = sys.argv[1], int(sys.argv[2]), sys.argv[3]
UPSTREAM = "api.anthropic.com"
lock = threading.Lock()

def body_model(body: bytes) -> str:
    try:
        return json.loads(body or b"{}").get("model", "")
    except Exception:
        return ""

if MODE == "replay":
    queues = {}
    with open(LOGF) as f:
        for line in f:
            r = json.loads(line)
            queues.setdefault((r["path"], r["model"]), []).append(r)
    print(f"replay: {sum(len(v) for v in queues.values())} exchanges, "
          f"{len(queues)} keys", flush=True)

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a):  # quiet
        pass

    def _read_body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def _respond(self, status, ctype, body):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        self.wfile.write(body)

    def _handle(self):
        body = self._read_body()
        key_path, model = self.path, body_model(body)
        # Session-shape evidence, both modes: request bytes, message count and
        # the characters of message content the model was sent (context).
        try:
            _j = json.loads(body or b"{}")
            _msgs = _j.get("messages", [])
            _chars = len(json.dumps(_msgs)) + len(json.dumps(_j.get("system", "")))
            print(f"shape {MODE} {key_path} model={model} req={len(body)}B "
                  f"msgs={len(_msgs)} ctx_chars={_chars} tools={len(_j.get('tools', []))}",
                  flush=True)
        except Exception:
            pass
        # CCPROXY_DUMP=<dir>: keep every request body (debugging what the
        # client actually sent, e.g. tool results, on a new machine).
        _dump = os.environ.get("CCPROXY_DUMP")
        if _dump:
            os.makedirs(_dump, exist_ok=True)
            with open(os.path.join(_dump, f"req_{int(time.time()*1000)}.json"), "wb") as _f:
                _f.write(body)
        if MODE == "record":
            conn = http.client.HTTPSConnection(UPSTREAM, timeout=600)
            hdrs = {k: v for k, v in self.headers.items()
                    if k.lower() not in ("host", "accept-encoding", "connection", "content-length")}
            hdrs["Host"] = UPSTREAM
            hdrs["Accept-Encoding"] = "identity"
            conn.request(self.command, self.path, body=body, headers=hdrs)
            resp = conn.getresponse()
            rbody = resp.read()
            ctype = resp.getheader("Content-Type") or "application/json"
            rec = {"path": key_path, "model": model, "method": self.command,
                   "req_sha": hashlib.sha256(body).hexdigest()[:16],
                   "status": resp.status, "ctype": ctype,
                   "body_b64": base64.b64encode(rbody).decode()}
            with lock, open(LOGF, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"rec {resp.status} {self.command} {key_path} model={model} "
                  f"req={len(body)}B resp={len(rbody)}B", flush=True)
            self._respond(resp.status, ctype, rbody)
        else:
            with lock:
                q = queues.get((key_path, model))
                rec = q.pop(0) if q else None
            if rec is None:
                print(f"MISS {self.command} {key_path} model={model}", flush=True)
                self._respond(404, "application/json",
                              b'{"error":{"type":"replay_miss"}}')
                return
            self._respond(rec["status"], rec["ctype"],
                          base64.b64decode(rec["body_b64"]))

    do_POST = _handle
    do_GET = _handle
    do_PUT = _handle
    do_DELETE = _handle

ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()

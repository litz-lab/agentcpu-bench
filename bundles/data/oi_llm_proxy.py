#!/usr/bin/env python3
"""SSE adapter between Open Interpreter (litellm, always stream=True) and a
non-streaming OpenAI-compatible upstream (the agentcpu recorder while
recording, replay_server.py while replaying). One process per run, started
and stopped by the driver; forwards each chat request with stream=false and
returns the JSON answer as a two-chunk SSE stream (one delta carrying the
whole message, one finish chunk), which is what litellm's OpenAI stream
parser expects. Usage: oi_llm_proxy.py <listen_port> <upstream_base_url>"""
import json, sys, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT, UP = int(sys.argv[1]), sys.argv[2].rstrip("/")
N = [0]

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(n) or b"{}")
        want_stream = bool(req.get("stream"))
        req["stream"] = False
        r = urllib.request.Request(UP + self.path, data=json.dumps(req).encode(),
                                   headers={"Content-Type": "application/json", "Authorization": "Bearer x"})
        try:
            with urllib.request.urlopen(r, timeout=600) as resp:
                body = resp.read(); status = resp.status
        except urllib.error.HTTPError as e:
            body = e.read(); status = e.code
        N[0] += 1
        print(f"proxy req {N[0]} {self.path} msgs={len(req.get('messages', []))} tools={len(req.get('tools', []) or [])} -> {status}", flush=True)
        if not want_stream or status != 200:
            self.send_response(status); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        full = json.loads(body); ch = full["choices"][0]; msg = ch.get("message", {})
        delta = {"role": "assistant", "content": msg.get("content")}
        if msg.get("tool_calls"):
            delta["tool_calls"] = [dict(tc, index=i) for i, tc in enumerate(msg["tool_calls"])]
        base = {"id": full.get("id", "chatcmpl-proxy"), "object": "chat.completion.chunk",
                "created": full.get("created", 0), "model": full.get("model", req.get("model"))}
        chunks = [dict(base, choices=[{"index": 0, "delta": delta, "finish_reason": None}]),
                  dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": ch.get("finish_reason") or ("tool_calls" if msg.get("tool_calls") else "stop")}])]
        out = b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks) + b"data: [DONE]\n\n"
        self.send_response(200); self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)

print(f"oi_llm_proxy: {PORT} -> {UP}", flush=True)
ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()

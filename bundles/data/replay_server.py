#!/usr/bin/env python3
"""Plain-HTTP OpenAI chat-completions replay server for a recorder bundle
(iter_NNN.json with canonical_response), served in recorded order regardless
of request content. Lets SWE-agent run its REAL litellm -> openai -> httpx
path against recorded Opus-5 responses with no network.
The recorded code names DABStep files under the bundle directory of the
recording machine (meta.json "recording_root"); responses are served with that
prefix replaced by this bundle's directory, so the code reads the same files here.
Usage: replay_server.py <port> <rec_dir>"""
import glob, json, os, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
PORT, REC = int(sys.argv[1]), sys.argv[2]
QUEUE = [json.load(open(f))["canonical_response"] for f in sorted(glob.glob(f"{REC}/iter_*.json"))]
HERE = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(REC, "..", "..", "meta.json")
ROOT = json.load(open(META)).get("recording_root") if os.path.exists(META) else None
if ROOT and os.path.realpath(ROOT) != os.path.realpath(HERE):
    QUEUE = [json.loads(json.dumps(r).replace(ROOT, HERE)) for r in QUEUE]
    print(f"replay: recorded paths under {ROOT} served as {HERE}", flush=True)
LOCK = threading.Lock(); SERVED = [0]
class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0); self.rfile.read(n)
        with LOCK:
            i = SERVED[0]; SERVED[0] += 1
        if i >= len(QUEUE):
            body = json.dumps({"error": {"message": f"replay exhausted after {len(QUEUE)} responses"}}).encode()
            self.send_response(500); print(f"MISS request {i}", flush=True)
        else:
            body = json.dumps(QUEUE[i]).encode(); self.send_response(200)
        self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
print(f"replay: {len(QUEUE)} responses on port {PORT}", flush=True)
ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()

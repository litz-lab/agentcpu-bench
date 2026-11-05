"""Multi-sample recording proxy for any OpenAI-compatible chat endpoint.

A localhost plain-HTTP server speaking the OpenAI v1 chat-completions
wire format, used to record the agentic-mode bundles. The upstream is
any OpenAI-compatible endpoint (Ollama, vLLM, Anthropic's
OpenAI-SDK-compatible endpoint, ...), selected via RECORD_BASE_URL /
RECORD_MODEL / RECORD_API_KEY below. For each incoming agent request it:

  1. Forwards the same request to the upstream endpoint K times
     (configurable via SAMPLES env var; default 3).
  2. Parses tool_calls from each sample.
  3. Picks the canonical response by majority vote on the chosen
     tool_name (ties broken by occurrence order).
  4. Saves all K samples + the canonical pick to disk under
     recordings/$RECORD_KEY/iter_NN.json.
  5. Returns the canonical response to the agent.

The agent runs in agentic mode (parses response.tool_calls, dispatches
the chosen tool). After recording, the canonical recordings can be
replayed deterministically through the existing canonical mock for
reproducible CPU-profile measurements.

Env vars:
  RECORD_KEY      namespace under recordings/; required
  RECORD_ROOT     base dir; default ./recordings
  RECORD_BASE_URL upstream endpoint; default http://localhost:11434/v1
                  (legacy fallback name: OLLAMA_BASE)
  RECORD_MODEL    model name; default llama3.1:8b
                  (legacy fallback name: OLLAMA_MODEL)
  RECORD_API_KEY  bearer token for the upstream; default none
  SAMPLES         K (number of samples per request); default 3
  PORT            listen port; default 18500
"""
from __future__ import annotations

import http.server
import json
import os
import socketserver
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

RECORD_KEY = os.environ.get("RECORD_KEY", "").strip()
if not RECORD_KEY:
    raise SystemExit("agent_ollama_record_server: set RECORD_KEY")

RECORD_ROOT = Path(os.environ.get("RECORD_ROOT", "recordings"))
# Generalized upstream: any OpenAI-compatible chat-completions endpoint.
# RECORD_BASE_URL / RECORD_MODEL / RECORD_API_KEY take precedence; the legacy
# OLLAMA_* names remain as fallbacks so existing recipes keep working.
#   Ollama    : RECORD_BASE_URL=http://localhost:11434/v1        (no key)
#   Anthropic : RECORD_BASE_URL=https://api.anthropic.com/v1     RECORD_MODEL=claude-fable-5
#               RECORD_API_KEY=sk-ant-...   (OpenAI-SDK-compatible endpoint)
#   vLLM      : RECORD_BASE_URL=http://host:8000/v1
RECORD_BASE_URL = os.environ.get(
    "RECORD_BASE_URL",
    os.environ.get("OLLAMA_BASE", "http://localhost:11434/v1")).rstrip("/")
RECORD_MODEL = os.environ.get(
    "RECORD_MODEL", os.environ.get("OLLAMA_MODEL", "llama3.1:8b")).strip()
RECORD_API_KEY = os.environ.get("RECORD_API_KEY", "").strip()
# back-compat aliases used elsewhere in this file
OLLAMA_BASE = RECORD_BASE_URL
# "http" (default) or "claude-cli" -- see _call_claude_cli
RECORD_BACKEND = os.environ.get("RECORD_BACKEND", "http").strip()
OLLAMA_MODEL = RECORD_MODEL
SAMPLES = int(os.environ.get("SAMPLES", "3"))
PORT = int(os.environ.get("PORT", "18500"))

# Debug knobs ---------------------------------------------------------
# DEBUG_LOG: if set, append per-request lines here too (in addition to
#   stderr). Use to confirm whether the runner's traffic actually
#   reaches the recorder when other processes share the host.
# MIN_BODY_BYTES: 0 by default -- save every well-formed request. The
#   filter is here only to reject pathological empty bodies; the runner's
#   own LLM calls can be small (~100 bytes) because the agent uses the
#   LLM for tool selection rather than processing the codebase context.
DEBUG_LOG = os.environ.get("DEBUG_LOG", "").strip()
MIN_BODY_BYTES = int(os.environ.get("MIN_BODY_BYTES", "0"))

REC_DIR = RECORD_ROOT / RECORD_KEY
REC_DIR.mkdir(parents=True, exist_ok=True)


_LOCK = threading.Lock()
_COUNTER = {"i": 0}


def _next_index() -> int:
    with _LOCK:
        i = _COUNTER["i"]
        _COUNTER["i"] = i + 1
        return i


def _call_claude_cli(req: dict, timeout: int = 180) -> dict:
    """One sample straight from the `claude` CLI -- no local HTTP server.

    The shim exists so any OpenAI-compatible endpoint can be the upstream,
    but a long-lived local server is fragile in sandboxed/batch contexts: the
    process group gets reaped and the recorder dies with a connection refused.
    Calling the CLI in-process removes that moving part entirely, which is the
    difference between a re-recording that runs unattended and one that does
    not.

    Set RECORD_BACKEND=claude-cli and RECORD_MODEL=<model id>.
    """
    from agentcpu.llm import shim_claude as _shim
    text = _shim._call_claude(req, model=OLLAMA_MODEL)
    return _shim._to_openai(text, OLLAMA_MODEL)


def _call_ollama(req: dict, timeout: int = 90) -> dict:
    """One request to the upstream endpoint; returns parsed JSON or raises."""
    if RECORD_BACKEND == "claude-cli":
        return _call_claude_cli(req)
    # Force Ollama's model and disable streaming for predictable response
    req = dict(req)
    req["model"] = OLLAMA_MODEL
    req["stream"] = False
    data = json.dumps(req).encode("utf-8")
    headers = {"content-type": "application/json"}
    if RECORD_API_KEY:
        headers["authorization"] = "Bearer " + RECORD_API_KEY
    last = None
    for attempt in range(3):
        try:
            r = urllib.request.urlopen(
                urllib.request.Request(
                    OLLAMA_BASE + "/chat/completions", data=data, headers=headers),
                timeout=timeout,
            )
            return json.loads(r.read())
        except urllib.error.HTTPError as e:  # retry rate limits / transient upstream errors
            last = e
            if e.code in (429, 500, 502, 503, 529) and attempt < 2:
                time.sleep(5 * (attempt + 1)); continue
            raise
    raise last


def _tool_signature(resp: dict) -> tuple:
    """Return (tool_name, args_keys_tuple) or (None, None) if no call."""
    msg = (resp.get("choices") or [{}])[0].get("message", {})
    tcs = msg.get("tool_calls") or []
    if not tcs:
        return (None, None)
    fn = tcs[0].get("function", {})
    name = fn.get("name")
    args = fn.get("arguments", "")
    try:
        keys = tuple(sorted(json.loads(args).keys())) if args else ()
    except (json.JSONDecodeError, AttributeError):
        keys = ()
    return (name, keys)


def _pick_canonical(samples: list[dict]) -> tuple[int, dict]:
    """Pick the canonical sample by majority vote on tool_name.

    Returns (sample_index, canonical_response_json).
    """
    sigs = [_tool_signature(s) for s in samples]
    names = [sig[0] for sig in sigs]
    counts = Counter(n for n in names if n is not None)
    if not counts:
        # No samples picked a tool; fall back to the first sample.
        return 0, samples[0]
    top_name, top_count = counts.most_common(1)[0]
    # Pick the first sample whose tool_name matches the winner.
    for i, n in enumerate(names):
        if n == top_name:
            return i, samples[i]
    return 0, samples[0]


def _handle_one(req: dict) -> bytes:
    """Do K-sample recording for one request; return canonical body."""
    samples = []
    errors = []
    for k in range(SAMPLES):
        try:
            samples.append(_call_ollama(req))
        except Exception as e:  # noqa: BLE001
            errors.append(f"sample{k}: {type(e).__name__}: {e}")
    if not samples:
        raise RuntimeError("all samples failed: " + "; ".join(errors))

    idx, canonical = _pick_canonical(samples)
    sigs = [_tool_signature(s) for s in samples]
    names = [sig[0] for sig in sigs]
    counts = dict(Counter(n for n in names))

    request_idx = _next_index()
    rec_path = REC_DIR / f"iter_{request_idx:03d}.json"
    rec_path.write_text(json.dumps({
        "request": req,
        "samples": samples,
        "tool_signatures": [list(s) if s[1] is not None else [s[0], []]
                             for s in sigs],
        "vote_counts": counts,
        "canonical_idx": idx,
        "canonical_response": canonical,
        "errors": errors,
    }, indent=2))

    return json.dumps(canonical).encode("utf-8")


class Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("content-length", 0))
        body = self.rfile.read(length) if length else b""
        try:
            req = json.loads(body) if body else {}
        except (json.JSONDecodeError, ValueError):
            req = {}

        # Debug trace: per-request line to stderr (and DEBUG_LOG if set).
        # Always on; cheap. Shows EVERY POST so the user can tell whether
        # the runner's traffic is the only thing arriving on this port.
        n_msgs = len(req.get("messages", [])) if isinstance(req, dict) else 0
        model = req.get("model", "?") if isinstance(req, dict) else "?"
        peer = self.client_address[0] if self.client_address else "?"
        line = (f"[{time.strftime('%H:%M:%S')}] POST from {peer} "
                f"len={length} model={model} n_msgs={n_msgs} "
                f"body_head={body[:120]!r}\n")
        try:
            sys.stderr.write(line); sys.stderr.flush()
        except Exception:  # noqa: BLE001
            pass
        if DEBUG_LOG:
            try:
                with open(DEBUG_LOG, "a") as fh:
                    fh.write(line)
            except Exception:  # noqa: BLE001
                pass

        # Reject pathological requests (empty messages array or below
        # MIN_BODY_BYTES, both off by default) with a stub 200. Keeps
        # background port scanners from poisoning the bundle without
        # affecting real runner traffic, which has n_msgs >= 1.
        if (length < MIN_BODY_BYTES) or n_msgs == 0:
            stub = json.dumps({
                "id": "stub", "object": "chat.completion",
                "created": 0, "model": model,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant",
                                         "content": "ok",
                                         "tool_calls": None}}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 1,
                          "total_tokens": 1},
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(stub)))
            self.end_headers()
            self.wfile.write(stub)
            return

        try:
            resp = _handle_one(req)
        except Exception as e:  # noqa: BLE001
            err = json.dumps({"error": {"message": str(e)}}).encode("utf-8")
            self.send_response(500)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)
            return
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, *args, **kwargs) -> None:  # noqa: ARG002
        pass


class _Threaded(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """Threaded HTTP server so long-running Ollama calls don't head-of-line
    block fast probe requests (or vice versa)."""
    daemon_threads = True
    allow_reuse_address = True


def main() -> None:
    httpd = _Threaded(("127.0.0.1", PORT), Handler)
    print(json.dumps({
        "url": f"http://127.0.0.1:{PORT}/v1",
        "rec_dir": str(REC_DIR),
        "ollama": OLLAMA_BASE,
        "model": OLLAMA_MODEL,
        "samples_per_call": SAMPLES,
        "min_body_bytes": MIN_BODY_BYTES,
        "debug_log": DEBUG_LOG or None,
    }), flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

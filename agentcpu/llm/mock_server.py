"""Mock HTTPS server returning canned OpenAI chat-completion responses.

Started as a daemon thread by agent_llm_client. Client-side openai-python
+ httpx connects via real TLS; server returns a pinned response. The
combination preserves bit-identical CPU code paths across runs (every
request invokes the same cipher primitives, same gzip table, same parser
state) while replacing the remote LLM with a local handler.

Determinism notes
-----------------
- TLS handshakes use ephemeral keys (different bytes per session) but the
  CPU code path is identical across runs: same cipher suite, same hash
  rounds, same byte counts. PMU counter values reproduce within
  measurement noise.
- The canned response body is gzipped once at module import; every
  request returns the same bytes.
- Server log messages are suppressed to avoid I/O noise.
"""
from __future__ import annotations

import gzip
import json
import os
import socket
import ssl
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ---------------------------------------------------------------------------
# Canned OpenAI chat-completion response (gzipped once)
# ---------------------------------------------------------------------------

_CANNED_RESPONSE = {
    "id": "chatcmpl-mock-deterministic",
    "object": "chat.completion",
    "created": 0,
    "model": "mock-llm-pinned",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": (
                    "Acknowledged. Per the agent loop contract, I will dispatch "
                    "search_and_summarize on the retrieved corpus, run LexRank "
                    "with the configured sentence count, and return a structured "
                    "evidence list. Final answer JSON object per the system "
                    "prompt's output-format examples."
                ),
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 1500,
        "completion_tokens": 60,
        "total_tokens": 1560,
    },
}

_BODY_BYTES = json.dumps(_CANNED_RESPONSE, sort_keys=True).encode("utf-8")
_BODY_GZIPPED = gzip.compress(_BODY_BYTES)

# ---------------------------------------------------------------------------
# Optional replay-from-recordings path. When AGENT_MOCK_REPLAY_KEY is set,
# the mock returns one of the pre-recorded Claude responses (cycling
# through them if more requests arrive than recordings exist). Used to
# validate that the canonical CPU profile is preserved when realistic
# (claude-generated) content replaces the canned response.
# ---------------------------------------------------------------------------

_REPLAY_KEY = os.environ.get("AGENT_MOCK_REPLAY_KEY", "").strip()
_REPLAY_ROOT = Path(os.environ.get("AGENT_MOCK_REPLAY_ROOT",
                                   "bundles/api"))
_REPLAY_BODIES: list[bytes] = []
_REPLAY_GZ: list[bytes] = []
_REPLAY_REQS: list[dict | None] = []      # recorded request (messages, tools) per iteration
_REPLAY_IDX_LOCK = threading.Lock()
_REPLAY_IDX = {"i": 0}
# Replay fidelity: every incoming request is compared with the recorded one
# at the same position (messages and tool schemas). A mismatch is a miss; the
# response is still served so the run completes, and the counts land in the
# summary (runner.py) so a replay with misses is never mistaken for the
# recorded session.
_REPLAY_STATS = {"requests": 0, "misses": 0, "first_miss": None}

if _REPLAY_KEY:
    rec_dir = _REPLAY_ROOT / _REPLAY_KEY
    if not rec_dir.is_dir():
        raise SystemExit(
            f"agent_mock_server: AGENT_MOCK_REPLAY_KEY={_REPLAY_KEY} but "
            f"{rec_dir} does not exist"
        )
    for p in sorted(rec_dir.glob("iter_*.json")):
        rec = json.loads(p.read_text())
        # Ollama recording proxy emits 'canonical_response'; older
        # claude-record format emits 'response'. Accept either.
        resp = rec.get("canonical_response") or rec.get("response")
        if resp is None:
            raise SystemExit(
                f"agent_mock_server: {p} has no 'canonical_response' "
                f"or 'response' key"
            )
        body = json.dumps(resp, sort_keys=True).encode("utf-8")
        _REPLAY_BODIES.append(body)
        _REPLAY_GZ.append(gzip.compress(body))
        req = rec.get("request")
        _REPLAY_REQS.append({"messages": req.get("messages"), "tools": req.get("tools")}
                            if isinstance(req, dict) else None)
    if not _REPLAY_BODIES:
        raise SystemExit(
            f"agent_mock_server: AGENT_MOCK_REPLAY_KEY={_REPLAY_KEY} "
            f"but {rec_dir} has no iter_*.json files"
        )


def _next_replay_body_gz(body_in: bytes | None = None) -> bytes:
    """Return the next replay body (gzipped), cycling if needed, and check the
    incoming request against the recorded one at this position."""
    with _REPLAY_IDX_LOCK:
        i = _REPLAY_IDX["i"]
        _REPLAY_IDX["i"] = i + 1
        _REPLAY_STATS["requests"] += 1
        rec = _REPLAY_REQS[i % len(_REPLAY_REQS)] if _REPLAY_REQS else None
        miss = i >= len(_REPLAY_GZ)          # more requests than recordings
        if not miss and rec is not None and body_in is not None:
            try:
                got = json.loads(body_in)
                miss = (got.get("messages") != rec["messages"]
                        or got.get("tools") != rec["tools"])
            except (json.JSONDecodeError, ValueError, AttributeError):
                miss = True
        if miss:
            _REPLAY_STATS["misses"] += 1
            if _REPLAY_STATS["first_miss"] is None:
                _REPLAY_STATS["first_miss"] = i
    return _REPLAY_GZ[i % len(_REPLAY_GZ)]


def replay_stats() -> dict | None:
    """Replay fidelity counters for the summary; None when not replaying."""
    if not _REPLAY_KEY:
        return None
    return dict(_REPLAY_STATS, recorded=len(_REPLAY_GZ), key=_REPLAY_KEY)

# ---------------------------------------------------------------------------
# Pre-built SSE streaming chunks. Real OpenAI/Anthropic APIs stream tokens
# via text/event-stream; the response is delivered token-by-token with
# small inter-chunk delays. Production agents spend non-trivial cycles in
# the SSE parser hot path (httpx + openai-python). Toggle with the env
# var AGENT_MOCK_STREAM=1.
# ---------------------------------------------------------------------------

_STREAM_TOKEN_TEXT = (
    "Acknowledged. Per the agent loop contract, I will dispatch "
    "search_and_summarize on the retrieved corpus, run LexRank with "
    "the configured sentence count, and return a structured evidence "
    "list. Final answer JSON object per the system prompt's "
    "output-format examples. The configured retrieval depth governs "
    "the corpus subset visible to the reranker; the system prompt "
    "specifies maximum-sentence cap and required tool call sequence."
)


def _build_stream_chunks() -> list[bytes]:
    """Pre-encode SSE chunks once at module import (no per-request work).

    Splits the canned response text into ~30 token-like chunks, each
    formatted as a single OpenAI streaming-API delta event. Returns the
    list of pre-encoded bytes (each ends with the SSE framing newline).
    """
    words = _STREAM_TOKEN_TEXT.split()
    # group ~3 words per chunk = roughly 30 chunks; mimics token streaming
    chunks: list[bytes] = []
    for i, j in enumerate(range(0, len(words), 3)):
        delta_text = " ".join(words[j : j + 3])
        if i > 0:
            delta_text = " " + delta_text
        ev = {
            "id": "chatcmpl-mock-stream",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "mock-llm-pinned",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": delta_text},
                    "finish_reason": None,
                }
            ],
        }
        line = b"data: " + json.dumps(ev).encode("utf-8") + b"\n\n"
        chunks.append(line)
    # Final chunk closes the stream
    final = {
        "id": "chatcmpl-mock-stream",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "mock-llm-pinned",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    chunks.append(b"data: " + json.dumps(final).encode("utf-8") + b"\n\n")
    chunks.append(b"data: [DONE]\n\n")
    return chunks


_STREAM_CHUNKS = _build_stream_chunks()
_STREAM_DELAY_S = float(os.environ.get("AGENT_MOCK_STREAM_DELAY_MS", "30")) / 1000.0
_STREAM_ENABLED = os.environ.get("AGENT_MOCK_STREAM", "0") == "1"


# ---------------------------------------------------------------------------
# Self-signed cert/key (cached on disk; openssl CLI for portability)
# ---------------------------------------------------------------------------

_CERT_DIR = Path(os.environ.get("AGENT_MOCK_CERT_DIR",
                                 "/tmp/agent_mock_certs"))
_CERT_PATH = _CERT_DIR / "cert.pem"
_KEY_PATH = _CERT_DIR / "key.pem"


def _ensure_self_signed_cert() -> tuple[Path, Path]:
    """Create a self-signed cert+key pair on disk if not already present.

    Cached so re-runs on the same host skip cert generation. Subject
    alternative name includes localhost + 127.0.0.1 so httpx accepts it.
    """
    if _CERT_PATH.exists() and _KEY_PATH.exists():
        return _CERT_PATH, _KEY_PATH
    _CERT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cnf",
                                     delete=False) as cnf:
        cnf.write(
            "[req]\n"
            "distinguished_name = req\n"
            "x509_extensions = v3_ext\n"
            "prompt = no\n"
            "[req]\n"
            "CN = localhost\n"
            "[v3_ext]\n"
            "subjectAltName = @alt_names\n"
            "[alt_names]\n"
            "DNS.1 = localhost\n"
            "IP.1 = 127.0.0.1\n"
        )
        cnf_path = cnf.name
    try:
        subprocess.run(
            [
                "openssl", "req", "-x509", "-newkey", "rsa:2048",
                "-keyout", str(_KEY_PATH),
                "-out", str(_CERT_PATH),
                "-days", "3650",
                "-nodes",
                "-subj", "/CN=localhost",
                "-config", cnf_path,
                "-extensions", "v3_ext",
            ],
            check=True,
            capture_output=True,
        )
    finally:
        os.unlink(cnf_path)
    return _CERT_PATH, _KEY_PATH


# ---------------------------------------------------------------------------
# HTTPS handler
# ---------------------------------------------------------------------------

class MockLLMHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 (stdlib API)
        # Drain request body so the client's send() completes cleanly.
        # Inspect for stream: true; respect either the request body or
        # the AGENT_MOCK_STREAM env var.
        length = int(self.headers.get("content-length", 0))
        body = self.rfile.read(length) if length else b""
        stream = _STREAM_ENABLED
        if not stream and body:
            try:
                stream = json.loads(body).get("stream", False)
            except (json.JSONDecodeError, ValueError):
                pass

        if stream:
            self._send_stream()
        else:
            self._send_json(body)

    def _send_json(self, body_in: bytes = b"") -> None:
        if _REPLAY_KEY:
            body = _next_replay_body_gz(body_in)
        else:
            body = _BODY_GZIPPED
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-encoding", "gzip")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_stream(self) -> None:
        # SSE: text/event-stream with chunked transfer; one delta event
        # per chunk with a small inter-chunk sleep so the client's SSE
        # parser actually has to handle multiple read() calls.
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("transfer-encoding", "chunked")
        self.end_headers()
        for chunk in _STREAM_CHUNKS:
            # HTTP/1.1 chunked-encoding: <hex-length>\r\n<bytes>\r\n
            length = f"{len(chunk):x}\r\n".encode("ascii")
            self.wfile.write(length)
            self.wfile.write(chunk)
            self.wfile.write(b"\r\n")
            try:
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            if _STREAM_DELAY_S > 0:
                time.sleep(_STREAM_DELAY_S)
        # Terminating zero-length chunk
        self.wfile.write(b"0\r\n\r\n")

    def log_message(self, *args, **kwargs) -> None:  # noqa: ARG002
        # Suppress per-request stderr noise.
        pass


# ---------------------------------------------------------------------------
# Public API: start_mock_server() returns the URL
# ---------------------------------------------------------------------------

_server_lock = threading.Lock()
_server_state: dict = {"httpd": None, "thread": None, "url": None}


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_mock_server() -> str:
    """Start the mock HTTPS server on a free localhost port (idempotent).

    Returns the base URL (e.g., "https://127.0.0.1:18443").
    """
    with _server_lock:
        if _server_state["url"] is not None:
            return _server_state["url"]

        cert_path, key_path = _ensure_self_signed_cert()
        port = _pick_free_port()
        httpd = HTTPServer(("127.0.0.1", port), MockLLMHandler)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))
        # Force TLS 1.3 so the cipher CPU path is consistent across runs.
        ctx.minimum_version = ssl.TLSVersion.TLSv1_3
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)

        thread = threading.Thread(
            target=httpd.serve_forever, daemon=True,
            name="agent-mock-llm-server",
        )
        thread.start()

        # Wait briefly for the server to be reachable.
        url = f"https://127.0.0.1:{port}"
        for _ in range(50):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    break
            except OSError:
                time.sleep(0.02)

        _server_state["httpd"] = httpd
        _server_state["thread"] = thread
        _server_state["url"] = url
        return url


def cert_path() -> Path:
    """Return the cached cert path (for client SSL verification config)."""
    return _CERT_PATH


if __name__ == "__main__":
    url = start_mock_server()
    print(f"mock LLM server: {url}")
    print(f"cert: {cert_path()}")
    print("press Ctrl-C to stop")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        pass

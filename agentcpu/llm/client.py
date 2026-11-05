"""Canonical LLM client factory for the AgentCPU benchmark suite.

Constructs an openai-python client wired with the same CPU-side stack
that production agent deployments run:

  - Real openai-python SDK (request build, pydantic-validated typed response)
  - Real httpx state machine + auto gzip decompress
  - Real TLS 1.3 (against agent_mock_server's localhost HTTPS handler;
    cert pinned to skip CA-chain validation cost variability)
  - HTTP/2 if the h2 package is available, else HTTP/1.1
  - OpenTelemetry SDK with httpx auto-instrumentation
    (no-op exporter; spans are constructed and serialized but not shipped)
  - stdlib logging configured with a NullHandler (production agents log
    every step; we exercise the formatting path without I/O)

The resulting client is API-compatible with `openai.OpenAI(...)`. This is
the canonical path for AgentCPU characterization. Use --real-llm at the
workload-script level to swap in a real endpoint for one-off validation.

Determinism
-----------
TLS handshakes use ephemeral keys; the *bytes* on the wire vary per
handshake but the CPU code path (cipher rounds, hash rounds, byte
counts) is fixed. PMU counters reproduce within measurement noise (the
small per-handshake variance is below the smallest unit our measurements
report). OTel span IDs are random; the span construction CPU code path
is fixed.
"""
from __future__ import annotations

import json
import logging
import os
import ssl
from typing import Any

import httpx

from agentcpu.llm import mock_server as _mock_server
from agentcpu.llm.mock_server import cert_path, start_mock_server


# ---------------------------------------------------------------------------
# Logging: NullHandler captures the formatting path without I/O
# ---------------------------------------------------------------------------

_AGENT_LOGGER = logging.getLogger("agent")


def _configure_logging() -> None:
    """Wire stdlib logging with a NullHandler at INFO level.

    Logger formatting (string interpolation, level checks, record
    construction) runs on every log call; the NullHandler skips the I/O
    syscall.  Production agents typically log per request/response.
    """
    if not _AGENT_LOGGER.handlers:
        _AGENT_LOGGER.addHandler(logging.NullHandler())
    _AGENT_LOGGER.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# OpenTelemetry SDK with no-op exporter
# ---------------------------------------------------------------------------

_OTEL_INITIALIZED = False


def _configure_otel() -> None:
    """Set up an OTel TracerProvider with httpx auto-instrumentation.

    Spans are constructed (real CPU work) but exported to a no-op sink
    (no I/O cost). Production agents emit spans for every HTTP call
    and every framework step; this captures the per-span construction
    overhead without shipping data.
    """
    global _OTEL_INITIALIZED
    if _OTEL_INITIALIZED:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor, SpanExporter, SpanExportResult,
        )
    except ImportError:
        # OTel not installed; fall through silently. The production-realistic
        # mode without OTel is still closer to production than simulate_*.
        return

    class _NoopExporter(SpanExporter):
        def export(self, spans):  # noqa: ARG002
            return SpanExportResult.SUCCESS

        def shutdown(self):
            pass

    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(_NoopExporter()))
    trace.set_tracer_provider(provider)

    # httpx auto-instrumentation: every request becomes a span.
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        HTTPXClientInstrumentor().instrument()
    except ImportError:
        pass

    _OTEL_INITIALIZED = True


# ---------------------------------------------------------------------------
# Deterministic in-process transport (for cycle-accurate simulation traces)
# ---------------------------------------------------------------------------

class _SSEByteStream(httpx.SyncByteStream):
    """Yield pre-built SSE event bytes as a sync stream (no socket, no timing)."""

    def __init__(self, chunks: list) -> None:
        self._chunks = chunks

    def __iter__(self):
        yield from self._chunks

    def close(self) -> None:
        pass


class _InProcessTransport(httpx.BaseTransport):
    """Serve the mock server's exact response bytes in-process.

    No real socket, TLS handshake, or server thread -- this eliminates the two
    empirically-measured sources of non-determinism under simulation:
    ephemeral-key TLS crypto (data-dependent bignum work) and the
    timing-dependent poll/spin between the client and the mock-server daemon
    thread. The client still runs the full substrate CPU path: openai-python
    request build, httpx response handling, gzip decompression, JSON parse, and
    pydantic validation -- all on the identical bytes the socket path served.
    Selected by AGENT_DETERMINISTIC_TRANSPORT; the real-TLS path is unchanged
    for perf-mode characterization on real hardware.
    """

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        ms = _mock_server

        body_in = request.content  # materialize request body (same as the wire)
        stream = ms._STREAM_ENABLED
        if not stream and body_in:
            try:
                stream = json.loads(body_in).get("stream", False)
            except (json.JSONDecodeError, ValueError):
                pass

        if stream:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream",
                         "cache-control": "no-cache"},
                stream=_SSEByteStream(ms._STREAM_CHUNKS),
            )

        body = ms._next_replay_body_gz(body_in) if ms._REPLAY_KEY else ms._BODY_GZIPPED
        return httpx.Response(
            200,
            headers={"content-type": "application/json",
                     "content-encoding": "gzip",
                     "content-length": str(len(body))},
            content=body,
        )


# ---------------------------------------------------------------------------
# OpenAI client factory
# ---------------------------------------------------------------------------

def make_llm_client(*, http2: bool | None = None) -> Any:
    """Construct the canonical AgentCPU LLM client.

    Real openai-python SDK + httpx + pydantic + OTel + logging against a
    localhost HTTPS mock server returning canned bytes. CPU code path is
    bit-identical to production; response is deterministic.

    Boots the localhost HTTPS mock server (idempotent), configures OTel
    + logging, and returns an `openai.OpenAI(...)` instance pointing at
    the mock. The client uses real httpx with TLS verification against
    the cached self-signed cert.

    `http2` controls whether HTTP/2 is enabled on the client. Default
    auto-detects the `h2` package; pass False to force HTTP/1.1 for
    diagnostic comparisons.
    """
    from openai import OpenAI

    _configure_logging()
    _configure_otel()

    # Transport selection. Default = real threaded HTTPS mock (TLS 1.3 over a
    # localhost socket), closest to production, used for perf-mode
    # characterization. AGENT_DETERMINISTIC_TRANSPORT swaps in an in-process
    # transport (no socket/TLS/thread) so the traced instruction stream is
    # reproducible under simulation; the client-side substrate CPU path is
    # preserved (see _InProcessTransport).
    if os.environ.get("AGENT_DETERMINISTIC_TRANSPORT", "0") == "1":
        http_client = httpx.Client(transport=_InProcessTransport())
        base_url = "https://mock.local/v1"
    else:
        base_url = start_mock_server() + "/v1"

        # SSL context that trusts our self-signed cert.
        ssl_ctx = ssl.create_default_context(cafile=str(cert_path()))

        if http2 is None:
            try:
                import h2  # noqa: F401
                http2 = True
            except ImportError:
                http2 = False

        http_client = httpx.Client(
            verify=ssl_ctx,
            http2=http2,
            # Production agents typically use connection pooling.
            limits=httpx.Limits(max_keepalive_connections=10,
                                max_connections=20),
        )

    client = OpenAI(
        base_url=base_url,
        api_key="mock-key",
        http_client=http_client,
    )

    if os.environ.get("AGENT_STREAM", "0") == "1":
        client = _wrap_with_streaming(client)

    return client


# ---------------------------------------------------------------------------
# Streaming wrapper: makes every chat.completions.create() use SSE and
# reassembles the deltas into a synthetic ChatCompletion. Lets the runner
# code path stay unchanged while exercising the production SSE parser hot
# path (httpx/openai-python stream iteration).
# ---------------------------------------------------------------------------

def _wrap_with_streaming(client: Any) -> Any:
    """Inject stream=True and consume SSE chunks back into a real ChatCompletion.

    Production agents stream LLM responses (token-by-token via SSE), which
    exercises a different httpx + openai-python code path than the bulk
    JSON response: per-chunk read(), per-chunk SSE-line parser, per-chunk
    pydantic validation of the delta event. Without this wrapper, AgentCPU
    measures only the bulk-JSON path. With it, every framework iteration
    drives the SSE path the way Claude Code (and any production agent) does.

    Returns a real openai-python ChatCompletion (pydantic-validated) so
    callers can use the standard model_dump/.choices[0].message API.
    """
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice
    from openai.types.completion_usage import CompletionUsage

    orig = client.chat.completions.create

    def streaming_create(*args, **kwargs):
        kwargs["stream"] = True
        stream = orig(*args, **kwargs)
        parts: list[str] = []
        finish_reason = "stop"
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                parts.append(delta.content)
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
        content = "".join(parts)
        return ChatCompletion(
            id="chatcmpl-stream",
            object="chat.completion",
            created=0,
            model="mock-llm-pinned",
            choices=[Choice(
                index=0,
                finish_reason=finish_reason,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=content,
                ),
            )],
            usage=CompletionUsage(
                prompt_tokens=0,
                completion_tokens=len(parts),
                total_tokens=len(parts),
            ),
        )

    client.chat.completions.create = streaming_create
    return client


# ---------------------------------------------------------------------------
# Smoke entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--no-http2", action="store_true")
    args = parser.parse_args()

    print("constructing AgentCPU LLM client...")
    client = make_llm_client(http2=not args.no_http2)
    print(f"  base_url = {client.base_url}")

    for i in range(args.n):
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model="mock-llm",
            messages=[
                {"role": "system", "content": "You are a research assistant."},
                {"role": "user", "content": f"Iteration {i}: do the task."},
            ],
            max_tokens=100,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"  iter {i}: {len(resp.choices[0].message.content or '')} bytes, "
              f"{elapsed_ms:.1f} ms")

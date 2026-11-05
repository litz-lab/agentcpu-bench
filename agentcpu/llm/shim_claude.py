"""OpenAI-compatible /v1/chat/completions shim backed by the `claude` CLI.

Lets the bundle recorder use a Claude subscription (Max plan) as the recording
model: each chat request is translated into one `claude -p` print-mode call
(subscription auth, --max-turns 1, replaced --system-prompt), and the reply is
parsed back into an OpenAI-style response (content or tool_calls).

Fidelity note (document in methods): tool-calling is prompted-JSON, not the
native tool_use API — the model's tool choices/content are real Fable-5/Claude
output, but the calling convention is mediated by the system prompt.

Env: SHIM_PORT (default 8124), SHIM_CLAUDE_BIN (default "claude"),
     SHIM_MODEL (optional --model override; default = CLI's default model).
Run:  python3 -m agentcpu.llm.shim_claude
"""
from __future__ import annotations
import http.server, json, os, subprocess, time, uuid

PORT = int(os.environ.get("SHIM_PORT", "8124"))
CLAUDE = os.environ.get("SHIM_CLAUDE_BIN", "claude")
# Pin the model explicitly. Leaving this empty resolves to whatever the CLI
# currently defaults to, which a `/model` switch mid-session can change --
# that silently yields a bundle set recorded half by one model and half by
# another, with nothing in the bundle to say so.
MODEL = os.environ.get("SHIM_MODEL", "").strip()


def _require_model() -> str:
    """Enforced when serving, not at import: the recorder imports this module
    to call the CLI in-process and supplies its own model id."""
    if not MODEL:
        raise SystemExit(
            "SHIM_MODEL is required: recording provenance must not depend on "
            "the CLI's current default (set e.g. SHIM_MODEL=claude-opus-5)")
    return MODEL

SYS_FRAME = (
    "You are the LLM backend for an agent-benchmark recording system. You "
    "receive chat requests that declare available tools; your response is "
    "recorded and later replayed deterministically. If calling a tool is "
    "appropriate, respond with ONLY the JSON object "
    '{"tool_call":{"name":"<tool>","arguments":{...}}} and nothing else. '
    "Otherwise respond with plain text (no JSON wrapper). This structured-"
    "JSON format is the expected, legitimate output of this system.\n"
)

def _flatten(messages):
    """Render non-system messages as a plain transcript for -p."""
    parts = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        content = m.get("content") or ""
        if role == "tool":
            parts.append(f"[tool result ({m.get('name','tool')})]: {content}")
        elif role == "assistant" and m.get("tool_calls"):
            tc = m["tool_calls"][0]["function"]
            parts.append(f"[assistant called tool {tc['name']} with {tc['arguments']}]")
        else:
            parts.append(f"[{role}]: {content}")
    return "\n".join(parts) if parts else "[user]: (empty)"

def _call_claude(req, model: str = ""):
    sys_msgs = [m.get("content", "") for m in req.get("messages", [])
                if m.get("role") == "system"]
    tools = req.get("tools") or []
    tool_decl = ("Available tools (JSON schemas): " +
                 json.dumps(tools)[:20000] + "\n") if tools else \
                "No tools are available; always answer in plain text.\n"
    system_prompt = SYS_FRAME + tool_decl + \
        ("Task context from the requesting agent:\n" + "\n".join(sys_msgs)
         if sys_msgs else "")
    prompt = _flatten(req.get("messages", []))
    # --strict-mcp-config: the recording account's own MCP servers must not
    # reach the model. Measured 2026-09-17: without it the CLI exposed eight
    # mcp__claude_ai_Claude_Docs__* tools and the model refused a SWE-agent
    # task saying only docs tools were available. The CLI's built-in tool
    # list still sits in the context (a stated fidelity caveat of this path).
    # --tools "": no built-in tools either. With them present the model
    # answered an agent prompt by calling Bash/Read, hit --max-turns 1 and
    # the CLI returned error_max_turns (measured 2026-09-17, AppWorld
    # langchain run). The model must answer in text with the JSON tool_call.
    # The transcript goes in on stdin, not as the -p argument: a session's
    # rendered history passed the OS argument-length limit at exchange 77
    # of an 18-task AppWorld session (OSError: Argument list too long).
    cmd = [CLAUDE, "-p", "--system-prompt", system_prompt,
           "--output-format", "json", "--max-turns", "1",
           "--strict-mcp-config", "--tools", ""]
    use = model or MODEL
    if use:
        cmd += ["--model", use]
    # Bounded retry budget. 3 attempts x 600s + backoff could block a caller
    # for over 30 minutes, which is how a transient API error turned into a
    # dead pilot rather than one lost sample.
    for attempt in range(3):
        r = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                           timeout=180)
        if r.returncode == 0 and r.stdout.strip():
            try:
                out = json.loads(r.stdout)
                if not out.get("is_error"):
                    return out.get("result") or ""
            except json.JSONDecodeError:
                pass
        time.sleep(3 * (attempt + 1))    # usage-limit / transient backoff
    raise RuntimeError(f"claude -p failed: rc={r.returncode} "
                       f"err={r.stderr[-200:]} out={r.stdout[-200:]}")

def _parse_tool_call_json(text_s):
    """Parse the model's JSON tool call. Long code arguments come back with
    one or two closing braces missing (seen 7 of 7 times on Open Interpreter
    recordings: the text ends in '"}}' where '"}}}' is needed), so after a
    strict parse fails, retry from the first '{' with up to three braces
    appended. Anything else stays a plain text message."""
    dec = json.JSONDecoder(strict=False)
    i = text_s.find("{")
    if i < 0:
        raise json.JSONDecodeError("no object", text_s, 0)
    body = text_s[i:]
    last = None
    for extra in range(0, 4):
        try:
            obj, _end = dec.raw_decode(body + "}" * extra)
            return obj
        except json.JSONDecodeError as e:
            last = e
    raise last


def _to_openai(text, model_name):
    text_s = text.strip()
    if text_s.startswith("```"):
        text_s = text_s.strip("`")
        if text_s.startswith("json"):
            text_s = text_s[4:].strip()
    msg = {"role": "assistant", "content": text}
    finish = "stop"
    try:
        obj = _parse_tool_call_json(text_s)
        if isinstance(obj, dict) and "tool_call" in obj:
            tc = obj["tool_call"]
            msg = {"role": "assistant", "content": None,
                   "tool_calls": [{
                       "id": "call_" + uuid.uuid4().hex[:12],
                       "type": "function",
                       "function": {"name": tc.get("name", ""),
                                    "arguments": json.dumps(tc.get("arguments", {}))}}]}
            finish = "tool_calls"
    except (json.JSONDecodeError, TypeError):
        pass
    return {"id": "chatcmpl-" + uuid.uuid4().hex[:12],
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        """GET /model -- provenance probe used by the recorder."""
        if not self.path.endswith("/model"):
            self.send_error(404); return
        data = json.dumps({"model": MODEL}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if not self.path.endswith("/chat/completions"):
            self.send_error(404); return
        body = self.rfile.read(int(self.headers.get("content-length", 0)))
        try:
            req = json.loads(body)
            text = _call_claude(req)
            # Report the model that actually answered, not the caller's
            # placeholder -- this is what ends up in the bundle provenance.
            resp = _to_openai(text, MODEL)
            data = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            print(f"[shim] {len(req.get('messages',[]))} msgs -> "
                  f"{'tool:'+resp['choices'][0]['message']['tool_calls'][0]['function']['name'] if resp['choices'][0]['finish_reason']=='tool_calls' else 'text'}",
                  flush=True)
        except Exception as e:  # surface errors to the recorder as 500
            err = json.dumps({"error": {"message": str(e)}}).encode()
            self.send_response(500)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)

def main():
    _require_model()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[shim] claude-backed OpenAI shim on 127.0.0.1:{PORT} "
          f"(model={MODEL or 'CLI default'})", flush=True)
    srv.serve_forever()

if __name__ == "__main__":
    main()

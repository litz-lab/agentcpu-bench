#!/usr/bin/env python3
"""Record/replay adapter for the OpenAI Responses API as the Codex CLI speaks
it (Codex 0.156 accepts only wire_api = "responses").

record mode: each POST /v1/responses is converted to one chat-completions
request and sent to the agentcpu recorder (which calls the claude CLI); the
recorder's answer (text or tool_calls) is converted back into Responses-API
items and streamed to Codex as SSE. The converted response object is saved as
rec_dir/resp_NNN.json together with a shape line of the request, so replay
serves the exact bytes Codex saw.
replay mode: serves rec_dir/resp_NNN.json in recorded order, no network.

Usage: codex_responses_proxy.py record <port> <recorder_chat_base_url> <rec_dir>
       codex_responses_proxy.py replay <port> <rec_dir>"""
import glob, json, sys, time, uuid, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODE, PORT = sys.argv[1], int(sys.argv[2])
if MODE == "record":
    UP, REC = sys.argv[3].rstrip("/"), sys.argv[4]
else:
    REC = sys.argv[3]
import os; os.makedirs(REC, exist_ok=True)
QUEUE = [json.load(open(f)) for f in sorted(glob.glob(f"{REC}/resp_*.json"))] if MODE == "replay" else []
N = [0]

def _text_of(content):
    if isinstance(content, str): return content
    return "".join(c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") in ("input_text", "output_text", "text"))

def responses_to_chat(req):
    """Codex Responses request -> chat-completions request for the recorder."""
    msgs = []
    if req.get("instructions"): msgs.append({"role": "system", "content": req["instructions"]})
    calls_by_id = {}
    for it in req.get("input", []):
        t = it.get("type", "message")
        if t == "message":
            msgs.append({"role": it.get("role", "user"), "content": _text_of(it.get("content", ""))})
        elif t == "function_call":
            calls_by_id[it["call_id"]] = it
            msgs.append({"role": "assistant", "content": None, "tool_calls": [{"id": it["call_id"], "type": "function",
                         "function": {"name": it["name"], "arguments": it.get("arguments", "{}")}}]})
        elif t == "function_call_output":
            out = it.get("output", "")
            if not isinstance(out, str): out = json.dumps(out)
            msgs.append({"role": "tool", "tool_call_id": it["call_id"], "content": out})
        elif t == "reasoning":
            continue
        else:
            msgs.append({"role": "user", "content": json.dumps(it)})
    tools = []
    for t in req.get("tools", []) or []:
        if t.get("type") == "function":
            tools.append({"type": "function", "function": {"name": t["name"], "description": t.get("description", ""), "parameters": t.get("parameters", {"type": "object", "properties": {}})}})
        # other tool types (local_shell, web_search, ...) are not offered to the recorder
    chat = {"model": req.get("model", "claude-opus-5"), "messages": msgs, "stream": False}
    if tools: chat["tools"] = tools
    return chat

def chat_to_response(chat_resp, model):
    """Recorder chat answer -> Responses API response object (non-streamed form)."""
    msg = chat_resp["choices"][0]["message"]
    items = []
    if msg.get("tool_calls"):
        for tc in msg["tool_calls"]:
            cid = "call_" + uuid.uuid4().hex[:16]
            items.append({"type": "function_call", "id": "fc_" + uuid.uuid4().hex[:16], "call_id": cid, "status": "completed",
                          "name": tc["function"]["name"], "arguments": tc["function"]["arguments"]})
    else:
        items.append({"type": "message", "id": "msg_" + uuid.uuid4().hex[:16], "status": "completed", "role": "assistant",
                      "content": [{"type": "output_text", "text": msg.get("content") or "", "annotations": []}]})
    return {"id": "resp_" + uuid.uuid4().hex[:16], "object": "response", "created_at": int(time.time()), "status": "completed",
            "model": model, "output": items, "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
            "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}}}

def sse(resp):
    """Stream a completed response as the event sequence Codex consumes."""
    ev = []
    def add(t, **kw): ev.append(dict(type=t, sequence_number=len(ev), **kw))
    base = dict(resp); base["output"] = []; base["status"] = "in_progress"
    add("response.created", response=base)
    add("response.in_progress", response=base)
    for i, item in enumerate(resp["output"]):
        start = dict(item);
        if item["type"] == "message": start = dict(item, content=[], status="in_progress")
        if item["type"] == "function_call": start = dict(item, arguments="", status="in_progress")
        add("response.output_item.added", output_index=i, item=start)
        if item["type"] == "message":
            part = item["content"][0]
            add("response.content_part.added", output_index=i, item_id=item["id"], content_index=0, part=dict(part, text=""))
            add("response.output_text.delta", output_index=i, item_id=item["id"], content_index=0, delta=part["text"], logprobs=[])
            add("response.output_text.done", output_index=i, item_id=item["id"], content_index=0, text=part["text"], logprobs=[])
            add("response.content_part.done", output_index=i, item_id=item["id"], content_index=0, part=part)
        else:
            add("response.function_call_arguments.delta", output_index=i, item_id=item["id"], delta=item["arguments"])
            add("response.function_call_arguments.done", output_index=i, item_id=item["id"], arguments=item["arguments"])
        add("response.output_item.done", output_index=i, item=item)
    add("response.completed", response=resp)
    return b"".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in ev)

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def do_GET(self):
        self.send_response(404); self.send_header("Content-Length", "0"); self.end_headers()
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0); req = json.loads(self.rfile.read(n) or b"{}")
        i = N[0]; N[0] += 1
        if not self.path.rstrip("/").endswith("/responses"):
            print(f"unhandled path {self.path}", flush=True); self.send_response(404); self.send_header("Content-Length", "0"); self.end_headers(); return
        tools = [t.get("name") or t.get("type") for t in req.get("tools", []) or []]
        print(f"shape {MODE} req {i} model={req.get('model')} input_items={len(req.get('input', []))} bytes={n} tools={tools} stream={req.get('stream')}", flush=True)
        if MODE == "record":
            json.dump(req, open(f"{REC}/req_{i:03d}.json", "w"))          # saved before forwarding so a failure is debuggable
            chat = responses_to_chat(req)
            json.dump(chat, open(f"{REC}/chat_{i:03d}.json", "w"))
            r = urllib.request.Request(UP + "/chat/completions", data=json.dumps(chat).encode(), headers={"Content-Type": "application/json", "Authorization": "Bearer x"})
            try:
                with urllib.request.urlopen(r, timeout=900) as up: chat_resp = json.loads(up.read())
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace"); print(f"UPSTREAM {e.code}: {body[:500]}", flush=True)
                err = json.dumps({"error": {"message": f"recorder {e.code}: {body[:300]}", "type": "server_error"}}).encode()
                self.send_response(400); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(err))); self.end_headers(); self.wfile.write(err); return
            resp = chat_to_response(chat_resp, req.get("model", "claude-opus-5"))
            json.dump({"request_shape": {"input_items": len(req.get("input", [])), "bytes": n, "tools": tools}, "response": resp}, open(f"{REC}/resp_{i:03d}.json", "w"), indent=1)
        else:
            if i >= len(QUEUE):
                print(f"MISS request {i}", flush=True); body = json.dumps({"error": {"message": "replay exhausted"}}).encode()
                self.send_response(500); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
            resp = QUEUE[i]["response"]
        out = sse(resp) if req.get("stream", True) else json.dumps(resp).encode()
        self.send_response(200); self.send_header("Content-Type", "text/event-stream" if req.get("stream", True) else "application/json")
        self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)

print(f"codex_responses_proxy {MODE} on {PORT}" + (f" -> {UP}" if MODE == "record" else f" ({len(QUEUE)} responses)"), flush=True)
ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()

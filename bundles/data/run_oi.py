#!/usr/bin/env python3
"""Drive Open Interpreter 0.4.3 on one DABStep task, non-interactively, with
its Python code running in the product's own Jupyter kernel (jupyter_client
KernelManager, ipykernel). The LLM side is an OpenAI-compatible URL given on
the command line (recorder while recording, replay_server while replaying),
reached through oi_llm_proxy.py because Open Interpreter streams.
Usage: run_oi.py <task_id> <llm_base_url> <out_dir> [--kernel-gc-probe]
Writes out_dir/messages.json (full conversation), summary.json (rounds,
code cells, kernel pid, final answer, DABStep verdict) and content_hash."""
import hashlib, json, os, re, sys, time
# Single-threaded native libraries in the orchestrator: HF tokenizers and
# tiktoken spin worker threads otherwise, which makes whole-tree instruction
# counts vary run to run (gate 4). Set before any import that reads them.
for _k, _v in (("TOKENIZERS_PARALLELISM", "false"), ("RAYON_NUM_THREADS", "1"), ("OMP_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1"), ("OPENBLAS_NUM_THREADS", "1")):
    os.environ.setdefault(_k, _v)
HERE = os.path.dirname(os.path.abspath(__file__))
task_ids, base, out = sys.argv[1].split(","), sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
tasks = {json.loads(l)["task_id"]: json.loads(l) for l in open(os.path.join(HERE, "dabstep/data/tasks/all.jsonl"))}
for l in open(os.path.join(HERE, "dabstep/data/tasks/dev.jsonl")):   # dev rows carry the answers
    d = json.loads(l); tasks[d["task_id"]] = d
ctx = os.path.join(HERE, "dabstep/data/context")

from interpreter import interpreter
interpreter.offline = True; interpreter.disable_telemetry = True; interpreter.auto_run = True
interpreter.verbose = False; interpreter.max_output = 4000; interpreter.loop = False
interpreter.conversation_history = False; interpreter.computer.import_computer_api = False
interpreter.llm.model = "openai/claude-opus-5"; interpreter.llm.api_base = base; interpreter.llm.api_key = "x"
interpreter.llm.supports_functions = True; interpreter.llm.supports_vision = False
interpreter.llm.context_window = 200000; interpreter.llm.max_tokens = 4096
interpreter.computer.languages = [l for l in interpreter.computer.languages if l.name.lower() in ("python", "shell")]
PROMPT_VARIANT = os.environ.get("OI_PROMPT_VARIANT", "files")   # files: list the files every task (10j); state: say the kernel keeps state
if PROMPT_VARIANT == "state":
    interpreter.custom_instructions = (
        "You are answering data-analysis questions over local files in "
        f"{ctx}. Read manual.md and payments-readme.md first when a question needs domain definitions. "
        "Use Python (pandas is available). Your Python runs in one persistent kernel for the whole conversation: "
        "variables and DataFrames from earlier code cells still exist, so reuse them instead of reloading files. "
        "When you have the answer, end your final message with a line of the form 'FINAL ANSWER: <answer>' and stop.")
else:
    interpreter.custom_instructions = (
        "You are answering a data-analysis question over local files. Work only with the files under "
        f"{ctx} (payments.csv, fees.json, merchant_data.json, merchant_category_codes.csv, acquirer_countries.csv, "
        "manual.md, payments-readme.md). Read manual.md and payments-readme.md first when the question needs "
        "domain definitions. Use Python (pandas is available). When you have the answer, end your final message "
        "with a line of the form 'FINAL ANSWER: <answer>' and stop.")

def norm(x): return re.sub(r"[\s'\"\.]+$", "", str(x).strip().lower())
def same(final, gold):
    """DABStep answers: exact after trimming; comma lists compare as sets; numbers within 1e-6."""
    if norm(final) == norm(gold): return True
    if "," in gold and "," in final:
        return {x.strip() for x in gold.split(",")} == {x.strip() for x in final.split(",")}
    try: return abs(float(final) - float(gold)) < 1e-6
    except ValueError: return False
per_task = []; t0 = time.time()
for tid in task_ids:                      # one conversation, one kernel, several tasks (concatenation)
    task = tasks[tid]
    prompt = f"{task['question']}\n\nGuidelines: {task['guidelines']}"
    n0 = len(interpreter.messages); t1 = time.time()
    interpreter.chat(prompt, display=False, stream=False)
    new = interpreter.messages[n0:]
    final = ""
    for m in reversed(new):
        if m.get("role") == "assistant" and m.get("type") == "message":
            mt = re.search(r"FINAL ANSWER:\s*(.+)", m.get("content", "") or "")
            if mt: final = mt.group(1).strip(); break
    gold = str(task.get("answer", "")).strip()
    per_task.append({"task_id": tid, "level": task.get("level"), "gold": gold, "final_answer": final,
                     "verdict": same(final, gold) if gold else None,
                     "rounds": sum(1 for m in new if m.get("role") == "assistant"),   # one LLM call per assistant message or code cell
                     "code_cells": sum(1 for m in new if m.get("role") == "assistant" and m.get("type") == "code"),
                     "wall_s": round(time.time() - t1, 1)})
wall = time.time() - t0

# kernel process: Open Interpreter starts it lazily on first Python execution
kpid = None
try:
    py = interpreter.computer.terminal._active_languages.get("python")
    kpid = py.km.provisioner.process.pid if py is not None else None
except Exception:
    pass
interpreter.computer.terminate()

full = interpreter.messages
json.dump(full, open(os.path.join(out, "messages.json"), "w"), indent=1)
rounds = sum(t["rounds"] for t in per_task); cells = sum(t["code_cells"] for t in per_task)
# content hash over the agent-visible conversation with run-dependent tokens normalised
# The prompt and the model's code name files under the bundle directory, so the hash treats this
# bundle's directory and the recording's (AGENTCPU_RECORDING_ROOT, set by the replay scripts) alike.
ROOTS = [r for r in (HERE, os.environ.get("AGENTCPU_RECORDING_ROOT")) if r]
def scrub(s):
    for r in ROOTS: s = s.replace(r, "BUNDLE")
    s = re.sub(r"0x[0-9a-f]{6,}", "0xADDR", s); s = re.sub(r"\d+\.\d+ ?(s|ms|seconds)", "T", s)
    s = re.sub(r"/tmp/[\w\-/\.]+", "/tmp/X", s)
    # Open Interpreter's line markers interleave with timing and can be split
    # across stdout chunks ("##" + "#active_line10##"); not agent-visible content
    s = re.sub(r"#*active_line\d*#*\n?", "", s); s = re.sub(r"(?m)^(#+|\d+##)\n", "", s)
    return s
h = hashlib.sha256("\n".join(f"{m.get('role')}|{m.get('type')}|{scrub(str(m.get('content')))}" for m in full).encode()).hexdigest()[:16]
summary = {"task_ids": task_ids, "tasks": per_task, "verdicts": [t["verdict"] for t in per_task],
           "rounds": rounds, "code_cells": cells, "messages": len(full), "kernel_pid": kpid, "wall_s": round(wall, 1),
           "content_hash": h, "open_interpreter": "0.4.3", "model": "claude-opus-5 via recorder", "prompt_variant": PROMPT_VARIANT}
json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
open(os.path.join(out, "content_hash"), "w").write(h + "\n")
print(json.dumps(summary))

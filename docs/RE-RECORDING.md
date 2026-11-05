# Recording a new session

A bundle is one recorded session. Recording a new one (another task,
another model, a newer product version) means running the product against
a real model once through the recorder, then proving the replay before it
is measured.

## 1. The recorder

`agentcpu/llm/recorder.py` is a short-lived OpenAI-compatible endpoint,
started once per run and never shared: it forwards each chat-completions
request to a model and writes the exchange as `iter_NNN.json` under
`RECORD_ROOT/RECORD_KEY/`. The backend is selected by
environment:

| backend | env | notes |
|---|---|---|
| claude CLI | `RECORD_BACKEND=claude-cli RECORD_MODEL=claude-opus-5` | used for every released bundle; the CLI is called with `--strict-mcp-config --tools ""`, so the account's MCP tools and the CLI's own tools never reach the model (`agentcpu/llm/shim_claude.py`) |
| any OpenAI-compatible HTTP endpoint | `RECORD_BACKEND=http RECORD_BASE_URL=... RECORD_MODEL=... RECORD_API_KEY=...` | local models, other providers |

`SAMPLES=1` records one answer per call. The recorder's `meta.json` names
the model that answered.

## 2. Per scenario

| scenario | record with | adapter between the product and the recorder |
|---|---|---|
| api | start the recorder (`RECORD_BACKEND=claude-cli RECORD_MODEL=claude-opus-5 RECORD_KEY=<bundle name> RECORD_ROOT=<out> PORT=<p> python3 -m agentcpu.llm.recorder`), then `python3 -m agentcpu langchain --agentic-mode --real-llm --llm-base-url http://127.0.0.1:<p>/v1 --tool-backend appworld --appworld-task <id> --iterations N ...` | none: the runner's OpenAI client is pointed at the recorder; `--iterations N` runs N AppWorld tasks in one conversation, starting at `--appworld-task`. The recorded `iter_NNN.json` directory is the bundle; replay it with `AGENT_MOCK_REPLAY_KEY=<bundle name>` |
| code-py | `RECORD_MODEL=claude-opus-5 bundles/code-py/record_swebench.sh <instance.json> <out>` | none: SWE-agent's litellm client talks to the recorder; the output `.traj` is the bundle |
| data | `RECORD_MODEL=claude-opus-5 bundles/data/record_oi.sh <task_ids> <out>` | `oi_llm_proxy.py` turns the recorder's answers into the SSE stream Open Interpreter expects |
| code-rust | `MODE=record bundles/code-rust/run_codex_task.sh <instance_id> <out>` (needs the full SWE-bench Multilingual table, see bundles/README.md) | `codex_responses_proxy.py`: Responses API to chat completions and back, saving `resp_NNN.json` with the exact bytes Codex saw |
| code-node | `python3 ccproxy.py record <port> session.jsonl`, then the claude binary with `ANTHROPIC_BASE_URL=http://127.0.0.1:<port>` (see `agentcpu/anchors/claude_code/README.md`) | `ccproxy.py` records the Messages API exchange and replays it by (path, model) order |

Codex samples with an unpinned temperature and yields after a time budget,
so two recordings of the same task are two different sessions; the
recorded one is what replays.

## 3. Environment at recording time

The replay must be able to reproduce every tool result, so the recording
environment is already the replay environment:

* no network except the loopback to the recorder: git remotes removed,
  cargo and npm in offline mode, npm's update notifier disabled;
* thread counts pinned before any import: `OMP_NUM_THREADS=1` and the other
  BLAS variables, `TOKENIZERS_PARALLELISM=false`, `RAYON_NUM_THREADS=1`;
* per-tool parallelism pinned where a tool picks it from the CPU count (for
  example Django's test runner, `DJANGO_TEST_PROCESSES=1`);
* `PYTHONHASHSEED=0`;
* for Claude Code, `CC_HOME` pointing at a clean home, so a machine's
  `~/.claude/settings.json` cannot redirect traffic, and
  `--strict-mcp-config`;
* paths the model sees are paths the replay can provide: the recorded
  Claude Code and SWE-agent sessions carry placeholders
  (`rebase_session.py`, `rebase_traj.py`), and the coding tasks run at
  `/testbed`.

## 4. Proving the bundle

Before the bundle is measured:

1. Replay it twice. Zero misses; content hash (and patch hash) equal to the
   recording's; whole-tree instruction counts within the spreads of
   ARCHITECTURE.md section 3.
2. Run the task source's verdict (`check_swebench.sh`, `check_codex.sh`,
   the DABStep answer check in `run_oi.py`, AppWorld `evaluate()` in the
   runner summary) and write it into `meta.json`. A failed task is kept
   with its verdict; it is not a reason to re-record.
3. Record the session's shape (rounds, prompt tokens first/last/median,
   tool calls).
4. Fill `meta.json`: model, product version, recording date, content hash,
   verdict, and known fidelity notes (for example: prompted-JSON tool
   calling where the product has no native tool use on this wire).

To trace it, add a replay command for it to an image and register it in
scarab-infra's `workloads/workloads_db.json` (ARCHITECTURE.md section 6).

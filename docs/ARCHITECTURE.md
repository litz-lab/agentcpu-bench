# AgentCPU architecture

AgentCPU measures what a CPU does while an LLM agent runs: the agent loop
between model calls (building requests, parsing responses, keeping the
history, dispatching tools) and the tools themselves (shells, compilers,
test runners, a code-interpreter kernel, an application backend). The model
is recorded once and replayed, so every session is a fixed instruction
stream that can be measured with counters, traced, and simulated.

## 1. Five layers

Every scenario is the same five layers; the top three differ per scenario,
the bottom two are shared.

| layer | role | per scenario |
|---|---|---|
| task source | supplies the world (a repository at a base commit, data files, an application database) and the verdict | SWE-bench Verified, SWE-bench Multilingual, DABStep, AppWorld |
| orchestrator | the agent loop: one process | Claude Code (Node), Codex CLI (Rust), SWE-agent, Open Interpreter, the LangChain runner (CPython) |
| tool backend | processes the orchestrator drives | shell, git and pytest against a checkout, cargo and rustc, a Jupyter kernel, the AppWorld worker |
| model | Claude Opus 5, called once at recording time; replayed afterwards without a model: a loopback server behind the product's wire format (Messages, Responses, SSE over Chat Completions), SWE-agent's own `run-replay`, or the runner's in-process transport | shared |
| measurement | counters over the whole process tree and marker phases in the Python runner; DynamoRIO capture, SimPoint, Scarab (all in scarab-infra) | shared |

The products are run as shipped: Claude Code and Codex are pointed at the
replay server through their base-URL settings; nothing in them is patched.
For the coding scenarios the container is the sandbox and the product's own
sandboxing is off; the network is the loopback only (git remotes removed,
cargo and npm offline).

## 2. Phases of a round

Every round of every agent executes the same protocol steps, because the
model APIs require them. The api runner marks them with `roi()` markers
from `libagent_markers.so` (`agentcpu/markers/`); the shipped binaries
cannot be instrumented, so for them phase boundaries come from the
transport (the request and response events of the replay server) and tool
cycles from per-process attribution.

| marker | what it brackets |
|---|---|
| `api_build`, `build_prompt` | assembling the request from history and tool schemas |
| `tokenize` | local token counting, where the harness does it |
| `http_parse` | the SDK call and response decoding |
| `api_parse` | turning the response into typed objects and tool calls |
| `context_update` | appending to the history |
| `tool_dispatch` | routing a tool call to its implementation |
| `tool_exec` | the tool itself |

Marker names are never renamed once baked into the library
(`tests/check_phases.py` checks the runner against them).

## 3. Deterministic replay and the content hash

A bundle is one recorded session: the model's responses, one file per
call, plus the scripts that replay and check it. Four scenarios record
through a short-lived recorder that calls the claude CLI
(`agentcpu/llm/recorder.py` with `RECORD_BACKEND=claude-cli`); Claude Code
records through `ccproxy.py record`, which forwards the product's own
Messages API requests and logs the exchange. Replay serves the recorded
response for each request without a model: a loopback server for Claude
Code (Messages), Codex (Responses API) and Open Interpreter (SSE over Chat
Completions), SWE-agent's own `sweagent run-replay` over the recorded
trajectory, and the api runner's in-process transport (no socket, so the
traced process has no network thread). A request that matches no recorded
exchange is a *miss*, and a session with a miss is not a valid replay.

A match requires the product to send what it sent during recording, which
requires every tool to return what it returned. The checks normalise only
what the model could not have depended on: durations, git object ids, the
checkout path, the line order of parallel tool output, the completion
order of parallel tool calls, and Open Interpreter's `##active_line N##`
markers. Everything else is hashed, and the hash of every replay is
compared with the hash of the recording (`content_hash.py`, `tool_hash.py`,
`codex_hash.py`, `run_oi.py`; the reference is in each bundle's
`meta.json` or recomputed from the recorded trajectory).

Whole-tree instruction counts differ by a fraction of a percent across
replays for the Python and Rust scenarios, and by up to about 2% for Claude
Code, whose JIT and garbage collector are not deterministic. For that
product the determinism check is content identity (same tool calls, same
files, same answer, no misses) plus instruction counts within a few
percent; simulation runs on one captured trace, which is a fixed artifact.

Environment that must be pinned for replay: `PYTHONHASHSEED=0`,
`OMP_NUM_THREADS=1` (and the other BLAS thread variables),
`TOKENIZERS_PARALLELISM=false`, `RAYON_NUM_THREADS=1`, and for the api
runner `AGENT_DETERMINISTIC_TRANSPORT=1`. The replay scripts and the
scarab-infra workload entries set them.

## 4. Session shape

Each session is a real task run end to end: two AppWorld tasks in one
conversation (api), one SWE-bench task (code-py, code-node, code-rust), and
seven DABStep tasks in one Open Interpreter session (data). Rounds, prompt
tokens and tool calls are measured from the recording and reported in the
bundle's `meta.json`, together with the task verdict. A session whose task
failed stays in the roster with its verdict.

## 5. What a bundle must pass

1. **Replay**: zero misses and the recording's content hash, twice.
2. **Completion**: the task source's own verdict, shown to fail on an
   untouched world and recorded in `meta.json`.
3. **Provenance**: `meta.json` names the model that answered (from the
   recorder), the product version and the recording date.

## 6. From a bundle to a trace

Nothing in this repository calls DynamoRIO, SimPoint or Scarab. scarab-infra
holds:

* the images: `workloads/agentcpu` (api, code-py, data, code-node) and
  `workloads/agentcpu_rust` (code-rust) put this repository at a pinned
  commit under `/tmp_home/agentcpu-bench` and run each scenario's
  `setup.sh`;
* the registration: `workloads/workloads_db.json`, suite `agentcpu`, with
  each workload's replay command (`binary_cmd`, `client_bincmd` for a
  replay server, `env_vars`), DynamoRIO arguments and SimPoints;
* the trace descriptors that capture them again.

The orchestrator traces follow one process, selected by
`SCARAB_MAIN_SHARD_GLOB` (the AppWorld worker and the Jupyter kernel are
traced the same way from the same replays). The tool-side traces of the
three coding sessions run the orchestrator natively and trace only what it
starts, through a DynamoRIO shell wrapper: `bundles/code-py/tools_replay.sh`,
`agentcpu/anchors/claude_code/tasks/requests_5414/replay.sh` and
`bundles/code-rust/tools_replay.sh` (with `replay_quiesce.py`, which holds
each recorded answer until the slowed tools have finished). Every process
of such a capture becomes its own workload. SimPoints are 10 M-instruction
segments with 50 M of warmup.

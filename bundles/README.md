# bundles/ - recorded sessions

One directory per scenario (code-node lives in
`../agentcpu/anchors/claude_code/tasks/requests_5414/`). A bundle is a
recorded session (the model's responses, one file per call) plus the
scripts that replay it, check the replay against the recording, and run the
task's own verdict. The model is Claude Opus 5 in every bundle; it is never
called at replay time. `../replay.sh <scenario>` runs the replay and the
check in one step.

| scenario | recorded session | replay | trace replay (scarab-infra) |
|---|---|---|---|
| api | `api/langchain_n2_287e338_1/` (recorder `iter_NNN.json`) | `python -m agentcpu langchain --agentic-mode --tool-backend appworld --appworld-task 287e338_1 ...` with `AGENT_MOCK_REPLAY_KEY=langchain_n2_287e338_1` | the same command |
| code-py | `code-py/trajectories/requests_5414/` (SWE-agent `.traj`; `recorder_iters/` holds the raw recorder output) | `run_requests.sh` (official SWE-bench image under docker) | `replay_local.sh` (tools as local child processes against `/testbed`); `tools_replay.sh` for the tool side |
| data | `data/rec/hard7/` (recorder iterations, the recording's own run output, `meta.json`) | `replay_oi.sh <task_ids> <iterations dir> <out>` | `replay_oi_trace.sh` |
| code-rust | `code-rust/bundles/bat-2393/` (`resp_NNN.json`, the exact bytes Codex saw) | `MODE=replay run_codex_task.sh sharkdp__bat-2393 <out> bundles/bat-2393/responses` (task image under docker) | `replay_codex_trace.sh`; `tools_replay.sh` with `replay_quiesce.py` for the tool side |
| code-node | `../agentcpu/anchors/claude_code/tasks/requests_5414/session_opus5.jsonl` | `replay.sh [out]` through `ccproxy.py` | the same `replay.sh` (tool side), or the Node build under DynamoRIO with `ccproxy.py replay` as the client |

Each bundle carries the recording's content hash (tool results after the
normalisations in docs/ARCHITECTURE.md section 3) and the task verdict;
the check scripts (`code-py/check_swebench.sh`, `code-rust/check_codex.sh`,
the DABStep answer check in `data/run_oi.py`, AppWorld `evaluate()` in the
runner summary) recompute both.

## Recording scripts

| scenario | script | adapter between the product and the recorder |
|---|---|---|
| api | `python -m agentcpu.llm.recorder` plus the runner with `--real-llm` | none |
| code-py | `code-py/record_swebench.sh` | none (SWE-agent's litellm client talks to the recorder) |
| data | `data/record_oi.sh` | `data/oi_llm_proxy.py` (SSE for Open Interpreter) |
| code-rust | `MODE=record code-rust/run_codex_task.sh` | `code-rust/codex_responses_proxy.py` (Responses API) |
| code-node | `ccproxy.py record` | `ccproxy.py` (Messages API) |

See docs/RE-RECORDING.md.

## What is not in git, and where to get it

Each `setup.sh` builds what its scenario needs; the scarab-infra images run
the same scripts.

* **Python environments.** `requirements.lock` in `api/`, `code-py/`,
  `data/` and `code-rust/` is the exact package set of the recorded
  environment (CPython 3.10.12 for api, 3.12.14 for code-py and code-rust,
  3.11.16 for data); `setup.sh` builds `venv/` from it with uv.
* **AppWorld.** `api/setup.sh` installs `appworld 0.1.3.post1` in its own
  venv (it needs pydantic v1) and downloads its data into `api/appworld/`;
  the runner starts it as a worker process
  (`agentcpu/tools/appworld_backend.py`). `source api/env.sh` sets the
  paths outside the image.
* **SWE-agent checkout.** `code-py/setup.sh` clones SWE-agent at commit
  3ea751c (v1.1.0); the trajectory names its tool bundles through the
  placeholder `__SWE_AGENT_DIR__`, which `rebase_traj.py` fills in at replay
  time. `code-py/setup_local.sh` puts the requests checkout at `/testbed`
  and builds `testbed_venv/` for the local-deployment replay.
* **SWE-bench images.** `swebench/sweb.eval.x86_64.<instance>` from Docker
  Hub, for the docker replays of code-py and code-rust and for the
  completion checks. On a host whose docker has no bridge network, see the
  `SWE_IMAGE` note in `code-py/run_requests.sh`.
* **DABStep data.** `data/setup.sh` downloads `data/context/` and
  `data/tasks/` of `adyen/DABstep` into `data/dabstep/`.
* **SWE-bench Multilingual table.** `code-rust/swebench_multilingual/data/`
  holds only the sharkdp__bat-2393 row (the drivers read the task's image,
  prompt and tests from it). To record another task, put the full table
  from `SWE-bench/SWE-bench_Multilingual` on Hugging Face at the same path.
* **Codex binary.** `code-rust/setup.sh` runs `npm ci` (package-lock.json
  pins 0.156.0) and links it as `code-rust/vendor`.
* **Claude Code.** `agentcpu/anchors/claude_code/install_products.sh`
  (called by the task's `setup.sh`) installs both builds into
  `agentcpu/anchors/claude_code/products/`.

Anything else placed under `bundles/` is ignored by git.

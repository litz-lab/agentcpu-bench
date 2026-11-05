# Reproducing

From a clean checkout to replayed sessions, traces and simulations. The
replays need only this repository and the downloads its setup scripts make.
Images, traces, counters and simulation come from scarab-infra, which
builds the images with this repository inside.

## 1. Environments

**Without the images.** Every session replays on an x86-64 Linux host with
uv, gcc, git, npm, node 22 and docker after its setup script:

```bash
bundles/api/setup.sh                                        # LangChain runner venv, AppWorld and its data, marker library
bundles/code-py/setup.sh                                    # SWE-agent 1.1.0 venv
bundles/code-py/setup_local.sh                              # requests checkout at /testbed (local-deployment replay only)
bundles/data/setup.sh                                       # Open Interpreter venv, Jupyter kernel, DABStep files
bundles/code-rust/setup.sh                                  # Codex CLI from npm, a small venv for the task table
agentcpu/anchors/claude_code/tasks/requests_5414/setup.sh   # task checkout and venv, both Claude Code builds
```

`bundles/README.md` lists what each one downloads.

**With the images.** From a scarab-infra checkout (the Dockerfiles clone this repository at the commit `AGENTCPU_COMMIT` pins):

```bash
./sci --build-image agentcpu        # api, code-py, data, code-node
./sci --build-image agentcpu_rust   # code-rust
```

* **agentcpu** holds this repository at `/tmp_home/agentcpu-bench`
  (`$AGENTCPU`) and runs `bundles/api/requirements.lock`,
  `bundles/code-py/setup.sh` and `setup_local.sh`, `bundles/data/setup.sh`
  and the code-node task's `setup.sh`, and builds
  `agentcpu/markers/marker.c`. DynamoRIO is built without annotation
  support so that V8 (code-node) traces correctly.
* **agentcpu_rust** takes the Rust toolchain and the bat checkout at
  `/testbed` from the task's SWE-bench Multilingual image, fetches the
  cargo registry so the replay runs offline, and runs
  `bundles/code-rust/setup.sh`.

## 2. Replaying a session and checking it

`./replay.sh <scenario>` replays one session and prints one verdict line:
replay misses, the content or tool hash against the recording's, and patch
identity or task verdicts; the exit status is 0 only when everything
matches. `./replay.sh all` replays all five. A replay with a miss or a hash
mismatch is not the recorded session; do not measure it. The per-scenario
commands it runs:

```bash
# api: AGENT_MOCK_REPLAY_KEY names the bundle directory under AGENT_MOCK_REPLAY_ROOT
source bundles/api/env.sh     # outside the image
AGENT_MOCK_REPLAY_KEY=langchain_n2_287e338_1 AGENT_MOCK_REPLAY_ROOT=$PWD/bundles/api \
AGENT_DETERMINISTIC_TRANSPORT=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=1 \
python3 -m agentcpu langchain --agentic-mode --tool-backend appworld \
  --appworld-task 287e338_1 --seed 42 --context-size short --iterations 2 \
  --trace-output /tmp/t.json --summary-output /tmp/s.json
# s.json: rounds per task, final answer, AppWorld evaluate() verdict, replay misses

# code-py
bundles/code-py/run_requests.sh /tmp/req_out      # prints the realised and expected content hash
bundles/code-py/check_swebench.sh bundles/code-py/trajectories/requests_5414/instance.json /tmp/req_out/produced.patch   # FAIL_TO_PASS

# data (task ids, recorded iterations, output dir)
bundles/data/replay_oi.sh 1273,1305,1464,1681,1753,1871,2697 \
  bundles/data/rec/hard7/recorder/dabstep_1273_1305_1464_1681_1753_1871_2697 /tmp/oi_out
# run.log last line: content hash and verdicts per task

# code-rust
MODE=replay bundles/code-rust/run_codex_task.sh sharkdp__bat-2393 /tmp/bat_out bundles/code-rust/bundles/bat-2393/responses
bundles/code-rust/check_codex.sh sharkdp__bat-2393 /tmp/bat_out/produced.patch /tmp/bat_chk   # FAIL_TO_PASS (gold|none as controls)

# code-node
agentcpu/anchors/claude_code/tasks/requests_5414/replay.sh /tmp/cc_out
```

Codex runs a turn's tool calls in parallel, so a code-rust replay under
heavy load can differ from the recording at one step while the patch stays
the same; rerun before treating a code-rust mismatch as a broken
environment.

Determinism check: run a replay twice and compare whole-tree instruction
counts (`perf stat -e instructions` over the command); expected spreads are
in ARCHITECTURE.md section 3.

## 3. Inside the images

The scarab-infra workload entries (`workloads/workloads_db.json`, suite
`agentcpu`) call these, with the environment in their `env_vars`:

| workload | command |
|---|---|
| api_langchain_n2, api_langchain_n2_tools | `python3 -m agentcpu langchain ...` (as above) |
| codepy_requests_5414 | `bundles/code-py/replay_local.sh` |
| codepy_requests_5414_tools | `bundles/code-py/tools_replay.sh` |
| data_oi_dabstep_hard7, data_oi_dabstep_hard7_kernel | `bundles/data/replay_oi_trace.sh <task_ids> <iterations dir> <out>` |
| codenode_requests_5414 | the Node build's `cli.js` under DynamoRIO, with `ccproxy.py replay` started by `client_bincmd` |
| codenode_requests_5414_tools | `agentcpu/anchors/claude_code/tasks/requests_5414/replay.sh` with `CLAUDE_BUILD=2.1.112` |
| coderust_codex_bat2393 (perf) | `bundles/code-rust/replay_codex_trace.sh bundles/code-rust/bundles/bat-2393 <out>` |
| coderust_bat2393_tools | `bundles/code-rust/tools_replay.sh bundles/code-rust/bundles/bat-2393 <out>` |

## 4. Traces and simulation (scarab-infra)

```bash
./sci --list-traces agentcpu          # the registered workloads and their sizes
./sci --fetch-traces agentcpu         # download the SimPoint traces into trace_home (or set traces_dir to a directory that holds them)
./sci --sim <descriptor>              # simulate them
./sci --perf <descriptor>             # hardware counters over the replays (agentcpu_perf, agentcpu_rust_perf)
./sci --trace <descriptor>            # capture the traces again (agentcpu_trace, agentcpu_trace_tools, agentcpu_rust_trace_tools)
```

`workloads/workloads_db.json` suite `agentcpu` is the source of truth for
SimPoint ids, weights and the per-workload replay command. The traces do
not need to be captured again to simulate them. Edit `root_dir`, `traces_dir` and
`scarab_path` in a descriptor for the local host before running.

Capture details are in the descriptors: the orchestrator is the main shard
selected by `SCARAB_MAIN_SHARD_GLOB`; code-node is traced on the Node build
of Claude Code (2.1.112), because the Bun binary crashes under DynamoRIO,
and the capture skips the first 1M instructions (`-trace_after_instrs`,
V8's start-up heap check); the tool-side captures trace every process the
orchestrator starts, each as its own workload (ARCHITECTURE.md section 6).

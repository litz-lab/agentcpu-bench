# AgentCPU

A CPU benchmark of LLM-agent sessions: what the CPU does while an agent
runs, with the model taken out of the loop. Each scenario is a session of a
deployed agent product (or the open-source harness of one) on its real
tools, recorded once with Claude Opus 5 and replayed deterministically, so
the same session can be measured with hardware counters, captured as a
DynamoRIO trace, and simulated in Scarab.

This repository holds the five recorded sessions behind the released
traces, the scripts that replay and check them, and the scripts that record
new ones. The container images, the trace capture, the counter runs and the
simulations live in [scarab-infra](https://github.com/litz-lab/scarab-infra)
(suite `agentcpu`).

## Scenarios

| scenario | orchestrator (runtime) | tools | task |
|---|---|---|---|
| api | LangChain agent loop (CPython 3.10) | AppWorld apps in a worker process: FastAPI, SQLModel, SQLite | two AppWorld tasks in one conversation, from 287e338_1 |
| code-py | SWE-agent 1.1.0 (CPython 3.12) | shell, pytest, flake8 against the checkout | SWE-bench Verified psf__requests-5414 |
| data | Open Interpreter 0.4.3 (CPython 3.11) | its Jupyter kernel: pandas, numpy over local files | seven DABStep tasks |
| code-node | Claude Code (Node build 2.1.112 for tracing; recorded on 2.1.274) | Bash, Read, Edit, Write, Grep, Glob | SWE-bench Verified psf__requests-5414 |
| code-rust | Codex CLI 0.156.0 (Rust, static binary) | bash, cargo, rustc, the test binaries | SWE-bench Multilingual sharkdp__bat-2393 |

The traces cover seven agent processes (the orchestrator of each session,
the AppWorld worker and the Jupyter kernel) and the tool side of the three
coding sessions, where every tool process is its own workload.

## Quickstart

```bash
# one-time environments (venvs from lockfiles, pinned checkouts, Codex and Claude Code from npm)
bundles/api/setup.sh; bundles/code-py/setup.sh; bundles/data/setup.sh; bundles/code-rust/setup.sh
agentcpu/anchors/claude_code/tasks/requests_5414/setup.sh

# replay every session; one verdict line each, exit 0 only if all match the recording
./replay.sh all
./replay.sh code-node        # one scenario
```

A verdict line reads, for example,
`code-node  requests_5414    ok   misses=0 tool_results=10 tool_hash=52e6e3a5dbce7761 match=yes (2.1.274 (Bun)) references=52e6e3a5,e6c4e874 patch=as recorded`.
A replay miss or a hash mismatch means the run is not the recorded session.

Images, traces and simulation, from a scarab-infra checkout:

```bash
./sci --build-image agentcpu         # api, code-py, data, code-node, with this repository inside
./sci --build-image agentcpu_rust    # code-rust, with the task's Rust toolchain
./sci --fetch-traces agentcpu        # the released SimPoint traces
```

See [docs/REPRODUCE.md](docs/REPRODUCE.md).

## Layout

```
replay.sh                  replay and check one scenario, or all five
agentcpu/                  the api agent loop and the shared record/replay pieces
  runner.py, registry.py   agent loop and runner registry
  frameworks/langchain.py  the LangChain runner
  core/, generator/        request building, history, determinism helpers; runner flags
  tools/appworld_backend.py  AppWorld worker process (the api tool backend)
  llm/                     recorder, claude CLI shim, replay transport and mock server
  markers/                 roi() phase markers (libagent_markers.so)
  anchors/claude_code/     code-node: Messages API proxy, tool-result hash, the task
bundles/                   api, code-py, data, code-rust: recorded sessions and scripts
tests/check_phases.py      phase order of the api runner against marker.c
docs/                      ARCHITECTURE, REPRODUCE, RE-RECORDING, RELEASE_MANIFEST
traces/                    pointer only
```

## Documentation

| doc | what it covers |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | layers, phases, deterministic replay and the content hash, from a session to a trace |
| [docs/REPRODUCE.md](docs/REPRODUCE.md) | environments, replaying and checking each session, images, traces and simulation |
| [docs/RE-RECORDING.md](docs/RE-RECORDING.md) | recording a new session |
| [docs/RELEASE_MANIFEST.md](docs/RELEASE_MANIFEST.md) | what is released and where it lives |
| [bundles/README.md](bundles/README.md) | per-scenario contents and external dependencies |

## License and attribution

Apache License 2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). The products
and libraries the benchmark drives are third-party and not vendored: Claude
Code (Anthropic), Codex CLI (OpenAI, Apache-2.0), SWE-agent and SWE-ReX
(MIT), Open Interpreter (AGPL-3.0), LangChain (MIT), AppWorld (Apache-2.0;
users download its data), the SWE-bench evaluation images, DABStep (Adyen;
users download the dataset), DynamoRIO (BSD), Scarab and scarab-infra. The
sharkdp__bat-2393 row of the SWE-bench Multilingual task table (MIT) is
included under `bundles/code-rust/swebench_multilingual/`. Recorded model responses are
outputs of Claude Opus 5.

## Citation

```bibtex
@misc{agentcpu,
  title  = {AgentCPU: a deployment-grounded CPU benchmark for LLM agents},
  author = {Surim Oh and Heiner Litz},
  year   = {2026},
  note   = {arXiv identifier pending},
}
```

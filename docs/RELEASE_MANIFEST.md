# AgentCPU release manifest

What the release contains and where each piece lives.

## 1. This repository

| path | contents |
|---|---|
| `replay.sh` | replays and checks one scenario, or all five |
| `bundles/api/`, `agentcpu/` | the LangChain runner, the AppWorld worker, the record/replay transport, phase markers; the recorded api session `langchain_n2_287e338_1` |
| `bundles/code-py/` | SWE-agent replay scripts and the recorded requests_5414 trajectory |
| `bundles/data/` | Open Interpreter driver, SSE adapter, replay server; the recorded hard7 session |
| `bundles/code-rust/` | Codex driver, Responses API proxy, content hash, the bat-2393 row of the SWE-bench Multilingual task table; the recorded bat-2393 session |
| `agentcpu/anchors/claude_code/` | Messages API proxy, tool-result hash, Claude Code installer; the recorded requests_5414 session |

Each scenario's `setup.sh` builds its environment from pinned versions
(lockfiles, commits, npm lockfiles); `docs/RE-RECORDING.md` covers new
recordings.

## 2. Traces

DynamoRIO drmemtrace traces of the replayed sessions, SimPoint-clustered
into 10 M-instruction segments with 50 M instructions of warmup, in the
format Scarab reads. They are registered in scarab-infra
(`workloads/workloads_db.json`, suite `agentcpu`) and downloaded with
`./sci --fetch-traces agentcpu`.

| subsuite | workloads | process traced |
|---|---|---|
| api | api_langchain_n2 | LangChain runner, orchestrator (CPython 3.10) |
| api | api_langchain_n2_tools | AppWorld worker (FastAPI, SQLModel, SQLite; CPython 3.12) |
| code-py | codepy_requests_5414 | SWE-agent 1.1.0 orchestrator (CPython 3.12) |
| data | data_oi_dabstep_hard7 | Open Interpreter 0.4.3 orchestrator (CPython 3.11) |
| data | data_oi_dabstep_hard7_kernel | its Jupyter kernel (ipykernel, pandas, numpy) |
| code-node | codenode_requests_5414 | Claude Code 2.1.112 on Node 22, JavaScript main thread |
| code-py-tools | codepy_requests_5414_tools_* | every tool process of the code-py session, one workload each |
| code-node-tools | codenode_requests_5414_tools_* | every tool process of the code-node session |
| code-rust-tools | coderust_bat2393_tools_* | every thread of every process of the Codex session, including Codex itself (`coderust_bat2393_tools_codex_*`) |

code-node is traced on Claude Code 2.1.112, the last Node-based release,
replaying the session recorded on the Bun build 2.1.274 (the Bun binary
crashes under DynamoRIO). Whole traces are kept and available on request.

## 3. Scarab changes needed to run the traces (litz-lab/scarab)

- x86 decoder mappings for the instruction classes agent traces contain (the
  AVX-512 mask family, AVX-512 lane shuffles, permutes and rotates, SSE3
  horizontal ops, SSE4.1 blends, SHA-NI, the remaining AES-NI forms,
  F2XM1/FSCALE).
- The memtrace reader for `.trace.lz4`, tolerance of ROI phase markers in
  windowed runs and the `--roi_marker_dump` knob (whole-segment runs pass 0),
  and fingerprinting fixes for basic blocks that cross a window boundary.
- The ITTAGE indirect target predictor (`--ibtb_mech ittage`), the
  configuration the traces are characterized with. Runs without the flag are
  unchanged.
- Return-stack fixes: one push or pop per call or return when the L0
  predictor is on, and the wrong-path-safe return stack in golden_cove.

These are contributed to litz-lab/scarab main.

## 4. Images

Built by scarab-infra (`./sci --build-image agentcpu`, `./sci --build-image
agentcpu_rust`; Dockerfiles in `workloads/agentcpu` and
`workloads/agentcpu_rust`), which clone this repository at the commit pinned
by `AGENTCPU_COMMIT`. Inside the images the repository is at
`/tmp_home/agentcpu-bench`. Simulation does not need these images: the
traces run in scarab-infra's `allbench_traces` image.

"""AbstractAgentRunner: the contract every AgentCPU framework runner implements.

A framework runner is a thin subclass that supplies only what differs per
framework; everything shared lives here. To add a framework:

    from agentcpu.registry import register
    from agentcpu.runner import AbstractAgentRunner

    @register("myframework")
    class MyFrameworkRunner(AbstractAgentRunner):
        name = "myframework"
        default_iterations = 10
        phase_sequence = ("api_build", "http_parse", "tool_exec",
                          "api_parse", "context_update")

        def add_arguments(self, parser): ...   # framework-specific knobs
        def setup(self, args): ...             # construct the real framework objects
        def round(self, i, args): ...          # ONE iteration; roi()-bracketed
        def config_dict(self, args): ...       # summary["config"]
        def result_dict(self, args, total_ms): ...  # summary["result"]

The base class owns, in this exact order (the determinism contract):

  1. argparse over the shared knobs (plus the subclass's add_arguments);
  2. enforce_determinism(seed) -- called here, i.e. AFTER the framework
     module's heavy imports (numpy/scipy/langchain/... import at module
     import time, before run() is entered) and after argparse, matching
     the original run_*.py main() ordering;
  3. LLM client construction: canonical deterministic mock transport
     (agentcpu.llm.client.make_llm_client, which honors
     AGENT_MOCK_REPLAY_KEY / AGENT_MOCK_REPLAY_ROOT /
     AGENT_DETERMINISTIC_TRANSPORT transparently) or, with --real-llm,
     a real OpenAI-compatible endpoint (validation only);
  4. subclass setup(args): framework construction (agents, graphs, tool
     registries) -- runs BEFORE the phase log is reset;
  5. reset_phase_log(), then subclass after_reset(args) for recorded
     one-time work (e.g. corpus construction);
  6. the round loop: round(i, args) once per --iterations, stopping early
     when round() returns True (agentic-mode final_answer). The loop itself
     is the run_loop() hook so frameworks whose ORIGINAL runner was async
     can wrap the identical loop in asyncio.run (autogen awaits its tools
     and AssistantAgent.run); then the teardown(args) hook for one-time
     cleanup that the original performed before computing total_ms
     (swe_agent's synthetic-codebase rmtree);
  7. output emission: --trace-output (span list) and --summary-output,
     via the run_metadata/config_dict/result_dict/extra_summary_fields
     hooks, preserving each framework's exact JSON schema.

Phase names
-----------
roi() phase names are baked into libagent_markers.so and the released
traces -- NEVER rename them; additions only (add to markers/marker.c).
The shared per-iteration names every framework composes from:

    api_build       build + serialize the LLM API request
    build_prompt    assemble the prompt message list
    tokenize        BPE tokenize/detokenize
    http_parse      the real openai-python/httpx call (state machine,
                    decompress, pydantic validation)
    api_parse       response model -> dict, tool-result stitching
    context_update  conversation-history growth
    tool_dispatch   agent-executor / graph routing overhead
    tool_exec       real tool kernel work

Framework-specific kernel names (tfidf, cosine_pagerank, numpy_svd,
scipy_stats, numpy_fft, ...) nest inside these. Each subclass declares
its canonical top-level order in `phase_sequence` (default knobs,
non-agentic mode) and its nested kernels in `nested_phases`; these are
documentation + test contract (tests/check_phases.py), and round() must
emit exactly that sequence.

Span schema
-----------
record() appends spans with the fixed schema
{run_id, span_id, phase, start_ns, end_ns, duration_ms, wall_ms, cpu_ms,
 wait_ms, meta}; wall = real time, cpu = thread CPU time, wait = wall-cpu
(network + GPU wait under --real-llm). Frameworks whose original runner
recorded a span without a CPU baseline pass cpu_start_ns=None, which sets
cpu_ms = wall_ms (back-compat with the original per-framework schemas).

Vocabulary
----------
Settled against the literature rather than invented here, because the layer
names are genuinely contested and the knob calibration depends on which layer
a number refers to.

    session   one process invocation == one conversation. Matches TraceLab's
              `session_id`, which is the artifact this suite calibrates to.
    round     one LLM call plus the tool calls it triggers. Matches
              TraceLab's `rounds` table (rounds_per_session is COUNT(rounds)
              GROUP BY session_id).
    task      one goal the agent completes; under the AppWorld backend, one
              AppWorld task. A session holds several.
    tool call one invocation of one tool.

Deliberately avoided:

    "step"    three-ways ambiguous, and not only across groups: mini-swe-agent
              counts one model call per step while SWE-agent proper allows
              several (max_requeries); MLE-bench uses "step" for tree-search
              nodes, CodeAct iterations and ResearchAgent actions in one paper.
    "episode" contested at two different layers -- Terminal-Bench puts it at
              the round layer ("an episode refers to a turn in a trial"),
              SWE-agent at the task layer ("task episode"). This suite used it
              for the session layer, which matched neither.
    "turn"    used interchangeably with "round" (AgentBench uses both), and
              already means something specific in the Anthropic/OpenAI APIs.

Caveat worth stating in any write-up: a session holding SEVERAL tasks is our
construction. Most benchmarks run one task per process. We nest them to reach
the fleet's session lengths (rounds p50 17, p90 161, p99 1786), which a single
AppWorld task -- 17 gold API calls at p99 -- cannot produce.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

from agentcpu.core.common import (build_conversation_history,
                                  TOOL_RESULT_HISTORY_CHARS)
from agentcpu.markers import reset_phase_log, roi

__all__ = ["AbstractAgentRunner", "SHARED_PHASES", "roi", "write_json"]

# The shared roi() vocabulary (see module docstring). Kept here so tools and
# tests have one authoritative list.
SHARED_PHASES = (
    "api_build", "build_prompt", "tokenize", "http_parse",
    "api_parse", "context_update", "tool_dispatch", "tool_exec",
)


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_json(path: str, payload: dict) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


class AbstractAgentRunner:
    """Base class for framework runners. See module docstring for the
    contract; subclass hooks are marked [required] / [optional] below."""

    # ---- class-level contract -------------------------------------------
    #: registry name; also the workload name in output JSON. [required]
    name: str = "abstract"
    #: default for --iterations (frameworks differ; e.g. langchain=2,
    #: langgraph=10). [optional]
    default_iterations: int = 2
    #: argparse description line. [optional]
    parser_description: str | None = None
    #: canonical TOP-LEVEL roi() order for one iteration at default knobs,
    #: non-agentic mode. [required]
    phase_sequence: tuple[str, ...] = ()
    #: roi() names nested inside a top-level phase, e.g.
    #: {"tool_exec": ("tfidf", "cosine_pagerank")}. [optional]
    nested_phases: dict[str, tuple[str, ...]] = {}
    #: declarative KnobSpec tuple mapping swept CLI knobs to TraceLab
    #: dimensions + calibration policies (agentcpu/generator/knobspec.py).
    #: Read only by the generator (`python -m agentcpu.generator`), never
    #: at run time. [optional]
    knobs: tuple = ()

    # ---- argparse ---------------------------------------------------------

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """[optional] Add framework-specific knobs."""

    def build_parser(self) -> argparse.ArgumentParser:
        """The shared knob set. Semantics are identical across frameworks;
        only defaults (task-id, iterations) vary per subclass."""
        parser = argparse.ArgumentParser(description=self.parser_description)
        parser.add_argument("--task-id", default=self.name)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--trace-output", required=True)
        parser.add_argument("--summary-output", required=True)
        parser.add_argument("--context-size",
                            choices=("short", "medium", "long"),
                            default="medium")
        parser.add_argument("--iterations", type=int,
                            default=self.default_iterations,
                            help="Number of agent iterations (in "
                                 "--agentic-mode this is a cap, not a "
                                 "target).")
        parser.add_argument("--system-prompt-tokens", type=int, default=0,
                            help="Scale the system prompt to ~N tokens "
                                 "(production CLAUDE.md / Copilot / Cursor "
                                 "system prompts are 4K-32K). Pushes "
                                 "tokenize + api_build cycle weight up.")
        parser.add_argument("--tool-list-size", type=int, default=0,
                            help="Attach N tool JSON schemas to every LLM "
                                 "request via tools=[...] argument. "
                                 "Production agents commonly expose 5-50 "
                                 "tools. Pushes api_build + pydantic cycle "
                                 "weight up.")
        parser.add_argument("--tool-backend", choices=("synthetic", "appworld"),
                            default="synthetic",
                            help="'appworld' swaps the framework's synthetic "
                                 "tools for AppWorld's real app APIs "
                                 "(FastAPI/pydantic/SQLModel/SQLite). The task "
                                 "then comes from AppWorld too, so grounding "
                                 "and answer checking are inherent rather "
                                 "than invented per framework.")
        parser.add_argument("--tool-result-chars", type=int,
                            default=TOOL_RESULT_HISTORY_CHARS,
                            help="Cap on the tool result kept in "
                                 "conversation history (what the model "
                                 "sees). Set from the operating point's "
                                 "tool-result size.")
        parser.add_argument("--appworld-task", default=None,
                            help="AppWorld task id; default picks "
                                 "deterministically from the loadable set "
                                 "using --seed")
        parser.add_argument("--real-llm", action="store_true",
                            help="Validation-only: real OpenAI-compatible "
                                 "endpoint (real network + GPU). "
                                 "Non-deterministic; do not use for primary "
                                 "characterization.")
        parser.add_argument("--agentic-mode", action="store_true",
                            help="Parse tool_calls from the LLM response and "
                                 "dispatch based on the chosen tool (real "
                                 "function-calling agent). Used during "
                                 "reference-set recording; replays through "
                                 "canonical mock afterward.")
        parser.add_argument("--llm-base-url",
                            default="http://localhost:11434/v1",
                            help="OpenAI-compatible base URL for --real-llm "
                                 "(default: Ollama on localhost).")
        parser.add_argument("--llm-model", default="llama3.2:1b",
                            help="Model name as known to the --real-llm "
                                 "endpoint.")
        parser.add_argument("--llm-api-key", default="ollama",
                            help="API key (Ollama ignores this; real OpenAI "
                                 "needs a real key).")
        parser.add_argument("--no-http2", action="store_true",
                            help="Force HTTP/1.1 in the canonical (default) "
                                 "mode (default: HTTP/2 if h2 is installed).")
        self.add_arguments(parser)
        return parser

    # ---- LLM client ---------------------------------------------------------

    def make_llm_client(self, args: argparse.Namespace):
        """Canonical path: real openai-python + httpx + pydantic + TLS + OTel
        against the in-process deterministic mock (replay bundles selected by
        AGENT_MOCK_REPLAY_KEY, transport by AGENT_DETERMINISTIC_TRANSPORT --
        both handled inside agentcpu.llm, transparently to runners).
        --real-llm: real endpoint, non-deterministic, validation only."""
        if args.real_llm:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise SystemExit(
                    "--real-llm requires the openai Python package: "
                    "pip install --user openai"
                ) from e
            # Recording through the claude shim is SLOW: one upstream call
            # per sample, K samples per request, and the shim retries on
            # transient CLI failures. The SDK's 10-minute default expired
            # mid-retry and killed a 30-minute pilot with zero iterations
            # recorded. Replay is unaffected (in-process transport), so this
            # only matters on the recording path.
            return OpenAI(base_url=args.llm_base_url,
                          api_key=args.llm_api_key,
                          timeout=float(os.environ.get(
                              "AGENT_LLM_TIMEOUT_S", "1800")),
                          max_retries=0)
        try:
            from agentcpu.llm.client import make_llm_client
        except ImportError as e:
            raise SystemExit(
                "Canonical mode requires openai + httpx[http2] + "
                "opentelemetry-sdk + opentelemetry-instrumentation-httpx. "
                "Install with: pip install --user openai 'httpx[http2]' "
                "opentelemetry-sdk opentelemetry-instrumentation-httpx"
            ) from e
        return make_llm_client(http2=not args.no_http2)

    # ---- span recording -------------------------------------------------

    def record(self, phase: str, wall_start_ns: int, meta: dict,
               cpu_start_ns: int | None = None) -> None:
        """Close a span opened at (wall_start_ns, cpu_start_ns) and append it.

        wall = real time; cpu = CPU-active time on this thread. On real-LLM,
        wall - cpu is the network + GPU wait time (thread is blocked in
        socket.recv, not consuming CPU). cpu_start_ns=None sets
        cpu_ms = wall_ms (spans whose original runner recorded no CPU
        baseline)."""
        wall_end = time.perf_counter_ns()
        cpu_end = time.thread_time_ns()
        wall_ms = (wall_end - wall_start_ns) / 1_000_000.0
        cpu_ms = (
            (cpu_end - cpu_start_ns) / 1_000_000.0
            if cpu_start_ns is not None else wall_ms
        )
        self.spans.append({
            "run_id": self.run_id,
            "span_id": f"{len(self.spans)+1:04d}",
            "phase": phase,
            "start_ns": wall_start_ns,
            "end_ns": wall_end,
            "duration_ms": wall_ms,           # back-compat: wall-clock
            "wall_ms": wall_ms,
            "cpu_ms": cpu_ms,
            "wait_ms": max(wall_ms - cpu_ms, 0.0),  # network + GPU wait
            "meta": meta,
        })

    # ---- shared per-round building blocks --------------------------------

    def call_llm(self, request_kwargs: dict):
        """The canonical LLM call under roi("http_parse"): real openai-python
        SDK call (httpx state machine, TLS, auto-decompress, pydantic
        validation). Canonical mode hits the in-process deterministic mock;
        --real-llm hits a real endpoint. Returns (response, content_bytes);
        the caller records its own span (frameworks bracket this call
        differently)."""
        with roi("http_parse"):
            response = self.llm_client.chat.completions.create(
                **request_kwargs
            )
            content = (response.choices[0].message.content
                       or "").encode("utf-8")
        return response, content

    def parse_agentic_tool_call(self, response):
        """In --agentic-mode, extract (tool_name, args_dict) from the LLM
        response's first tool_call; (None, {}) otherwise or when the LLM
        emitted none. Malformed JSON arguments degrade to {}."""
        tool_name = None
        tool_args: dict = {}
        if self.args.agentic_mode:
            tcs = (response.choices[0].message.tool_calls or [])
            if tcs:
                fn = tcs[0].function
                tool_name = fn.name
                try:
                    tool_args = (json.loads(fn.arguments)
                                 if fn.arguments else {})
                except (json.JSONDecodeError, TypeError):
                    tool_args = {}
        if tool_name == "final_answer" and getattr(self, "appworld", None):
            self.appworld_complete(tool_args)
        return tool_name, tool_args

    def appworld_complete(self, tool_args: dict) -> None:
        """Close the task in the world the way AppWorld's own agents do:
        apis.supervisor.complete_task(status="success", answer=...). Without
        this the evaluator's completion test can never pass (measured: 4/5
        on a task whose API work was done). The answer is the 'answer' field
        of a JSON final_answer when present, else the raw text."""
        text = (tool_args or {}).get("text", "")
        answer = text
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "answer" in obj:
                answer = obj["answer"]
        except (json.JSONDecodeError, TypeError):
            pass
        code = ("apis.supervisor.complete_task(status='success', "
                f"answer={answer!r})\nprint('task completed')")
        try:
            self.appworld.execute(code)
        except Exception as e:                       # report, do not crash
            self.appworld_complete_error = f"{type(e).__name__}: {e}"

    def update_history(self, iteration: int, query: str, tool_name: str,
                       tool_result, tool_args: dict | None = None,
                       truncate: int | None = None,
                       with_cpu: bool = True) -> None:
        """Grow self.conversation_history under roi("context_update") via
        core.common.build_conversation_history, and record the
        context_accumulate span. tool_args are the arguments the model
        actually sent (so the model can see its own prior calls); the tool
        result is kept up to --tool-result-chars unless truncate overrides
        it; with_cpu=False records without a CPU baseline (cpu_ms = wall_ms),
        matching the original langgraph span schema."""
        t = time.perf_counter_ns()
        t_cpu = time.thread_time_ns() if with_cpu else None
        cap = (truncate if truncate is not None
               else getattr(self.args, "tool_result_chars",
                            TOOL_RESULT_HISTORY_CHARS))
        with roi("context_update"):
            result_str = (tool_result if isinstance(tool_result, str)
                          else str(tool_result))
            self.conversation_history = build_conversation_history(
                self.conversation_history, query,
                tool_name, result_str, self.rng,
                tool_args=tool_args, max_result_chars=cap,
            )
        self.record("context_accumulate", t, {
            "iteration": iteration,
            "history_len": len(self.conversation_history),
        }, cpu_start_ns=t_cpu)

    # ---- subclass hooks ---------------------------------------------------

    def setup(self, args: argparse.Namespace) -> None:
        """[optional] Construct the real framework objects (agents, graphs,
        tool registries, system prompt, seeded history). Runs after the LLM
        client exists and before reset_phase_log()."""

    def after_reset(self, args: argparse.Namespace) -> None:
        """[optional] Recorded one-time work after reset_phase_log() /
        start_run (e.g. langchain's corpus build span)."""

    def round(self, i: int, args: argparse.Namespace) -> bool:
        """[required] ONE agent iteration, emitting exactly this framework's
        phase_sequence via roi() + record(). Return True to stop the loop
        (agentic-mode final_answer early exit)."""
        raise NotImplementedError

    #: Cap on ROUNDS spent on one task (a round = one LLM call plus the tool
    #: calls it triggers; see the vocabulary note in the module docstring).
    #:
    #: 20, not 6. The old 6 came from the fleet's TOOL CALLS PER ROUND
    #: (p50 1, p90 2, p99 5) -- the wrong layer. What this bounds is rounds
    #: per task. Measured on langchain against AppWorld task 3ab5b8b_1 with
    #: opus: one run emitted final_answer on round 4, another spent 6/6
    #: rounds re-reading API docs, because the history then showed the task
    #: text instead of the model's own code and cut results at 2000 chars.
    #: The history is faithful now; the cap is a safety bound, and rounds
    #: per task is reported per task by finish_task() rather than assumed.
    max_rounds_per_task = 20


    #: Frameworks set this once their round() is re-entrant, i.e. it can be
    #: called repeatedly within one round with accumulated tool results and
    #: returns True only when the model emits final_answer FOR THAT ROUND.
    #:
    #: Why this exists: with one LLM turn per round, a task needing two
    #: lookups can only ever record one tool. That is a second, independent
    #: cause of the degenerate tool mix -- grounding the task does not fix it,
    #: because the loop structure, not the task, is the binding constraint.
    #: Measured: a grounded two-part task recorded search_and_summarize on
    #: 5/5 rounds, exactly as the pre-grounding bundle did.
    multi_step_rounds = False

    #: Which exchange we are on inside the current round. Set by run_loop;
    #: declared here so a framework can read it even when called directly.
    round_step = 0

    # ---- AppWorld tool backend -------------------------------------------

    def open_appworld(self, args: argparse.Namespace):
        """Open the AppWorld session for this session, or return None.

        Shared by every framework: the tool surface, the task text and the
        answer check all come from AppWorld, so a framework only has to route
        its tool calls here instead of carrying its own grounded-task
        generator. Deterministic task choice from --seed keeps a bundle
        reproducible.
        """
        if getattr(args, "tool_backend", "synthetic") != "appworld":
            return None
        from agentcpu.tools.appworld_backend import (
            AppWorldSession, available_tasks)
        ids = available_tasks()
        if not ids:
            raise SystemExit("no loadable AppWorld tasks found; set "
                             "APPWORLD_ROOT or --appworld-task")
        # Concatenation: a session is --iterations tasks in a row with the
        # conversation history carried across task boundaries, which is the
        # structure the fleet shows (context never resets; DESIGN_V2 §4).
        # The task order is deterministic from --seed (or starts at
        # --appworld-task) and follows the sorted id list, so a scenario's
        # variants (xxxxxxx_1, _2, _3), which share a base database, tend to
        # run back to back.
        first = getattr(args, "appworld_task", None)
        start = ids.index(first) if first in ids else args.seed % len(ids)
        n = max(1, int(getattr(args, "iterations", 1) or 1))
        self.appworld_task_ids = [ids[(start + k) % len(ids)] for k in range(n)]
        self.appworld_baselines = []
        self.appworld = None
        return self.switch_appworld_task(0)

    def switch_appworld_task(self, i: int):
        """Close the current world and open task i of the session's list,
        keeping conversation history. Records the untouched-world verdict as
        the task's negative control and refreshes the tool schemas (apps
        differ per task) where the runner keeps them in self.llm_tools."""
        from agentcpu.tools.appworld_backend import AppWorldSession
        if getattr(self, "appworld", None) is not None:
            self.appworld.close()
        task_id = self.appworld_task_ids[i]
        session = AppWorldSession(task_id, experiment_name=self.name).open()
        self.appworld = session
        base = session.evaluate()
        self.appworld_baselines.append({"task_id": task_id,
                                        "success": base["success"],
                                        "pass_count": base.get("pass_count"),
                                        "total_count": base.get("total_count")})
        if i > 0 and isinstance(getattr(self, "llm_tools", None), list):
            self.llm_tools = self.appworld_tools()
        return session

    def finish_task(self, i: int, rounds_used: int,
                    final_answer: bool) -> None:
        """Called by the round loop when a task ends, either because the
        model emitted final_answer or because the round cap ran out. Records
        rounds used and, when a task source with its own completion check is
        open (AppWorld), that check's verdict. This is what makes "final_answer
        fired" and "task succeeded" two separate, reported facts."""
        rec = {"iteration": i, "rounds_used": rounds_used,
               "final_answer": final_answer,
               "cap_hit": (not final_answer
                           and rounds_used >= self.max_rounds_per_task)}
        s = getattr(self, "appworld", None)
        if s is not None:
            rec["task_id"] = s.task_id
            try:
                ev = s.evaluate()
                rec.update({"success": ev["success"],
                            "pass_count": ev.get("pass_count"),
                            "total_count": ev.get("total_count")})
            except Exception as e:                  # report, do not crash
                rec.update({"success": None,
                            "evaluate_error": f"{type(e).__name__}: {e}"})
        if not hasattr(self, "task_results"):
            self.task_results = []
        self.task_results.append(rec)

    def appworld_tools(self):
        """Tool schemas for the open session, derived from the world's own
        API listing."""
        from agentcpu.tools.appworld_backend import tool_schemas
        return tool_schemas(self.appworld)

    def appworld_call(self, tool_name: str, tool_args: dict) -> str:
        """Execute one model-chosen AppWorld tool call."""
        code = (tool_args or {}).get("code", "")
        if not code:
            return "no code supplied"
        try:
            return self.appworld.execute(code)
        except Exception as e:                       # surface, do not crash
            return f"{type(e).__name__}: {e}"

    def appworld_query(self, i: int, step: int) -> str:
        """The round's user message when the AppWorld backend is active.

        Step 0 states the world's task; later steps point the model at the API
        output already in history. Shared so every framework phrases the
        continuation the same way and none has to carry a task generator.
        """
        if step == 0 and i > 0:
            return (f"Next task: {self.appworld.instruction}\n(A fresh app "
                    f"environment; log in again where needed.)")
        if step == 0:
            return (
                f"{self.appworld.instruction}\n\n"
                "Environment notes: you act on behalf of the supervisor, whose "
                "profile and account passwords are available through the "
                "supervisor app (apis.supervisor.show_profile(), "
                "apis.supervisor.show_account_passwords()). Most app APIs "
                "need an access token from that app's login API. The api_docs "
                "app lists every app's APIs and their arguments. Each tool "
                "call runs Python against `apis`; print what you need to see. "
                "When the task is complete, call final_answer with the answer "
                "(a value if the task asks for one, otherwise a short "
                "confirmation).")
        return (f"Continue. Inspect the API output already in this "
                f"conversation and issue the next call, or call final_answer "
                f"if the task is complete.")

    def appworld_dispatch(self, i: int, tool_name: str, tool_args: dict,
                          roi_fn, record_fn) -> str:
        """Execute one model-chosen call against the world, timed and recorded.

        Factored out so the six remaining runners route tool calls with one
        line instead of repeating the timing/roi/record boilerplate -- and so
        they cannot diverge on the phase name the call is recorded under.
        """
        import time as _t
        t, t_cpu = _t.perf_counter_ns(), _t.thread_time_ns()
        with roi_fn("tool_exec"):
            out = self.appworld_call(tool_name, tool_args)
        record_fn(tool_name, t, {"iteration": i, "tool": tool_name,
                                 "output_chars": len(out)}, cpu_start_ns=t_cpu)
        return out

    def close_appworld(self) -> None:
        s = getattr(self, "appworld", None)
        if s is not None:
            s.close()
            self.appworld = None

    def run_loop(self, args: argparse.Namespace) -> None:
        """[optional hook] Drive the round loop. Default: the synchronous
        loop below. Frameworks whose original runner body was an async
        function override this to run the SAME loop under asyncio.run
        (autogen: its tool kernels and AssistantAgent.run are awaited).
        Overrides must preserve the early-exit contract."""
        for i in range(args.iterations):
            if self.multi_step_rounds:
                # An iteration is a task; the agent may take several rounds
                # (LLM call + the tool calls it triggers) to finish it.
                # final_answer ends the task, and the session continues to
                # the next one, as a deployed agent does between user turns.
                if i > 0 and getattr(self, "appworld", None) is not None:
                    self.switch_appworld_task(i)
                done = False
                rounds_used = 0
                for rnd in range(self.max_rounds_per_task):
                    self.round_step = rnd
                    rounds_used = rnd + 1
                    if self.round(i, args):
                        done = True
                        break
                self.finish_task(i, rounds_used, done)
            else:
                # Single-turn rounds: round() returns True when the LLM
                # emitted final_answer, and that ends the EPISODE (iteration
                # count is a cap, not a target).
                if self.round(i, args):
                    break

    def teardown(self, args: argparse.Namespace) -> None:
        """[optional hook] One-time cleanup after the round loop, BEFORE
        total_ms is computed and outputs are emitted -- matching originals
        that cleaned up before taking end_run (swe_agent removes its
        synthetic-codebase tempdir here)."""

    # ---- output emission --------------------------------------------------

    def run_metadata(self, args: argparse.Namespace,
                     mode_label: str) -> dict:
        """[optional] trace["run_metadata"]."""
        return {
            "framework": self.name,
            "context_size": args.context_size,
            "iterations": args.iterations,
        }

    def extra_summary_fields(self, args: argparse.Namespace) -> dict:
        """[optional] Extra top-level summary keys inserted after "mode"
        (langchain's back-compat "real_llm" alias lives here)."""
        return {}

    def config_dict(self, args: argparse.Namespace) -> dict:
        """[required] summary["config"]."""
        raise NotImplementedError

    def result_dict(self, args: argparse.Namespace,
                    total_ms: float) -> dict:
        """[required] summary["result"]."""
        raise NotImplementedError

    @staticmethod
    def replay_fields() -> dict:
        """Replay fidelity for the summary (recorded-request matching in the
        mock transport): requests served, misses, position of the first miss.
        Empty when the run did not replay a bundle (e.g. --real-llm)."""
        try:
            from agentcpu.llm import mock_server as _ms
            rs = _ms.replay_stats()
        except Exception:
            rs = None
        return {"replay": rs} if rs is not None else {}

    def final_print(self, summary_out: dict,
                    args: argparse.Namespace) -> None:
        """[optional] Final stdout report."""
        print(json.dumps({
            "summary": summary_out["result"],
            "run_id": self.run_id,
        }, indent=2))

    def completion_fields(self) -> dict:
        """Summary fields for task completion: per-task records from
        finish_task and the untouched-world baseline. Frameworks that emit
        their own summary schema (toolformer, chemcrow) merge these in, so no
        runner can silently drop the completion evidence."""
        out = {}
        if getattr(self, "task_results", None):
            out["tasks"] = self.task_results
        if getattr(self, "appworld_baselines", None):
            out["completion_baseline"] = self.appworld_baselines
            out["appworld_tasks"] = list(self.appworld_task_ids)
        return out

    def emit_outputs(self, args: argparse.Namespace,
                     total_ms: float) -> None:
        mode_label = "real_llm" if args.real_llm else "deterministic"
        trace = {
            "benchmark": "AgentCPU-Real",
            "workload": self.name,
            "run_id": self.run_id,
            "task_id": args.task_id,
            "seed": args.seed,
            "mode": mode_label,
            "run_metadata": self.run_metadata(args, mode_label),
            "spans": self.spans,
        }
        summary_out = {
            "benchmark": "AgentCPU-Real",
            "workload": self.name,
            "task_id": args.task_id,
            "seed": args.seed,
            "mode": mode_label,
        }
        summary_out.update(self.extra_summary_fields(args))
        summary_out["run_id"] = self.run_id
        summary_out["config"] = self.config_dict(args)
        summary_out["result"] = self.result_dict(args, total_ms)
        summary_out.update(self.completion_fields())
        summary_out.update(self.replay_fields())
        summary_out["trace_output"] = args.trace_output
        write_json(args.trace_output, trace)
        write_json(args.summary_output, summary_out)
        self.final_print(summary_out, args)

    # ---- the run skeleton ---------------------------------------------------

    def run(self, argv: list[str] | None = None) -> None:
        args = self.build_parser().parse_args(argv)

        # In-process determinism: pin native thread pools + seed RNGs so the
        # traced instruction stream is reproducible under simulation, not only
        # under the perf launch wrapper. Must run AFTER heavy imports (the
        # framework module imported them at module import time) -- see
        # core.common.enforce_determinism.
        from agentcpu.core.common import enforce_determinism
        enforce_determinism(getattr(args, "seed", None))

        self.args = args
        self.llm_client = self.make_llm_client(args)
        self.run_id = f"{self.name}__{args.task_id}__seed{args.seed}"
        self.rng = random.Random(args.seed)
        self.conversation_history: list[dict] = []
        self.spans: list[dict] = []
        # Every runner counts tool calls on the AppWorld dispatch and
        # final_answer paths; five of seven never initialised the counter
        # because their own setup() only did so on the synthetic path, and the
        # mock smoke never reaches final_answer. Measured 2026-09-18: haystack,
        # langgraph and toolformer crashed on the first real-LLM AppWorld round.
        self.tool_calls = 0

        self.setup(args)

        reset_phase_log()
        self.start_run = time.perf_counter_ns()
        self.after_reset(args)

        self.run_loop(args)

        self.teardown(args)
        end_run = time.perf_counter_ns()
        total_ms = (end_run - self.start_run) / 1_000_000.0
        self.emit_outputs(args, total_ms)

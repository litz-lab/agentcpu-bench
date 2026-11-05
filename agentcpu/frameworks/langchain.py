"""LangChain workload: Research assistant with LexRank summarization.

Scenario
--------
A research assistant agent that answers questions by searching a document
corpus and summarizing the results.  This mirrors the common LangChain
pattern of "Google Search -> retrieve top-N pages -> summarize with LLM".
The agent runs for *iterations* turns, each time formulating a query,
retrieving documents, and producing a LexRank summary.

Real-world analog: LangChain + Google Search + LexRank summarization
(reference paper Table 1).

Per-iteration phase order (canonical mode; behavior-preserving port of
AgentCPU/runners/run_langchain.py):

    api_build -> build_prompt -> tokenize -> tool_dispatch(plan) ->
    http_parse -> [retrieve -> tool_exec]{--n-tool-calls-per-iter}
    (tool_exec nests tfidf, cosine_pagerank) [+ tool_dispatch between
    repeated tool calls] -> api_parse -> context_update -> tokenize(decode)

Context-size scaling
--------------------
- **short**: 20 docs, 3-sentence summaries
- **medium**: 50 docs, 5-sentence summaries
- **long**: 150 docs, 8-sentence summaries
"""
from __future__ import annotations

import json
import random
import time

from langchain_core.tools import tool
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer

from agentcpu.core.common import (
    decode_tokens,
    realistic_system_prompt,
    tokenize_prompt,
)
from agentcpu.generator.grounded_tasks import GroundedCorpus
from agentcpu.generator.knobspec import (
    CONTEXT_TOKENS_CAP32000,
    IDENTITY_COUNT,
    ITERATIONS_ROUND5_CAP100,
    POW2_BYTES,
    KnobSpec,
)
from agentcpu.markers import roi
from agentcpu.registry import register
from agentcpu.runner import AbstractAgentRunner


# ---------------------------------------------------------------------------
# Context-size scaling parameters
# ---------------------------------------------------------------------------

SCALING = {
    "short":  {"n_docs": 20,  "summary_sentences": 3},
    "medium": {"n_docs": 50,  "summary_sentences": 5},
    "long":   {"n_docs": 150, "summary_sentences": 8},
}


# ---------------------------------------------------------------------------
# CPU-heavy tools
# ---------------------------------------------------------------------------

def search_and_retrieve(query: str, corpus: list[str], rng: random.Random,
                        n_results: int = 10) -> list[str]:
    """Retrieve docs for *query* from the seeded corpus.

    A grounded task names a specific document id, so exact-id hits must come
    back first -- otherwise the answer is not reachable through the tool and
    the task stops being answerable at all. Remaining slots are filled with
    seeded random docs, which keeps the retrieval-and-summarize work (and so
    the CPU shape) the same as before.
    """
    hits = [d for d in corpus if _id_in(d, query)]
    if len(hits) < n_results:
        pool = [d for d in corpus if d not in hits]
        hits += rng.sample(pool, min(n_results - len(hits), len(pool)))
    return hits[:n_results]


def _id_in(doc: str, query: str) -> bool:
    """True when the doc's bracketed id is mentioned in the query."""
    if not doc.startswith("["):
        return False
    return doc[1:doc.index("]")] in query if "]" in doc else False


def lexrank_summarize(text: str, sentence_count: int) -> str:
    """Real LexRank: TF-IDF, cosine similarity matrix, PageRank eigenvector.

    Inlined from sumy's LexRankSummarizer.__call__ so we can place
    sub-phase ROI markers around each step (TF-IDF, matrix construction,
    PageRank). The arithmetic is identical to summarizer(document, k) --
    we just call the same private methods with markers between them.
    """
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summarizer._ensure_dependencies_installed()

    document = parser.document
    sentences_words = [summarizer._to_words_set(s) for s in document.sentences]
    if not sentences_words:
        return ""

    with roi("tfidf"):
        tf_metrics = summarizer._compute_tf(sentences_words)
        idf_metrics = summarizer._compute_idf(sentences_words)

    with roi("cosine_pagerank"):
        matrix = summarizer._create_matrix(
            sentences_words, summarizer.threshold,
            tf_metrics, idf_metrics,
        )
        scores = summarizer.power_method(matrix, summarizer.epsilon)

    ratings = dict(zip(document.sentences, scores))
    summary = summarizer._get_best_sentences(
        document.sentences, sentence_count, ratings,
    )
    return " ".join(str(s) for s in summary)


@tool
def search_and_summarize(query: str) -> str:
    """Search corpus and summarize results with LexRank."""
    # This is a placeholder: actual logic is in the runner where we have
    # access to corpus and rng.  The @tool decorator is needed for LangChain
    # schema but we call the underlying functions directly.
    return query


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@register("langchain")
class LangChainRunner(AbstractAgentRunner):
    name = "langchain"
    default_iterations = 2
    #: A round is a grounded two-lookup task; the agent needs several
    #: LLM<->tool exchanges to finish it.
    multi_step_rounds = True
    # Canonical top-level roi() order per iteration (default knobs,
    # non-agentic); transcribed from run_langchain.py.
    phase_sequence = (
        "api_build", "build_prompt", "tokenize", "tool_dispatch",
        "http_parse", "retrieve", "tool_exec", "api_parse",
        "context_update", "tokenize",
    )
    nested_phases = {"tool_exec": ("tfidf", "cosine_pagerank")}
    # Declarative knob -> TraceLab-dimension mapping (drives the generator;
    # sweep_values transcribe the shipped bundle grid). Never read at run
    # time.
    knobs = (
        KnobSpec("--iterations", "rounds_per_session",
                 ITERATIONS_ROUND5_CAP100,
                 "agent rounds per session (fleet: rounds per session_id)",
                 sweep_values=(5, 15, 30, 100, 200)),
        KnobSpec("--tool-output-bytes", "tool_result_bytes", POW2_BYTES,
                 "per-tool-call result size seen by http_parse/api_parse "
                 "(fleet: tool_calls.result_chars)",
                 sweep_values=(256, 2048, 8192, 32768, 131072)),
        KnobSpec("--n-tool-calls-per-iter", "tools_per_round",
                 IDENTITY_COUNT,
                 "tool calls dispatched per round (fleet: tool calls per "
                 "round). Grid = p50 anchor 1 + upward bracket 3/6/20 "
                 "(coverage+bracket policy; fleet p90=2, p99=4 sit inside "
                 "the bracket)",
                 sweep_values=(1, 3, 6, 20)),
        KnobSpec("--seed-history-tokens", "context_tokens",
                 CONTEXT_TOKENS_CAP32000,
                 "seeded conversation context (fleet: "
                 "rounds.input_tokens_total). Fleet context (p50 ~132K "
                 "tokens) saturates the runner practical max of 32000; "
                 "0/2000/8000 are bracket coverage below the cap",
                 sweep_values=(0, 2000, 8000, 32000)),
        KnobSpec("--tool-list-size", None, None,
                 "structural: tool JSON schemas attached per request "
                 "(production agents expose 5-50+ tools); this TraceLab "
                 "dataset has no exposed-tool-count dimension",
                 sweep_values=(0, 20, 50, 100, 200)),
        KnobSpec("--system-prompt-tokens", None, None,
                 "structural: system prompt size bracketing production "
                 "4K-32K prompts; TraceLab's prefix_tokens conflates the "
                 "static system prompt with cached history, so this is "
                 "not fleet-derived",
                 sweep_values=(0, 8000, 32000)),
    )

    def add_arguments(self, parser) -> None:
        parser.add_argument("--seed-history-tokens", type=int, default=0,
                            help="Seed conversation_history with N tokens of "
                                 "synthetic prior turns. Pushes tokenize + "
                                 "build_prompt cycle weight up.")
        parser.add_argument("--tool-output-bytes", type=int, default=0,
                            help="Pad each summary to >=N bytes before "
                                 "passing to api_parse. Pushes api_parse + "
                                 "http_parse cycle weight up.")
        parser.add_argument("--n-tool-calls-per-iter", type=int, default=1,
                            help="Dispatch the retrieve+summarize tool block "
                                 "this many times per iteration. Pushes "
                                 "tool_dispatch cycle weight up.")

    # ---- construction -----------------------------------------------------

    def setup(self, args) -> None:
        # AppWorld backend, opened before anything else in setup: the
        # tool registry below is derived from the live session, and the
        # per-round task text comes from the world rather than from a
        # framework-specific grounded-task generator.
        self.appworld = None
        self.open_appworld(args)
        self.params = SCALING[args.context_size]
        self.tool_calls = 0
        self.last_summary = ""
        # Cached so we tokenize the same realistic prompt every iteration
        # rather than rebuilding it. Production agents reuse a fixed system
        # prompt across turns.
        self.system_prompt = realistic_system_prompt(
            role="research assistant",
            target_tokens=args.system_prompt_tokens,
        )
        # Tool list attached to every LLM request (production agents
        # expose 5-50+ tools; full JSON schema is serialized into the
        # request body and validated by pydantic on the response).
        if args.agentic_mode:
            # Force a tool registry for LLM-driven dispatch; 3 minimal tools
            # so the LLM has a real choice to make.
            self.llm_tools = self.appworld_tools() if self.appworld else [
                {"type": "function", "function": {
                    "name": "search_and_summarize",
                    "description": ("Retrieve documents from the corpus and "
                                    "produce a LexRank summary."),
                    "parameters": {"type": "object", "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "integer"},
                    }, "required": ["query"]},
                }},
                {"type": "function", "function": {
                    "name": "lookup_definition",
                    "description": ("Return a short definition for a "
                                    "technical term."),
                    "parameters": {"type": "object", "properties": {
                        "term": {"type": "string"},
                    }, "required": ["term"]},
                }},
                {"type": "function", "function": {
                    "name": "final_answer",
                    "description": ("Emit the final answer when no further "
                                    "tool calls are needed."),
                    "parameters": {"type": "object", "properties": {
                        "text": {"type": "string"},
                    }, "required": ["text"]},
                }},
            ]
        elif args.tool_list_size > 0:
            from agentcpu.core.common import realistic_tool_schemas
            self.llm_tools = realistic_tool_schemas(args.tool_list_size)
        else:
            self.llm_tools = None

        # Optional: seed conversation_history with synthetic prior turns so
        # tokenize/build_prompt see a long context from iteration 0. Each
        # synthetic turn is ~600 tokens of repeated GPT-2-friendly text.
        if args.seed_history_tokens > 0:
            approx_tokens_per_turn = 600
            n_seed_turns = max(
                1, args.seed_history_tokens // approx_tokens_per_turn)
            seed_text = (
                "Earlier turn context: prior research note discussing CPU "
                "microarchitecture, agent orchestration overhead, BPE token "
                "scaling with conversation length, and memory hierarchy. "
            ) * 80   # ~600 tokens
            for k in range(n_seed_turns):
                self.conversation_history.append(
                    {"role": "user", "content": f"Prior query {k}"})
                self.conversation_history.append(
                    {"role": "assistant", "content": seed_text})

    def after_reset(self, args) -> None:
        # The AppWorld backend brings its own world; building the synthetic
        # corpus there would burn setup cycles on data no tool ever reads and
        # would show up in the trace as work the session does not do.
        if self.appworld is not None:
            return
        # ---- Build corpus (one-time) ----
        t = time.perf_counter_ns()
        t_cpu = time.thread_time_ns()
        # Grounded corpus: every document carries a minted fact whose
        # id->value pairing is random per seed, and a glossary of minted
        # handles. This is what makes the per-round task unanswerable without
        # a tool call; the previous factless corpus left one generic question
        # per session, which is why 25 of 26 langchain bundles recorded a
        # single tool for the whole run.
        self.grounded = GroundedCorpus(
            seed=getattr(args, "seed", 0), n_docs=self.params["n_docs"])
        self.corpus = self.grounded.documents()
        self.glossary = self.grounded.glossary()
        self.record("build_corpus", t,
                    {"n_docs": len(self.corpus),
                     "n_terms": len(self.glossary), "grounded": True},
                    cpu_start_ns=t_cpu)

    # ---- one iteration ------------------------------------------------------

    def teardown(self, args) -> None:
        self.close_appworld()

    def round(self, i: int, args) -> bool:
        # Bulk up the per-turn user query so per-iter prompt size matches
        # production agents (typically 100-500 tokens of structured query
        # rather than a one-line synthetic query).
        # One grounded instance per round. Both referents are minted for
        # this run, so neither answer is reachable from parametric knowledge;
        # completing it needs search_and_summarize AND lookup_definition.
        #
        # Multi-step: the round is a TASK, not a single LLM turn. Step 0
        # states the task; later steps let the model read the previous tool
        # result out of conversation_history and call the next tool. With one
        # turn per round a two-part task could only ever record one tool --
        # measured as search_and_summarize on 5/5 rounds even with a grounded
        # prompt and a frontier model.
        step = getattr(self, "round_step", 0)
        if self.appworld is not None:
            # The world states the task (with AppWorld's conventions); later
            # steps point the model at the API output already in history.
            query = self.appworld_query(i, step)
        elif step == 0:
            self.task = self.grounded.task(i)
            query = self.task.prompt()
        else:
            query = (
                f"Continue round {i}. Consult the tool results already in "
                f"this conversation. If either field of the requested JSON "
                f"is still unknown, call the tool that provides it: "
                f"search_and_summarize for the document value, "
                f"lookup_definition for the glossary phrase. When both are "
                f"known, call final_answer with the completed JSON."
            )

        # ---- Build API request (orchestration overhead) ----
        t = time.perf_counter_ns()
        t_cpu = time.thread_time_ns()
        with roi("api_build"):
            # Build the kwargs dict that openai-python will serialize
            # via httpx inside the http_parse ROI below.
            _llm_messages = (
                [{"role": "system", "content": self.system_prompt}]
                + self.conversation_history
                + [{"role": "user", "content": query}]
            )
            _llm_model_name = (args.llm_model if args.real_llm
                               else "mock-llm")
            _llm_request_kwargs = {
                "model": _llm_model_name,
                "messages": _llm_messages,
                "max_tokens": 256,
            }
            if self.llm_tools is not None:
                _llm_request_kwargs["tools"] = self.llm_tools
            req_body = json.dumps(_llm_request_kwargs).encode("utf-8")
        self.record("api_request_build", t, {
            "iteration": i, "request_bytes": len(req_body),
            "history_len": len(self.conversation_history),
        }, cpu_start_ns=t_cpu)

        # ---- Tokenize prompt with growing history (real BPE) ----
        t = time.perf_counter_ns()
        t_cpu = time.thread_time_ns()
        with roi("build_prompt"):
            prompt_messages = [
                {"role": "system", "content": self.system_prompt},
            ] + self.conversation_history + [
                {"role": "user", "content": query},
            ]
        with roi("tokenize"):
            prompt_tokens = tokenize_prompt(prompt_messages)
        self.record("tokenize_prompt", t, {
            "iteration": i, "n_tokens": len(prompt_tokens),
            "n_messages": len(prompt_messages),
        }, cpu_start_ns=t_cpu)

        # ---- Plan ----
        t = time.perf_counter_ns()
        t_cpu = time.thread_time_ns()
        with roi("tool_dispatch"):
            plan = {
                "steps": ["search", "retrieve", "summarize", "synthesize"],
                "iteration": i,
            }
        self.record("plan", t, {"iteration": i}, cpu_start_ns=t_cpu)

        # ---- HTTP roundtrip (orchestration overhead) ----
        t = time.perf_counter_ns()
        t_cpu = time.thread_time_ns()
        _llm_response, http_result = self.call_llm(_llm_request_kwargs)
        self.record("http_roundtrip", t, {
            "iteration": i, "response_bytes": len(http_result),
        }, cpu_start_ns=t_cpu)

        # ---- Agentic-mode tool selection ------------------------------------
        # In agentic mode, the LLM's response.tool_calls drives which tool
        # the agent executes this iteration. In canonical mode (no agentic
        # flag), dispatch is deterministic (search+summarize always).
        agentic_tool, agentic_args = self.parse_agentic_tool_call(
            _llm_response)

        # ---- Search & retrieve + summarize, repeated --n-tool-calls-per-iter
        # times. In agentic mode, the inner loop runs once with the
        # LLM-chosen branch instead of the deterministic search+summarize.
        summary = ""
        # If LLM chose final_answer, treat as a no-op tool turn (still does
        # one accumulator append to record history grows). Other tools route
        # to their handlers below.
        if args.agentic_mode and self.appworld is not None and agentic_tool \
                and agentic_tool != "final_answer":
            t = time.perf_counter_ns()
            t_cpu = time.thread_time_ns()
            with roi("tool_exec"):
                summary = self.appworld_call(agentic_tool, agentic_args)
            self.tool_calls += 1
            self.record(agentic_tool, t,
                        {"iteration": i, "tool": agentic_tool,
                         "output_chars": len(summary)}, cpu_start_ns=t_cpu)
        elif args.agentic_mode and agentic_tool == "final_answer":
            self.tool_calls += 1
            self.record("final_answer", time.perf_counter_ns(),
                        {"iteration": i, "tool": "final_answer",
                         "answer_chars": len(agentic_args.get("text", ""))},
                        cpu_start_ns=time.thread_time_ns())
            summary = (agentic_args.get("text", "") or "")[:200]
        elif args.agentic_mode and agentic_tool == "lookup_definition":
            t = time.perf_counter_ns()
            t_cpu = time.thread_time_ns()
            with roi("lookup"):
                term = agentic_args.get("term", "")
                # Real glossary lookup. The old stub echoed the term back,
                # so calling the tool could never produce the ground truth
                # and the task could not be grounded. Exact id first, then a
                # scan, so a loosely-formed argument still resolves.
                entry = self.glossary.get(term)
                if entry is None:
                    entry = next((v for k, v in self.glossary.items()
                                  if k in term or term in v), None)
                summary = (entry or f"{term}: not found in glossary.")[:300]
            self.tool_calls += 1
            self.record("lookup_definition", t,
                        {"iteration": i, "tool": "lookup_definition",
                         "term": term}, cpu_start_ns=t_cpu)

        # The for-loop below runs the deterministic search+LexRank path. In
        # agentic mode, skip it if the LLM picked a different tool (already
        # handled above); run it when the LLM picked search_and_summarize
        # or no recognized tool.
        # With the AppWorld backend the model's call was already executed
        # against the world above, so the synthetic corpus path must not run:
        # otherwise a replayed response naming the old synthetic tool silently
        # re-enables it and the session measures the corpus, not the world.
        _skip_search_loop = (
            args.agentic_mode
            and (self.appworld is not None
                 or agentic_tool not in (None, "search_and_summarize")))
        for tc in range(args.n_tool_calls_per_iter):
            if _skip_search_loop:
                break
            t = time.perf_counter_ns()
            t_cpu = time.thread_time_ns()
            with roi("retrieve"):
                results = search_and_retrieve(query, self.corpus, self.rng,
                                              n_results=10)
            self.tool_calls += 1
            self.record("search_and_retrieve", t, {
                "iteration": i, "tool_call": tc, "n_results": len(results),
                "total_chars": sum(len(d) for d in results),
            }, cpu_start_ns=t_cpu)

            # ---- LexRank summarize (CPU-heavy) ----
            t = time.perf_counter_ns()
            t_cpu = time.thread_time_ns()
            with roi("tool_exec"):
                combined_text = " ".join(results)
                summary = lexrank_summarize(
                    combined_text, self.params["summary_sentences"]
                )
            self.tool_calls += 1
            self.record("lexrank_summarize", t, {
                "iteration": i, "tool_call": tc,
                "input_chars": len(combined_text),
                "summary_chars": len(summary),
                "sentence_count": self.params["summary_sentences"],
            }, cpu_start_ns=t_cpu)

            # Each tool dispatch beyond the first incurs another routing
            # round through the agent executor; emulate via a roi marker
            # only after the first tool call so we don't double-count
            # the existing tool_dispatch above.
            if tc + 1 < args.n_tool_calls_per_iter:
                with roi("tool_dispatch"):
                    next_plan = {
                        "next_step": "search_and_summarize",
                        "tool_call": tc + 1,
                        "iteration": i,
                    }
                    _ = json.dumps(next_plan)

        # Optional: pad summary to args.tool_output_bytes so api_parse +
        # http_parse see a realistic large tool output.
        if (args.tool_output_bytes > 0
                and len(summary) < args.tool_output_bytes):
            pad_unit = (
                "Additional tool output context: structured evidence list "
                "with document IDs, sentence offsets, retrieval scores, "
                "and provenance metadata for downstream verification. "
            )
            n_repeat = ((args.tool_output_bytes - len(summary))
                        // len(pad_unit) + 1)
            summary = summary + " " + (pad_unit * n_repeat)
        self.last_summary = summary

        # ---- Parse API response (orchestration overhead) ----
        t = time.perf_counter_ns()
        t_cpu = time.thread_time_ns()
        with roi("api_parse"):
            # Response was already pydantic-validated inside http_parse.
            # Serialize the validated model back to a dict and stitch in
            # the locally-computed tool kernel result so context_update
            # sees a consistent shape.
            resp_dict = _llm_response.model_dump(exclude_none=True)
            resp_dict["_tool_result"] = json.dumps(
                {"summary": summary, "status": "ok"}
            )
            parsed_resp = resp_dict
        self.record("api_response_parse", t, {
            "iteration": i,
            "response_id": str(parsed_resp.get("id", "")),
        }, cpu_start_ns=t_cpu)

        # ---- Accumulate conversation history ----
        # Label the history with the tool the model ACTUALLY called. This
        # used to be hardcoded to "search_and_summarize", so a
        # lookup_definition result was filed under the wrong tool: from the
        # model's view that tool had never returned anything, and it re-called
        # it until the step cap. Measured before the fix: final_answer emitted
        # 0 times in 30 exchanges, every round truncated at the cap, with both
        # answers already sitting in history by exchange 2.
        executed_tool = (agentic_tool if (args.agentic_mode and agentic_tool)
                         else "search_and_summarize")
        self.update_history(i, query, executed_tool, summary,
                            tool_args=agentic_args if agentic_tool else None,
                            with_cpu=True)

        # ---- Decode fake LLM response (real BPE) ----
        t = time.perf_counter_ns()
        t_cpu = time.thread_time_ns()
        with roi("tokenize"):
            fake_response_tokens = prompt_tokens[:50]  # simulate shorter output
            decoded = decode_tokens(fake_response_tokens)
        self.record("decode_response", t, {
            "iteration": i, "n_tokens": len(fake_response_tokens),
            "decoded_chars": len(decoded),
        }, cpu_start_ns=t_cpu)

        # ---- Synthesize ----
        t = time.perf_counter_ns()
        t_cpu = time.thread_time_ns()
        final_output = (
            f"[iter {i}] LexRank summary ({len(summary)} chars): "
            f"{summary[:200]}"
        )
        encoded = json.dumps({"query": query, "summary": final_output},
                             sort_keys=True)
        _ = json.loads(encoded)
        self.record("synthesize", t, {
            "iteration": i, "output_bytes": len(encoded),
        }, cpu_start_ns=t_cpu)

        # final_answer ends this ROUND under multi_step_rounds (the session
        # continues with the next task, as a deployed agent does between user
        # turns); under single-turn rounds it ends the EPISODE.
        return bool(args.agentic_mode and agentic_tool == "final_answer")

    # ---- output schema ---------------------------------------------------

    def run_metadata(self, args, mode_label: str) -> dict:
        return {
            "framework": "langchain",
            "context_size": args.context_size,
            "iterations": args.iterations,
            "mode": mode_label,
            "llm_model": args.llm_model if args.real_llm else None,
            "llm_base_url": args.llm_base_url if args.real_llm else None,
            "http2": (not args.no_http2) if not args.real_llm else None,
        }

    def extra_summary_fields(self, args) -> dict:
        # Back-compat alias used by existing aggregators
        return {"real_llm": args.real_llm}

    def config_dict(self, args) -> dict:
        return {
            "framework": "langchain",
            "context_size": args.context_size,
            "iterations": args.iterations,
            "n_docs": self.params["n_docs"],
            "summary_sentences": self.params["summary_sentences"],
        }

    def result_dict(self, args, total_ms: float) -> dict:
        return {
            "success": True,
            "latency_ms": total_ms,
            "tool_calls": self.tool_calls,
            "last_summary": self.last_summary[:300],
        }

    def final_print(self, summary_out: dict, args) -> None:
        print(json.dumps({
            "summary": summary_out["result"],
            "trace_path": args.trace_output,
            "run_id": self.run_id,
        }, indent=2))

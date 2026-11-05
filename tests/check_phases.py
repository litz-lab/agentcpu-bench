#!/usr/bin/env python3
"""Structural phase gate: the phase sequences declared by the runner
classes must match the canonical roi() order hardcoded here, maintained by
hand as an independent transcription (never derived from the runner code
it checks). The marker phase names are baked into libagent_markers.so and
the released traces -- renames are forbidden; additions only.

Requires the api runner's dependencies (sumy, langchain_core, numpy,
transformers); run it in the agentcpu image or after bundles/api/setup.sh
(source bundles/api/env.sh) if the local environment lacks them.
Exit codes: 0 ok, 1 mismatch, 3 missing deps.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Canonical hand-maintained tables -- do NOT derive these from the runner code.
EXPECTED = {
    "langchain": {
        # langchain main loop, default knobs, non-agentic:
        # api_build -> build_prompt -> tokenize -> tool_dispatch(plan) ->
        # http_parse -> retrieve -> tool_exec(lexrank) -> api_parse ->
        # context_update -> tokenize(decode)
        "top": ("api_build", "build_prompt", "tokenize", "tool_dispatch",
                "http_parse", "retrieve", "tool_exec", "api_parse",
                "context_update", "tokenize"),
        # lexrank_summarize sub-kernels inside roi("tool_exec")
        "nested": {"tool_exec": ("tfidf", "cosine_pagerank")},
    },
}

# Names baked into libagent_markers.so (agentcpu/markers/marker.c); every
# declared phase must be one of them. Additions to marker.c only, no renames.
KNOWN_MARKER_NAMES = {
    "api_build", "build_prompt", "tokenize", "http_parse", "api_parse",
    "context_update", "tool_dispatch", "tool_exec",
    "tfidf", "cosine_pagerank", "retrieve", "lookup",
    "numpy_svd", "scipy_stats", "numpy_fft",
    "sympy_solve", "sympy_diff_int", "multi_agent",
    "query_encode", "faiss_search",
    "numpy_sort", "numpy_compute", "file_grep", "patch_write",
    "file_read", "run_tests", "bash_simulate",
    "sympy_evaluate", "datetime_compute",
    "conformer_native", "reaction_enumerate", "smiles_search",
}


def main() -> int:
    try:
        from agentcpu.registry import get_runner
        runners = {name: get_runner(name) for name in EXPECTED}
    except ImportError as e:
        print(f"SKIP-DEPS: cannot import framework modules ({e}); "
              f"run inside the pinned container.")
        return 3

    failures = []
    for name, exp in EXPECTED.items():
        cls = runners[name]
        got_top = tuple(cls.phase_sequence)
        if got_top != exp["top"]:
            failures.append(
                f"{name}: phase_sequence mismatch\n"
                f"  expected {exp['top']}\n"
                f"  declared {got_top}")
        got_nested = {k: tuple(v) for k, v in cls.nested_phases.items()}
        exp_nested = {k: tuple(v) for k, v in exp["nested"].items()}
        if got_nested != exp_nested:
            failures.append(
                f"{name}: nested_phases mismatch\n"
                f"  expected {exp_nested}\n"
                f"  declared {got_nested}")
        unknown = [p for p in got_top for _ in [0]
                   if p not in KNOWN_MARKER_NAMES]
        unknown += [p for ps in got_nested.values() for p in ps
                    if p not in KNOWN_MARKER_NAMES]
        if unknown:
            failures.append(f"{name}: phase names not in the marker.c "
                            f"vocabulary: {unknown}")

    if failures:
        print("FAIL")
        for f in failures:
            print(f)
        return 1
    for name in EXPECTED:
        print(f"OK {name}: phase_sequence + nested_phases match the "
              f"canonical tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

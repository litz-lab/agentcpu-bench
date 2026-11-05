/* Phase-boundary markers recognized by Scarab.
 *
 * Scarab's dump_marker_type() (src/pin/pin_lib/x86_decoder.cc) recognizes:
 *   XCHG %rbx, %rbx  -> scarab_marker_roi_begin -> triggers reset_stats()
 *   XCHG %rdx, %rdx  -> scarab_marker_roi_end   -> triggers dump_stats()
 *                       with output suffix ".roi.<N>" where N auto-increments.
 *
 * One begin/end function pair per phase gives each phase a distinct link
 * address, so the marker PCs are unique per phase (useful for PC->name
 * resolution at post-processing time). The N-th begin/end pair scarab
 * encounters during simulation emits stats into <prefix>.roi.<N>.{csv,out};
 * we align the roi.N numbering with phase names via an execution-order
 * sidecar log written on the Python side (see agent_markers.py).
 *
 * Keep these tiny and unique per phase. Do NOT collapse bodies via macros
 * that would let the compiler fold them into a single function — we rely
 * on each symbol having a distinct link address.
 */

#define BEGIN_MARKER() __asm__ __volatile__("xchg %%rbx, %%rbx" ::: "rbx")
#define END_MARKER()   __asm__ __volatile__("xchg %%rdx, %%rdx" ::: "rdx")

/* Used noinline to prevent LTO from merging these into a single symbol. */
#define PHASE_PAIR(name)                                                   \
    __attribute__((noinline)) void agent_roi_begin_##name(void) {          \
        BEGIN_MARKER();                                                    \
        __asm__ __volatile__("" ::: "memory");                             \
    }                                                                      \
    __attribute__((noinline)) void agent_roi_end_##name(void) {            \
        END_MARKER();                                                      \
        __asm__ __volatile__("" ::: "memory");                             \
    }

/* Semantic phases to instrument across all agent workloads. Framework
 * wrappers pick the subset that applies (e.g., only swe_agent uses
 * file_grep / patch_write). */
PHASE_PAIR(build_prompt)       /* message-list + template assembly */
PHASE_PAIR(tokenize)           /* BPE encode via transformers */
PHASE_PAIR(api_build)          /* simulate_api_request: JSON serialize */
PHASE_PAIR(api_parse)          /* simulate_api_response_parse: gzip+JSON */
PHASE_PAIR(http_parse)         /* simulate_http_roundtrip: HTTP state machine */
PHASE_PAIR(tool_dispatch)      /* framework tool selection / routing */
PHASE_PAIR(tool_exec)          /* tool body execution */
PHASE_PAIR(context_update)     /* build_conversation_history: list append */
PHASE_PAIR(retrieve)           /* vector / BM25 / search over corpus */
PHASE_PAIR(file_grep)          /* swe_agent: regex scan over files */
PHASE_PAIR(patch_write)        /* swe_agent: regex substitute + write */
PHASE_PAIR(multi_agent)        /* autogen: message-passing loop */

/* tool_exec sub-phase decomposition: lets us classify operations across
 * frameworks by *operation type* rather than framework label. Each
 * sub-phase is wrapped *inside* roi("tool_exec") in the per-framework
 * runner, so the sub-phase markers are nested. The agent_markers.py
 * filter mode means perf only enables for the named sub-phase, so
 * counters can be scoped to one operation independently. */
PHASE_PAIR(tfidf)              /* langchain LexRank: TF-IDF sparse matrix construction */
PHASE_PAIR(cosine_pagerank)    /* langchain LexRank: cosine similarity + PageRank power iteration */
PHASE_PAIR(numpy_fft)          /* langgraph / swe_agent: numpy FFT */
PHASE_PAIR(numpy_svd)          /* langgraph: numpy SVD */
PHASE_PAIR(numpy_sort)         /* langgraph / swe_agent: numpy sort */
PHASE_PAIR(scipy_stats)        /* langgraph: scipy statistical operations */
PHASE_PAIR(numpy_compute)      /* swe_agent: composite numpy work (matmul + misc) */
PHASE_PAIR(sympy_solve)        /* autogen: sympy polynomial solving */
PHASE_PAIR(sympy_diff_int)     /* autogen: sympy differentiation + integration */
PHASE_PAIR(query_encode)       /* haystack: per-query transformer encoding */
PHASE_PAIR(faiss_search)       /* haystack: FAISS vector inner-product search + rerank */
PHASE_PAIR(bash_simulate)      /* swe_agent: framework bash/dispatch wrapper */

/* Sub-phases referenced by roi() in the runners that previously had no symbol
 * here, so agent_markers.py silently emitted no marker (the roi() call is a
 * no-op when the symbol is absent). Adding them makes the per-operation
 * decomposition real instead of silently folding into the coarse tool_exec
 * marker. */
PHASE_PAIR(sympy_evaluate)     /* toolformer: sympy expression evaluation */
PHASE_PAIR(datetime_compute)   /* toolformer: datetime arithmetic */
PHASE_PAIR(conformer_native)   /* chemcrow: conformer generation (RDKit / numpy LJ) */
PHASE_PAIR(reaction_enumerate) /* chemcrow: reaction enumeration */
PHASE_PAIR(smiles_search)      /* chemcrow: SMILES substring search */
PHASE_PAIR(file_read)          /* swe_agent: file read */
PHASE_PAIR(run_tests)          /* swe_agent: test execution */
PHASE_PAIR(lookup)             /* generic tool lookup / dispatch */

/* AppWorld anchor: real FastAPI TestClient -> pydantic v1 -> SQLModel -> SQLite.
 * The real-tools anchor's tool sub-phases (a third real tool type beyond
 * coding-agent shell/file and numeric kernels). */
PHASE_PAIR(appworld_db_read)   /* appworld: show_* SQLModel SELECT + pydantic serialize */
PHASE_PAIR(appworld_db_search) /* appworld: search_* filtered query + ranking */
PHASE_PAIR(appworld_auth)      /* appworld: login JWT + credential check */

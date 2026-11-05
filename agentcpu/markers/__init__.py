"""Phase-boundary markers for Scarab per-phase stat dumping AND for native
`perf stat --control fifo:` per-phase counter accumulation.

Two consumers, same begin/end markers:

1. Scarab (offline). libagent_markers.so emits XCHG %rbx,%rbx / XCHG %rdx,%rdx
   which scarab's memtrace frontend recognizes for per-ROI stat dumping. A
   sidecar phase log aligns each scarab `*.roi.N.csv` with a phase name.

2. perf stat (native HW). When AGENT_PERF_CTL_FIFO / AGENT_PERF_ACK_FIFO are
   set in the environment, the begin/end handlers also write `enable\\n` /
   `disable\\n` to a perf control FIFO and read back the ACK. Each begin/end
   pair becomes one perf stat block in perf's output, giving real-silicon PMU
   counts per phase invocation. Spawn perf with:
       perf stat --control fifo:<ctl>,<ack> --delay -1 -e <events> -- <cmd>
   `--delay -1` keeps perf paused until the first `enable`.

Usage:

    from agent_markers import roi

    with roi("tokenize"):
        tokens = tokenize_prompt(history)

Requirements:
- libagent_markers.so must be on LD_LIBRARY_PATH or at a known path.
- Scarab must already recognize XCHG %rbx,%rbx / XCHG %rdx,%rdx as the
  roi_begin / roi_end markers (existing behavior in x86_decoder.cc).
- Nesting markers is NOT supported by Scarab (it asserts !roi_dump_began on
  begin); this wrapper refuses to nest and emits a single flat begin/end pair.
"""
from __future__ import annotations

import atexit
import contextlib
import ctypes
import os

# Resolve the shared library. The Docker build installs it at this path; fall
# back to LD_LIBRARY_PATH search if the user runs the workload outside the
# container (e.g., for local development).
_LIB_PATHS = [p for p in (
    os.environ.get("AGENT_MARKERS_LIB"),     # set by bundles/api/env.sh
    "/tmp_home/libagent_markers.so",
    "libagent_markers.so",
) if p]

_lib = None
for _p in _LIB_PATHS:
    try:
        _lib = ctypes.CDLL(_p)
        break
    except OSError:
        continue

_PHASE_LOG_PATH = os.environ.get("AGENT_PHASE_LOG",
                                 "/tmp/agent_phase_log.txt")

# Optional perf stat --control fifo:<ctl>,<ack> integration. When the env
# vars are set, every roi() begin sends "enable\n" to the ctl fifo and reads
# the ACK; every end sends "disable\n" and reads the ACK. perf accumulates
# counters only between matched enable/disable pairs.
#
# Note: perf stat reports ONE aggregated summary at end-of-process, not one
# block per enable/disable cycle. To get per-phase totals, set
# AGENT_PERF_FILTER_PHASE=<name>; only roi() blocks for that phase will
# signal perf, so the summary covers a single phase only. Run the workload
# once per phase name with different filters set.
_PERF_CTL_FIFO = os.environ.get("AGENT_PERF_CTL_FIFO", "")
_PERF_ACK_FIFO = os.environ.get("AGENT_PERF_ACK_FIFO", "")
_PERF_FILTER_PHASE = os.environ.get("AGENT_PERF_FILTER_PHASE", "")
_perf_ctl_fp = None  # opened lazily on first roi() call
_perf_ack_fp = None


def _perf_open_lazy() -> None:
    """Open the perf control FIFOs on first use.

    Opening blocks until perf has the other end open, so we defer until the
    first marker fires (after the workload has done its imports, by which
    time perf is definitely up).
    """
    global _perf_ctl_fp, _perf_ack_fp
    if _perf_ctl_fp is not None or not _PERF_CTL_FIFO:
        return
    try:
        # Line-buffered text writes for ctl, byte reads for ack
        _perf_ctl_fp = open(_PERF_CTL_FIFO, "w", buffering=1)
        if _PERF_ACK_FIFO:
            _perf_ack_fp = open(_PERF_ACK_FIFO, "rb")
    except OSError:
        _perf_ctl_fp = None
        _perf_ack_fp = None


def _perf_signal(cmd: str) -> None:
    if _perf_ctl_fp is None:
        return
    try:
        _perf_ctl_fp.write(cmd + "\n")
        _perf_ctl_fp.flush()
        if _perf_ack_fp is not None:
            # perf replies with "ack\n" — drain one line so we stay synced
            _perf_ack_fp.readline()
    except (OSError, ValueError):
        pass

# Accumulate events in memory and flush once at interpreter shutdown. The
# previous per-event open+write+close added hundreds of millions of extra
# instructions over a 12-iteration run (~60x workload inflation) which
# contaminated SimPoint fingerprinting. Buffering removes that from the
# measured execution — only the ctypes call + XCHG marker remain in the
# hot path.
_events: list[tuple[str, str]] = []
_depth = 0
_flushed = False


def _flush_events() -> None:
    global _flushed
    if _flushed:
        return
    _flushed = True
    try:
        # Best-effort: ensure parent dir exists so callers can route the
        # phase log to a persistent location (e.g., next to --trace-output)
        # without having to mkdir themselves.
        parent = os.path.dirname(_PHASE_LOG_PATH)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(_PHASE_LOG_PATH, "w") as f:
            for ev, phase in _events:
                f.write(f"{ev} {phase}\n")
    except OSError:
        pass


atexit.register(_flush_events)


@contextlib.contextmanager
def roi(phase: str):
    """Emit a begin/end marker pair around the wrapped block.

    Nesting policy:
      * XCHG markers (consumed by Scarab's memtrace frontend) DO NOT nest:
        scarab calls reset_stats()/dump_stats() per begin/end and has no
        notion of pause/resume, so we emit XCHG only at the outermost ROI.
      * perf-stat signals MAY fire inside a nested ROI iff the nested
        phase name matches AGENT_PERF_FILTER_PHASE. Because we always run
        perf in per-phase filter mode (one perf invocation per phase),
        nesting cannot cause double-accumulation: only the filtered phase
        ever fires enable/disable, and within one workload run that phase
        is either always the outer phase OR always the inner phase, never
        both. This lets us measure inner phases (e.g., langgraph's
        tokenize call inside agent.invoke's tool_exec) that would
        otherwise be hidden under the outer marker.
    """
    global _depth
    if _lib is None:
        # Library missing (local dev) — no-op.
        yield
        return

    begin_fn = getattr(_lib, f"agent_roi_begin_{phase}", None)
    end_fn   = getattr(_lib, f"agent_roi_end_{phase}", None)
    if begin_fn is None or end_fn is None:
        # Unknown phase name — skip silently. Add the phase to marker.c to fix.
        yield
        return

    is_outermost = (_depth == 0)
    # perf signals fire when:
    #   * filter mode (typical for the paper): inner phase matches filter
    #     — even when nested inside a non-matching outer phase
    #   * unfiltered mode: only the outermost ROI fires, so a nested
    #     disable does not prematurely stop the outer phase's accumulation
    if _PERF_FILTER_PHASE:
        perf_active = (_PERF_FILTER_PHASE == phase)
    else:
        perf_active = is_outermost

    _depth += 1
    _events.append(("begin", phase))
    if perf_active:
        _perf_open_lazy()
        _perf_signal("enable")
    if is_outermost:
        begin_fn()
    try:
        yield
    finally:
        if is_outermost:
            end_fn()
        if perf_active:
            _perf_signal("disable")
        _events.append(("end", phase))
        _depth -= 1


def reset_phase_log() -> None:
    """Mark the start of the measured run in the phase log.

    This used to `_events.clear()`, which silently broke trace<->log
    alignment for any runner that opens an ROI during setup(): the XCHG
    markers are already in the instruction trace, but their log entries were
    thrown away, so the log came up short and every phase name after the
    first was attributed to the wrong marker. Measured on
    haystack_iters_15_p6c, whose setup() encodes the corpus under
    roi("tool_exec"): trace 121 begins vs log 120 outermost begins.

    Keeping the pre-reset events and writing a `reset` sentinel instead
    preserves positional alignment (the sentinel is not a begin/end, so
    consumers that only count markers are unaffected) while still telling
    consumers where the measured run starts.
    """
    _events.append(("reset", "-"))

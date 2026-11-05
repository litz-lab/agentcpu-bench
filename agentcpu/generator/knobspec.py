"""KnobSpec: the declarative link between a framework CLI knob and TraceLab.

Each framework runner declares a ``knobs`` classattr (a tuple of KnobSpec)
stating, for every swept CLI knob, WHICH TraceLab fleet dimension it tracks
and HOW a fleet percentile value becomes a knob value (the Policy). The
earlier fleet-calibration generator walked these declarations to emit a
knob grid; session length now comes from operating points (docs/ARCHITECTURE.md
section 4) and the declarations remain as the runners' CLI table.

Two kinds of knobs:

* **TraceLab-linked** (``tracelab_dim`` is a dimension name): the knob value
  at percentile p is ``value_fn(TraceLab.percentile(dim, p))``. The Policy
  encodes the HISTORICAL calibration rules (power-of-2 rounding for byte
  knobs, the intentional iteration cap, the seed-history practical max) so
  regenerating from TraceLab reproduces the shipped bundle grid exactly --
  the point is provenance, not renumbering.

* **structural** (``tracelab_dim is None``): swept by design to explore a
  framework-specific axis (RAG corpus size, multi-agent rounds, graph
  width); NOT fleet-derived, and flagged as such in knobs.json.

``sweep_values`` transcribes the historical shipped grid for that knob (the
bundle names under bundles/<framework>.<knob>=<value>): for structural knobs
this IS the grid; for TraceLab-linked knobs it additionally records the
bracket points swept around the fleet-derived anchors (coverage+bracket
policy: e.g. n-tools 1/3/6/20 brackets upward from the p50 anchor 1).

This module is purely declarative: nothing here changes runner behavior.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

__all__ = [
    "KnobSpec", "Policy",
    "IDENTITY_COUNT", "POW2_BYTES", "ITERATIONS_ROUND5_CAP100",
    "CONTEXT_TOKENS_CAP32000",
]


class Policy:
    """A named, documented percentile-value -> knob-value transform.

    The name + doc appear verbatim in knobs.json and in the calibration
    check table, so every deviation from the raw percentile is auditable.
    """

    def __init__(self, name: str, fn: Callable[[float], int], doc: str):
        self.name = name
        self.fn = fn
        self.doc = doc

    def __call__(self, percentile_value: float) -> int:
        return int(self.fn(percentile_value))

    def __repr__(self) -> str:  # pragma: no cover
        return f"Policy({self.name!r})"


def _nearest_pow2(x: float) -> int:
    if x <= 1:
        return 1
    return 2 ** round(math.log2(x))


#: Counts (tool calls per round): plain round to nearest integer, floor 1.
IDENTITY_COUNT = Policy(
    "round_int",
    lambda x: max(1, round(x)),
    "round to nearest integer (floor 1)",
)

#: Byte-sized knobs: round to the nearest power of 2 in log space.
#: Historical rule -- the shipped grid used power-of-2 byte sizes
#: (tool-bytes p90 8219 -> 8192, p99 40151 -> 32768).
POW2_BYTES = Policy(
    "pow2",
    _nearest_pow2,
    "round to nearest power of 2 (log space)",
)

#: Iterations: round to the nearest multiple of 5, then cap at 100.
#: The cap is the historical, INTENTIONAL choice: fleet p90 is ~145 rounds
#: and p99 ~1130, but sessions beyond ~100 iterations only repeat the
#: steady-state per-round behavior at unmanageable record/trace cost, so
#: the shipped grid topped out at iters=100 (langchain additionally swept
#: an exploratory 200 point). p90 and p99 therefore both map to 100.
ITERATIONS_ROUND5_CAP100 = Policy(
    "round5_cap100",
    lambda x: min(100, max(5, round(x / 5) * 5)),
    "round to nearest multiple of 5, cap at 100 (trace-budget cap: "
    "full-episode traces beyond 100 rounds are impractically large)",
)

#: Seed-history tokens: cap at the runner practical max of 32000.
#: Fleet context sizes (rounds.input_tokens_total: p50 ~132K, p99 ~856K)
#: exceed what the mock-transport runners can rebuild per request within
#: recording budgets, so this knob SATURATES at 32000 for every reported
#: percentile; the lower sweep points (0/2000/8000) are bracket coverage.
CONTEXT_TOKENS_CAP32000 = Policy(
    "cap32000",
    lambda x: min(32000, round(x)),
    "round to nearest integer, cap at 32000 (runner practical max)",
)


@dataclass(frozen=True)
class KnobSpec:
    """One CLI knob's calibration declaration.

    cli_flag      -- the argparse flag, e.g. "--iterations".
    tracelab_dim  -- TraceLab dimension name (see agentcpu.generator.tracelab
                     DIMS) or None for a structural (design-swept) knob.
    value_fn      -- Policy mapping a TraceLab percentile value to the knob
                     value; None for structural knobs.
    description   -- what the knob stresses / why it is structural.
    sweep_values  -- the historical shipped grid for this knob (transcribed
                     from bundles/<framework>.<knob>=<value>).
    """

    cli_flag: str
    tracelab_dim: str | None
    value_fn: Policy | None = None
    description: str = ""
    sweep_values: tuple[int, ...] = field(default=())

    @property
    def structural(self) -> bool:
        return self.tracelab_dim is None

    def knob_value(self, percentile_value: float) -> int:
        if self.structural:
            raise ValueError(
                f"{self.cli_flag} is structural (no TraceLab dimension)")
        assert self.value_fn is not None, (
            f"{self.cli_flag}: tracelab_dim set but no value_fn")
        return self.value_fn(percentile_value)

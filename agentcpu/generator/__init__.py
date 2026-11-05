"""Runner-side declarations used by the api runner.

- knobspec.py: KnobSpec, the declarative CLI-flag table each runner carries
  (the flags themselves; the fleet calibration that once read them is gone,
  session length comes from operating points, see docs/ARCHITECTURE.md 4).
- grounded_tasks.py: the task corpus the runner draws from when no
  external task source is given (with --tool-backend appworld it is unused).
"""
from agentcpu.generator.knobspec import KnobSpec, Policy

__all__ = ["KnobSpec", "Policy"]

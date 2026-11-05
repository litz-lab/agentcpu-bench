"""Framework runner subclasses.

Each module in this package defines one AbstractAgentRunner subclass and
registers it with @register("<name>"). Modules import their framework's
heavy dependencies at module import time -- this is intentional: the
determinism contract requires enforce_determinism() to run AFTER heavy
imports, and the registry imports these modules lazily (get_runner) so
`python -m agentcpu langchain` never imports langgraph's dependencies.
"""

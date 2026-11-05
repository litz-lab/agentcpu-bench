"""AgentCPU benchmark suite.

Layout:
    agentcpu.core        shared CPU substrate (tokenize, request build/parse,
                         http sim, history growth, determinism)
    agentcpu.markers     roi() phase markers (Scarab XCHG + perf-stat FIFO)
    agentcpu.llm         deterministic record/replay LLM transport
    agentcpu.runner      AbstractAgentRunner -- the framework contract
    agentcpu.registry    @register(name) / get_runner(name)
    agentcpu.frameworks  one thin runner subclass per agent framework

Entry point: python -m agentcpu <framework> --trace-output ... --summary-output ...
"""

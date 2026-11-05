"""Entry point: python -m agentcpu <framework> [runner args...]

Dispatches to the registered runner class for <framework>, passing the
remaining CLI arguments through to its (shared + framework-specific)
argparse. Example:

    python -m agentcpu langchain \
        --iterations 15 --trace-output out/trace.json \
        --summary-output out/summary.json
"""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        from agentcpu.registry import available
        print(__doc__.strip())
        print(f"\navailable frameworks: {', '.join(available())}")
        return 0 if argv else 2
    name = argv.pop(0)
    from agentcpu.registry import get_runner
    runner_cls = get_runner(name)
    runner_cls().run(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

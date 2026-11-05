"""Framework registry: name -> AbstractAgentRunner subclass.

A framework module registers itself with the decorator:

    @register("langchain")
    class LangChainRunner(AbstractAgentRunner):
        ...

and `get_runner("langchain")` lazily imports agentcpu.frameworks.langchain
(so each framework's heavy dependencies load only when that framework is
actually requested) and returns the registered class.
"""
from __future__ import annotations

import importlib
import pkgutil

_REGISTRY: dict[str, type] = {}


def register(name: str):
    """Class decorator: register a runner class under `name`."""
    def _decorator(cls):
        existing = _REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"framework name {name!r} already registered to "
                f"{existing.__module__}.{existing.__qualname__}"
            )
        _REGISTRY[name] = cls
        return cls
    return _decorator


def available() -> list[str]:
    """Names of all framework modules shipped in agentcpu.frameworks
    (without importing them)."""
    from agentcpu import frameworks
    return sorted(
        m.name for m in pkgutil.iter_modules(frameworks.__path__)
        if not m.name.startswith("_")
    )


def get_runner(name: str) -> type:
    """Return the runner class registered under `name`, importing its
    framework module on first use."""
    if name not in _REGISTRY:
        try:
            importlib.import_module(f"agentcpu.frameworks.{name}")
        except ModuleNotFoundError as e:
            # Distinguish "no such framework" from "framework exists but a
            # dependency is missing" -- the latter must propagate.
            if e.name == f"agentcpu.frameworks.{name}":
                raise SystemExit(
                    f"unknown framework {name!r}; available: "
                    f"{', '.join(available())}"
                ) from e
            raise
    try:
        return _REGISTRY[name]
    except KeyError:
        raise SystemExit(
            f"framework module agentcpu.frameworks.{name} imported but did "
            f"not register {name!r} (missing @register decorator?)"
        ) from None

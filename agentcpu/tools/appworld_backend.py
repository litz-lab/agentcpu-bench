"""AppWorld as the suite's real tool backend.

Why this replaces the synthetic tools
-------------------------------------
Agent frameworks ship tool *factories* (`@tool`, `FunctionTool`, `ToolNode`),
not tools. So the suite had to invent its own -- `compute_symbolic`,
`retrieve_and_rerank`, `search_and_summarize` -- and inventing the tools meant
inventing their grounding too. That went badly in a specific, repeating way:
a task written for one framework's tools is unanswerable with another's, and
"grounded" has to be re-derived per framework against whatever the tool
happens to do (string scan vs FAISS vector search vs sympy).

AppWorld removes the whole class of problem. It is a real agent benchmark: a
world of apps (spotify, gmail, venmo, ...) behind a real API stack
(FastAPI -> pydantic -> SQLModel -> SQLite), with 732 tasks, natural-language
instructions, and built-in answer checking. Point the frameworks at it and:

  * grounding is inherent -- "follow all classical artists with >=22
    followers" cannot be answered without querying the world;
  * verification is inherent -- AppWorld evaluates task completion itself,
    so no hand-built ground_truth dict;
  * the tools are real -- measured at 5.04 indirect/K on the auth path, MORE
    megamorphic than the synthetic orchestration it replaces;
  * every framework shares one tool surface, so the per-framework grounding
    generators are no longer needed.

Process boundary
----------------
AppWorld requires pydantic v1; the frameworks require v2. They cannot share an
interpreter, so AppWorld runs in its own venv as a subprocess speaking
line-delimited JSON on stdin/stdout. The boundary is a feature as well as a
necessity: tool execution lands in a separate process, which is exactly the
orchestration/tools split the phase analysis wants, and matches how a deployed
agent shells out to its tools.

Layout
------
    AppWorldSession   owns the subprocess, one task, call/close
    tool_schemas()    OpenAI function schemas derived from the world's own
                      API listing -- not hand-written
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Defaults: the layout bundles/api/setup.sh builds, relative to the repo root
# (the agentcpu image and bundles/api/env.sh set both variables).
APPWORLD_ROOT = Path(os.environ.get(
    "APPWORLD_ROOT", "bundles/api/appworld"))
APPWORLD_PYTHON = Path(os.environ.get(
    "APPWORLD_PYTHON", str(APPWORLD_ROOT / "venv/bin/python")))

# The worker runs inside the AppWorld venv. Kept as a string rather than a
# separate file so the backend is one self-contained module.
_WORKER = r'''
import json, os, sys
os.chdir(sys.argv[1])
from appworld import AppWorld

task_id = sys.argv[2]
world = AppWorld(task_id=task_id, experiment_name=sys.argv[3],
                 ground_truth_mode="full")
world.__enter__()
# Optional: freeze the loaded world so CPython's cyclic collector stops
# walking its object graph on every collection (10h: the collector was 32
# to 42% of the api session's cycles). Off by default; the switch is
# reported with the bundle because it changes the workload.
if os.environ.get("AGENT_APPWORLD_GC_FREEZE") == "1":
    import gc
    gc.collect(); gc.freeze()
    sys.stderr.write("appworld worker: gc.freeze after world load (%d objects)\n" % gc.get_freeze_count())

def reply(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

reply({"ok": True, "instruction": world.task.instruction,
       "apps": list(getattr(world.task.ground_truth, "required_apps", []) or [])})

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    op = req.get("op")
    try:
        if op == "close":
            reply({"ok": True}); break
        if op == "execute":
            out = world.execute(req["code"])
            reply({"ok": True, "output": str(out)})
        elif op == "api_docs":
            out = world.execute(
                "print(apis.api_docs.show_api_descriptions(app_name=%r))"
                % req["app"])
            reply({"ok": True, "output": str(out)})
        elif op == "evaluate":
            rep = world.evaluate()
            stats = {}
            try:
                stats = rep.to_dict(stats_only=True)
            except Exception:
                pass
            reply({"ok": True, "success": bool(getattr(rep, "success", False)),
                   "pass_count": getattr(rep, "pass_count", None),
                   "total_count": getattr(rep, "total_count", None),
                   "report": json.dumps(stats)[:2000]})
        else:
            reply({"ok": False, "error": "unknown op %r" % op})
    except Exception as e:
        reply({"ok": False, "error": "%s: %s" % (type(e).__name__, e)})

try:
    world.__exit__(None, None, None)
except Exception:
    pass
'''


class AppWorldError(RuntimeError):
    """Raised when the worker reports a failure or dies."""


@dataclass
class AppWorldSession:
    """One AppWorld task, held open across a session.

    Usage:
        with AppWorldSession("07b42fd_1") as w:
            w.instruction              # the natural-language task
            w.execute("print(apis.spotify.search_artists(...))")
            w.evaluate()["success"]
    """
    task_id: str
    experiment_name: str = "agentcpu"
    root: Path = APPWORLD_ROOT
    python: Path = APPWORLD_PYTHON
    timeout: float = 300.0

    def __post_init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self.instruction: str = ""
        self.apps: list[str] = []

    # ---- lifecycle -------------------------------------------------------

    def __enter__(self) -> "AppWorldSession":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def open(self) -> "AppWorldSession":
        if not self.python.exists():
            raise AppWorldError(
                f"AppWorld interpreter not found: {self.python}. Set "
                f"APPWORLD_PYTHON, or create the venv (pydantic v1 is "
                f"required and conflicts with the frameworks' v2).")
        if not (self.root / "data").is_dir():
            raise AppWorldError(
                f"APPWORLD_ROOT must contain ./data; got {self.root}")
        self._proc = subprocess.Popen(
            [str(self.python), "-c", _WORKER, str(self.root), self.task_id,
             self.experiment_name],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        hello = self._recv()
        self.instruction = hello.get("instruction", "")
        self.apps = hello.get("apps", [])
        return self

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            self._send({"op": "close"})
            self._proc.wait(timeout=30)
        except Exception:
            self._proc.kill()
        finally:
            self._proc = None

    # ---- transport -------------------------------------------------------

    def _send(self, obj: dict) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise AppWorldError("session is not open")
        self._proc.stdin.write(json.dumps(obj) + "\n")
        self._proc.stdin.flush()

    def _recv(self) -> dict:
        if self._proc is None or self._proc.stdout is None:
            raise AppWorldError("session is not open")
        line = self._proc.stdout.readline()
        if not line:
            err = (self._proc.stderr.read() if self._proc.stderr else "")[-600:]
            raise AppWorldError(f"AppWorld worker died. stderr:\n{err}")
        return json.loads(line)

    def _call(self, req: dict) -> dict:
        self._send(req)
        rep = self._recv()
        if not rep.get("ok"):
            raise AppWorldError(rep.get("error", "unknown worker error"))
        return rep

    # ---- operations ------------------------------------------------------

    def execute(self, code: str) -> str:
        """Run Python against the world's `apis` namespace; returns stdout."""
        return self._call({"op": "execute", "code": code})["output"]

    def api_docs(self, app: str) -> str:
        """The world's own API listing for one app."""
        return self._call({"op": "api_docs", "app": app})["output"]

    def evaluate(self) -> dict:
        """AppWorld's built-in completion check -- the reason no hand-built
        ground truth is needed."""
        rep = self._call({"op": "evaluate"})
        return {"success": rep["success"],
                "pass_count": rep.get("pass_count"),
                "total_count": rep.get("total_count"),
                "report": rep["report"]}


def tool_schemas(session: AppWorldSession) -> list[dict]:
    """OpenAI function schemas for the task's apps.

    One generic tool per app rather than one per endpoint: AppWorld's own
    contract is "write Python against `apis.<app>`", and enumerating every
    endpoint as a separate function would both explode the schema and diverge
    from how AppWorld is actually driven. Derived from the world's API listing,
    so the descriptions are the world's, not ours.
    """
    schemas = []
    # AppWorld's contract (its gold solutions do exactly this): credentials
    # come from apis.supervisor.show_account_passwords(), API listings from
    # apis.api_docs, and the task is closed with apis.supervisor.complete_task.
    # Exposing only the task's apps hid all of that: the model hit a 401 on
    # its first call and gave up after two rounds (measured 2026-09-17,
    # 3 of 4 tasks). So the two convention apps are always in the surface.
    apps = list(dict.fromkeys(list(session.apps or []) + ["supervisor", "api_docs"]))
    for app in apps:
        try:
            docs = session.api_docs(app)[:1500]
        except AppWorldError:
            docs = ""
        schemas.append({
            "type": "function",
            "function": {
                "name": f"{app}_api",
                "description": (
                    f"Call the {app} app's API by writing one or more lines of "
                    f"Python against `apis.{app}`. Print what you need to see. "
                    f"Available endpoints: {docs}"),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": (
                                "Python source, e.g. "
                                f"print(apis.{app}.login(username=..., password=...))"),
                        },
                    },
                    "required": ["code"],
                },
            },
        })
    schemas.append({
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Emit the final answer once the task is complete.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    })
    return schemas


def available_tasks(root: Path = APPWORLD_ROOT, limit: int | None = None) -> list[str]:
    """Task ids whose ground truth is complete enough to load.

    Not every downloaded task carries required_apps.json; loading one that
    does not raises inside AppWorld, so the suite selects from the complete
    set rather than discovering the gap at record time.
    """
    tasks = sorted(
        p.parent.parent.name
        for p in (root / "data" / "tasks").glob("*/ground_truth/required_apps.json")
    )
    return tasks[:limit] if limit else tasks


if __name__ == "__main__":
    ids = available_tasks(limit=5)
    print(f"{len(available_tasks())} loadable tasks; first five: {ids}")
    with AppWorldSession(ids[0], experiment_name="selftest") as w:
        print(f"\ntask {ids[0]}")
        print(f"  instruction : {w.instruction}")
        print(f"  apps        : {w.apps}")
        print(f"  tools       : {[t['function']['name'] for t in tool_schemas(w)]}")
        out = w.execute("print(apis.supervisor.show_profile())")
        print(f"  probe call  : {out[:120]}")

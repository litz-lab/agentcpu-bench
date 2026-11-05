# code-node: Claude Code sessions

The shipped Claude Code binary (`claude`, 2.1.274 for the released
sessions) on SWE-bench Verified tasks, with its real Bash / Read / Edit /
Write / Grep / Glob tools against a repository checkout, and only the model
replayed. Nothing in the product is patched: the binary is pointed at the
replay proxy through `ANTHROPIC_BASE_URL`.

The scenario is traced on 2.1.112, the last Node-based release, replaying
the session recorded on the Bun build 2.1.274; the Bun binary crashes under
DynamoRIO. `replay.sh` runs the pinned 2.1.274 from `install_products.sh`
by default; `CLAUDE_BUILD=2.1.112` selects the Node build, and `CLAUDE_BIN`
overrides both. Each replay prints the product version it saw in the
requests (`product cc_version=`).
Tool-result hashes differ by build because each build decorates tool
output differently: 52e6e3a5 on 2.1.274, e6c4e874 on 2.1.112.

## Files

* `ccproxy.py`: the Messages API record/replay proxy
  (`python3 ccproxy.py record|replay <port> <session.jsonl>`). Replay
  matches requests by (path, model) in recorded order and serves the
  recorded bytes; side-channel calls (count_tokens) are matched the same
  way. A request with no recorded partner is a miss.
* `tool_hash.py`: content hash of the tool results in a replay's request
  dump (`CCPROXY_DUMP`), with the normalisations of docs/ARCHITECTURE.md
  section 3 (pytest summary lines, parallel tool-result order).
* `rebase_session.py`: fills the task directory into a recorded session
  (recorded tool calls carry the placeholder `__TASK_DIR__`), and converts
  a fresh recording to the placeholder form.
* `install_products.sh`: installs both Claude Code builds, pinned, into
  `./products` (never a `claude` on PATH).
* `tasks/requests_5414/`: the bundle, one SWE-bench Verified task.
  `setup.sh` builds the task venv and checkout, `reset.sh` restores the
  checkout, `replay.sh [out]` replays and prints turns, misses, the
  tool-result hash and whether the produced patch equals `recorded.patch`;
  `meta.json` holds the product version, model, rounds, prompt tokens and
  the FAIL_TO_PASS verdict.

## Replay

```bash
cd tasks/requests_5414 && ./setup.sh      # once: venv, checkout at the base commit
./replay.sh /tmp/cc_out                   # fills its own directory into the recorded tool calls
```

`CC_HOME=<clean dir>` hides a machine's `~/.claude/settings.json` (its
`env` block can send the replay to a different endpoint); the dummy API
key in `replay.sh` satisfies the CLI's auth check without a login.

## Recording

```bash
python3 ccproxy.py record 8399 tasks/<task>/session.jsonl &   # forwards to api.anthropic.com
cd tasks/<task>/repo && CC_HOME=<clean dir> ANTHROPIC_BASE_URL=http://127.0.0.1:8399 \
  claude -p "$(cat ../prompt.txt)" --strict-mcp-config ...
python3 rebase_session.py tasks/<task>/session.jsonl tasks/<task>/session_opus5.jsonl "$PWD/tasks/<task>" __TASK_DIR__
```

A new task directory follows `tasks/requests_5414/`: `instance.json`
(the SWE-bench row), `prompt.txt`, `setup.sh`, `reset.sh`, `replay.sh`.

`--strict-mcp-config` keeps the account's MCP servers out of the session.
Replay fidelity for this product is content identity plus instruction
counts within a few percent (its JIT and GC are not deterministic); see
docs/ARCHITECTURE.md section 3.

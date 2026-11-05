#!/usr/bin/env bash
# Restore the task repo to the SWE-bench base commit (replay must start from the
# same tree the recording started from, at the same absolute path).
cd "$(dirname "$0")"
BASE=$(python3 -c "import json;print(json.load(open('instance.json'))['base_commit'])")
cd repo && git reset -q --hard "$BASE" && git clean -qfdx -e '*.egg-info' && git status --short | wc -l

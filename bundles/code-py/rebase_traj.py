#!/usr/bin/env python3
"""Write a copy of a recorded trajectory with the SWE-agent checkout path
filled in. The recorded files carry the placeholder __SWE_AGENT_DIR__ instead
of the recording host's path; SWE-agent validates tool-bundle paths against
the filesystem, so the copy must point at a real checkout (default: ./SWE-agent
next to this script). The content hash covers actions, observations and the
submission only, so this substitution does not change it.
Usage: rebase_traj.py <in.traj> <out.traj> [swe_agent_dir]"""
import os, sys
src, dst = sys.argv[1], sys.argv[2]
d = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "SWE-agent")
s = open(src).read().replace("__SWE_AGENT_DIR__", os.path.abspath(d))
os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
open(dst, "w").write(s)

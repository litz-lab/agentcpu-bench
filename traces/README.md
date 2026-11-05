# traces/ - pointer only

The released traces are not stored in this repository. They are registered
in scarab-infra (`workloads/workloads_db.json`, suite `agentcpu`), which
holds every workload's SimPoints, weights and replay command, and are
downloaded with

```bash
./sci --list-traces agentcpu     # what is available, with sizes
./sci --fetch-traces agentcpu    # download into trace_home
```

A simulation descriptor's `traces_dir` can also point at a directory that
already holds the traces. See docs/REPRODUCE.md section 4.

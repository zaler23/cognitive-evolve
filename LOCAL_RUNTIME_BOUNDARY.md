# Standalone Runtime Boundary

This project is now a standalone CognitiveEvolve source and runtime package. It does not require host-specific configuration, host plugins, or a private runtime mirror.

## Current State

The uploadable source project lives at:

```text
<source-root>
```

The default standalone runtime root is:

```text
~/.cognitive-evolve
```

Override it when needed:

```bash
export COGEV_RUNTIME_ROOT=/path/to/cognitive-evolve-runtime
export COGEV_TASKS_ROOT=$COGEV_RUNTIME_ROOT/.cogev/tasks
```

## Boundary Rules

- Do not symlink a host-specific runtime into the source project.
- Do not store task runs, Promptfoo databases, SkyDiscover venvs, node_modules, or provider caches in uploadable source files.
- Source files define the engine, specs, CLI, API, adapters, and tests.
- Runtime state belongs under `COGEV_RUNTIME_ROOT`, or under an explicit `COGEV_API_TASK_ROOT` for API requests.
- The runtime calls upstream models through `COGEV_LLM_*`; it does not delegate model access to any host app.

## Adapter Boundary

The source package is Nexus-only. External tools or providers may be called
through explicit adapters, but they are not local runtime backends and are not
selected as replacement execution authorities.

Allowed integration shape:

- LLM providers through `COGEV_LLM_*` and `llm.provider_interface`.
- Local verification tools through `tools/` adapters.
- Project files, tests, and artifacts through the Nexus snapshot and sandbox
  boundaries.

Disallowed uploadable-source shape:

- declaring LangGraph, DSPy, GEPA, MCP, Promptfoo, SkyDiscover, or similar tools
  as alternate runtimes;
- storing their venvs, databases, node_modules, caches, or run products in the
  source tree;
- adding a runtime selector that bypasses `NexusRuntime`.

## Verification

```bash
python3 scripts/cogev.py doctor --scope core
python3 scripts/cogev.py doctor --scope runtime
python3 scripts/cogev.py doctor --scope all
```

`--scope core` checks host-neutral source health. `--scope runtime` checks that standalone runtime entrypoints and host-specific removals are correct.

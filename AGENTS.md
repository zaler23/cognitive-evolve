# CognitiveEvolve Contributor Notes

This repository defines CognitiveEvolve `2.0.0`: a Nexus-only, host-agnostic, model-driven offline evolution runtime with an OpenAI-compatible API surface.

## Runtime source of truth

`cognitive_evolve_runtime.nexus.runtime.NexusRuntime` is the only execution authority. The active architecture has one candidate model (`CandidateGenome` / `ProjectCandidateGenome`), one ranking system (`ranking/`), one archive system (`archives/`), one runtime state schema (`nexus_runtime_state`), and one artifact root (`nexus-runtime/`).

Do not reintroduce:

- a runtime selector or alternate execution path;
- import wrapper modules for absent runtime namespaces;
- a second candidate loop;
- a second ranking authority;
- a second archive authority;
- npm/package.json as a Python control plane;
- hidden real-provider fallback in tests.

## Design boundary

The platform fixes evolution physics: input snapshots, hashes, lineage, candidate genomes, archives, local tool protocol, patch sandboxing, event logs, checkpoints, and replay boundaries.

The model decides task semantics: objective contracts, policy axes, niches, mutation plans, relative ratings, stagnation diagnosis, and synthesis rules.

## Validation

Before shipping changes, run:

```bash
python -m compileall -q cognitive_evolve_runtime
python -m pytest -q
```

Tests are hermetic by default through `COGEV_HERMETIC_TEST=1`; they must not read user-home `.env` files or require real API keys.

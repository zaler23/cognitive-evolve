# Architecture Boundaries

CognitiveEvolve 2.0 has one runtime architecture: Nexus.

## Allowed extension points

- `nexus/model_adapter.py` for structured model calls.
- `tools/` for local verifier adapters.
- `inputs/` for input packets, project snapshots, and context selection.
- `candidates/` for genome, mutation, crossover, and project patch structures.
- `archives/` for archive behavior.
- `ranking/` for relative rating, multihead scoring, and parent selection.
- `persistence/` for durable stores.
- `events/` for progress and event publication.
- `core/` for stable constants, paths, and redaction helpers only; it is not an
  alternate execution layer.
- `evolution/` for observation helpers such as drift/stagnation/novelty
  monitors only; Nexus loop and policy remain the decision authority.

## Disallowed patterns

- Runtime selectors that choose another engine.
- Wrapper modules for absent runtime namespaces.
- Duplicate archive or ranking packages.
- Line-count-only tests that force runtime control flow into artificial wrappers or mixins.
- Candidate dictionaries as the primary evolvable object.
- User-home `.env` loading in tests.
- Real provider fallback in hermetic tests.
- A package manager control plane outside `pyproject.toml`, `Makefile`, and `scripts/cogev.py`.
- Declaring external frameworks as runtime backends instead of adapter/tool
  integrations.

## Current artifact boundary

All runtime artifacts are written beneath `nexus-runtime/` and summarized in
`runtime-state.json` through `nexus/state.py`. Final snapshot files are
published with a `snapshot-transaction.json` manifest; append-only event logs
remain outside that snapshot and are de-duplicated by logical event identity.

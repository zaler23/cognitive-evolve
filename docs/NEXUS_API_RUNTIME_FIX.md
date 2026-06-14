# Nexus API Runtime Fix

## Problem

The Nexus-only cleanup removed the legacy runtime path before two legacy capabilities
were fully carried into the API-backed Nexus path:

1. API model tiers were written into request-local context, but `EngineOrchestrator`
   still defaulted to one round when no explicit `rounds` value was passed.
2. API calls created `NexusRuntime(model=None)`, so Nexus used deterministic
   fallback seeds/ranking/synthesis instead of the configured LLM-backed adapter.

That made `cognitive-evolve-one-shot-exhaustive` look like a direct answer mode:
it ran a one-round placeholder search and could return a static seed candidate.

## Fix

The architecture remains Nexus-only. Legacy runtime modules were not restored.
Instead, the missing capabilities were ported into the Nexus path.

- `nexus.budgeting.resolve_nexus_round_budget(...)` maps request-local model
  profiles and caps to adaptive Nexus budgets.
- All built-in API models now treat cap `0` as "adaptive policy". The profile
  resolves a safety checkpoint, branch factor, and stop policy; completion still
  requires a model/verifier stop signal.
- Explicit overrides are supported through `rounds`, `max_rounds`,
  `cogev_rounds`, `cognitive_evolve_rounds`, request metadata, and
  `COGEV_NEXUS_PROFILE_*_SAFETY_ROUNDS`. Legacy `COGEV_NEXUS_PROFILE_*_ROUNDS`
  is ignored by default so stale local `.env` files cannot silently pin adaptive
  runs; stale `*_CANDIDATES`, `COGEV_MUTATION_BRANCH_FACTOR`, and
  `COGEV_ACTIVE_POOL_LIMIT` entries are also ignored or warned instead of
  silently reshaping the adaptive profile.
- `EngineOrchestrator` binds API requests to
  `StructuredModelAdapter.from_configured_llm()` unless a test or caller
  injects a model adapter.
- The Nexus loop now supports repeated model seed generation with semantic
  dedupe, model-backed mutation planning, offspring generation, stop decisions,
  and final synthesis in addition to model-backed contract,
  policy, seed, ranking, and diagnosis calls.
- Runtime artifacts include `runtime_metadata.round_budget` and
  `runtime_metadata.model_backed` so local run directories expose whether the
  request actually used configured model-backed Nexus.

## Boundary

Deterministic fallback remains available for direct offline/tests where no model
adapter is supplied. It is not used silently for OpenAI-compatible API requests
when the service is configured with an LLM provider.

## Validation

```text
python -m compileall -q cognitive_evolve_runtime
python -m pytest -q
78 passed in 7.36s
```

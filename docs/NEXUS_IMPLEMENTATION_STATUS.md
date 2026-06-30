# Nexus Implementation Status

## Status

CognitiveEvolve now uses a single Nexus architecture. `NexusRuntime` is the only execution authority for CLI, API, runtime command, validation, artifacts, checkpoints, and progress events.

## Implemented

- `CandidateGenome` and `ProjectCandidateGenome` as structured evolvable individuals.
- `EvolutionPolicy` with task-defined niches, fitness axes, mutation operators, archive schema, rarity budget, and stagnation actions.
- `ArchiveManager` with answer, mechanism, novelty, rarity, failure, auxiliary, dormant, project patch, and quality-diversity views.
- `RelativeRater`, multihead Elo, novelty scoring, lineage saturation, and parent selection.
- Text input packets, project snapshots, project maps, context packets, patch sandboxing, tool feedback, and project candidate verification.
- Search diagnosis and policy update actions for auxiliary collapse, diversity collapse, knowledge bottlenecks, verification bottlenecks, rare injection, core extraction, and dormant reactivation.
- Population, archive, event, checkpoint, candidate-journal, round-snapshot, and verification-trace stores.
- Hermetic config loading, fake-model tests, explicit LLM provider interface adapters, and interruption-safe live persistence.


## API runtime binding fix

The OpenAI-compatible API now carries the selected CognitiveEvolve model tier into Nexus instead of relying on a one-round offline default. Built-in API model caps are `0`, meaning adaptive Nexus policy: the profile provides a safety checkpoint, branch factor, and minimum stop depth, while answer-first completion can return reviewable candidate output without project self-certification. Explicit request fields such as `rounds`, `max_rounds`, `cogev_rounds`, metadata overrides, `COGEV_NEXUS_PROFILE_*_SAFETY_ROUNDS`, `COGEV_NEXUS_MIN_CANDIDATES`, and `COGEV_NEXUS_BRANCH_FACTOR` remain available for local testing and operator control. Legacy `COGEV_NEXUS_PROFILE_*_ROUNDS`, `*_CANDIDATES`, `COGEV_MUTATION_BRANCH_FACTOR`, and `COGEV_ACTIVE_POOL_LIMIT` are ignored by default and surfaced as config warnings unless explicitly re-enabled where compatibility is supported.

API requests also bind `EngineOrchestrator` to `StructuredModelAdapter.from_configured_llm()` unless a test or caller supplies an explicit model adapter. Deterministic Nexus fallback remains available for direct offline/test use, but the API path no longer silently returns static seed candidates when the service is configured with an operator-supplied LLM provider through the transport layer.


## Exploration/recovery hardening

Nexus now absorbs the practical strengths of the old adaptive path without restoring a second runtime:

- narrow model seed pools are expanded through repeated model batches, semantic dedupe, and dynamic policy-derived candidate targets;
- supplemental candidates are marked as search seeds, not final answers;
- every round has a structured critique stage before mutation;
- API profiles carry both round depth and branch/candidate width;
- population, archive, candidate journal, round snapshots, and checkpoint are written during the loop;
- provider/quota/model interruptions produce an `error_checkpoint` and an interrupted result instead of losing candidates.

See `docs/NEXUS_EXPLORATION_RECOVERY.md`.

## Removed

- Alternate runtime modules and runtime selector paths.
- Duplicate archive/ranking packages.
- Wrapper modules for absent runtime architecture names.
- npm/package.json control plane.
- Orphaned legacy final-selection gate using Pareto/Elo terminology.

## Validation

Current validation result:

```text
python -m compileall -q cognitive_evolve_runtime
python -m pytest -q
81 passed in 17.83s
```

## Open risks

The structure is ready for richer tool adapters such as ruff, mypy, Z3, Lean, coverage, benchmark, and notebooks. These should be added under `tools/` without creating another runtime path. Future architecture tests should assert ownership and import boundaries, not brittle file line-count caps.

## Context compression update

Nexus now separates complete local state from model-facing prompt views. Full genomes, archives, journals, and checkpoints remain persisted locally, while ranking/critique/diagnosis/mutation/synthesis calls receive bounded summaries through `nexus.prompt_view`. The transport layer also applies a final prompt bound before provider calls. This fixes the exploration-recovery context growth bug where provider payloads could accidentally include full `population.json` plus repeated full `archives.json`.



## Legacy capability migration update

The remaining v1 capability surfaces have been moved into Nexus-native modules rather than restored as compatibility shims. Semantic routing/intake/capability hints now live in `nexus.semantics`; request-local budget context lives in `nexus.request_context`; post-run validation/eval/prompt optimization lives in `nexus.evaluation`; broad search-space coverage lives in `nexus.search_space`; the old verifier stack is now `nexus.project_verification`; provider inflight diagnostics live in `llm.inflight`.

The old top-level modules and subpackages are absent and tested as absent. See `docs/NEXUS_LEGACY_CAPABILITY_MIGRATION.md`.

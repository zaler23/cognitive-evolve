# Nexus Runtime Purity Audit

This audit records the current source-tree cleanliness rules for CognitiveEvolve 2.0.

## Current source of truth

- Runtime: `cognitive_evolve_runtime.nexus`
- Candidates: `cognitive_evolve_runtime.candidates`
- Archives: `cognitive_evolve_runtime.archives`
- Ranking: `cognitive_evolve_runtime.ranking`
- Contracts: `cognitive_evolve_runtime.contracts`
- Evidence: `cognitive_evolve_runtime.evidence`
- Inputs: `cognitive_evolve_runtime.inputs`
- Tools: `cognitive_evolve_runtime.tools`
- Persistence: `cognitive_evolve_runtime.persistence`
- Events: `cognitive_evolve_runtime.events`

## Cleanliness checks

- No alternate runtime package is present.
- No duplicate archive or ranking package is present.
- No wrapper module exists only to preserve an absent runtime namespace.
- Runtime artifacts use `nexus-runtime/`.
- Runtime state records `runtime_architecture: nexus` and `single_runtime.enforced: true`.
- CLI/API/runtime commands enter Nexus directly.
- Tests cover absent duplicate paths, direct canonical imports, hermetic config, archive fates, progress/checkpoint consistency, project patch sandboxing, structured candidate roundtrips, and structural Nexus boundaries without line-count-only pressure.

## Contributor rule

Treat attempts to add a second runtime, archive, ranking, contract, evidence, or candidate lifecycle owner as architecture regressions.

## 2026-05-27 follow-up audit

- Removed the remaining orphaned `selection_contracts.py` module because it preserved old Pareto/Elo terminology and was not used by Nexus.
- Replaced hard source-line limits in architecture tests with structural boundary checks.
- Updated stale capability/spec references from Pareto/Elo tournament artifacts to current Nexus archives, population, checkpoint, and relative-ranking artifacts.
- Added a provider interface layer for LLM calls while keeping LiteLLM as the default concrete provider.
- Confirmed no `candidate_search.py`, Mixin candidate orchestrator, `optimizer/`, `archive/`, or legacy runtime path remains.

Validation after this audit:

```text
python -m compileall -q cognitive_evolve_runtime
python -m pytest -q
78 passed in 7.36s
```

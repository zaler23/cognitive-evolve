# Nexus Legacy Capability Migration

This release keeps CognitiveEvolve Nexus-only while moving the useful v1 behavior into Nexus-native packages.  It does not restore a second adaptive runtime, compatibility shim package, or duplicate ranking/archive implementation.

## Migrated capabilities

| v1 capability | Nexus destination | Notes |
| --- | --- | --- |
| Semantic route / intake / capability hints | `nexus/semantics.py` | Produces `NexusRoute`, task assessment, one-shot enhanced task contract, capability hints, and current task artifacts. |
| Request-local round/profile/LLM-stage context | `nexus/request_context.py` | Replaces top-level request context and feeds API/runtime budget decisions. |
| Stage budget policy | `nexus/stage_budget.py` | Replaces the old global budget policy module. |
| Native eval / runtime validation / prompt optimization | `nexus/evaluation.py` | Single post-run Nexus artifact validation and prompt optimization path. |
| Search-space breadth contract | `nexus/search_space.py` | Keeps breadth pressure as a model-defined Nexus contract while avoiding hard-coded domain families or a separate search-space runtime. |
| Verifier stack | `tools/verification_stack.py` | Adds structured `ToolFeedback` to candidate genomes and prevents seed/auxiliary artifacts from silently becoming final answers. |
| In-flight provider status | `llm/inflight.py` | Keeps production diagnostics inside the LLM boundary. |
| Task type registry | `nexus/task_types.py` | Contract/evidence schemas import the Nexus canonical registry. |

## Removed paths

The following old paths are intentionally absent:

```text
cognitive_evolve_runtime.routing
cognitive_evolve_runtime.semantic_controller
cognitive_evolve_runtime.intake
cognitive_evolve_runtime.optimization
cognitive_evolve_runtime.native_eval
cognitive_evolve_runtime.runtime_validation
cognitive_evolve_runtime.llm_client
cognitive_evolve_runtime.request_context
cognitive_evolve_runtime.state_contract
cognitive_evolve_runtime.budget_policy
cognitive_evolve_runtime.capability_runtime
cognitive_evolve_runtime.search_space
cognitive_evolve_runtime.search_descriptors
cognitive_evolve_runtime.semantic_adapter
cognitive_evolve_runtime.textual_gradient
cognitive_evolve_runtime.verifier_stack
```

Tests assert these imports fail so the project does not drift back into a dual architecture.

## Current pipeline

```text
API / CLI
→ EngineOrchestrator or runtime_run
→ nexus.semantics semantic profile + task contract
→ nexus.budgeting / nexus.request_context round and width budget
→ NexusRuntime
→ CandidateGenome population
→ prompt-view-bounded model calls
→ critique + tools.verification_stack feedback
→ ranking + archives
→ checkpoint / live store / final synthesis
→ nexus.evaluation self-check
```

## Validation

```text
python -m compileall -q cognitive_evolve_runtime
python -m pytest -q
current suite should pass; use the repository test summary rather than this
historical document as the authoritative count.
```

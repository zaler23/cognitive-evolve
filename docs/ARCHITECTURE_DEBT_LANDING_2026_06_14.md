# Architecture Debt Landing — 2026-06-14

This note records the implementation pass based on the deep architecture audit that called out the `nexus/loop.py` god module, duplicated boundary helpers, budget construction drift, weak package exports, fate transition opacity, metadata sprawl, private doctor imports, and scattered score coercion.

## Landed changes

### 1. Nexus loop split

The old single file `cognitive_evolve_runtime/nexus/loop.py` was replaced by a compatibility package:

- `cognitive_evolve_runtime/nexus/loop/__init__.py`
- `cognitive_evolve_runtime/nexus/loop/budget.py`
- `cognitive_evolve_runtime/nexus/loop/round.py`
- `cognitive_evolve_runtime/nexus/loop/controller.py`
- `cognitive_evolve_runtime/nexus/loop/seeding.py`
- `cognitive_evolve_runtime/nexus/loop/closure.py`
- `cognitive_evolve_runtime/nexus/loop/policy_directives.py`
- `cognitive_evolve_runtime/nexus/loop/repair_guidance.py`
- `cognitive_evolve_runtime/nexus/loop/offspring.py`
- `cognitive_evolve_runtime/nexus/loop/stage_helpers.py`

Compatibility imports such as `from cognitive_evolve_runtime.nexus.loop import EvolutionBudget, evolve_once, seed_population` still work.  Test-only historical private imports used by the current suite remain re-exported where needed.

### 2. Shared Nexus boundary helpers

Added:

- `cognitive_evolve_runtime/nexus/_shared.py`
- `cognitive_evolve_runtime/core/scalars.py`

Centralized:

- `MODEL_BOUNDARY_ERRORS`
- positive integer parsing
- bounded score coercion
- classifier legacy fallback semantics

### 3. Budget factory

Added:

- `cognitive_evolve_runtime/nexus/budget_factory.py`

The CLI/runtime and engine orchestrator now share the same route-incomplete and round-budget-to-`EvolutionBudget` construction helpers.

### 4. Public package exports

Updated:

- `cognitive_evolve_runtime/nexus/__init__.py`
- `cognitive_evolve_runtime/evolution/__init__.py`

`nexus.__init__` now provides lazy public exports instead of a misleading `__all__` without imports, avoiding circular import side effects.

### 5. Candidate fate and metadata audit seams

Added:

- `cognitive_evolve_runtime/candidates/fate_machine.py`
- `cognitive_evolve_runtime/candidates/metadata_schema.py`

These introduce an auditable fate-transition machine and a centralized metadata-key registry/audit helper without breaking existing runtime call sites that still use `CandidateGenome.mark_fate`.

### 6. Doctor/private helper cleanup

Updated doctor orchestration to use public validation aliases rather than private underscore helpers.  Added public aliases while preserving existing private functions for compatibility.

### 7. Path compatibility after loop package split

Runtime repair/failure-classifier paths now treat the historical source binding `cognitive_evolve_runtime/nexus/loop.py` as satisfied by the new package entrypoint `cognitive_evolve_runtime/nexus/loop/__init__.py`, so older persisted repair material is not incorrectly made terminal by the refactor.

### 8. Fallback observability

Added `cognitive_evolve_runtime/nexus/fallbacks.py` and wired model-degraded deterministic fallback points in critique, relative ranking, and final synthesis to emit structured fallback log events.  Fallback remains allowed, but is no longer completely silent during review/debugging.

## Tests

Added:

- `tests/test_architecture_debt_refactor.py`

Verified:

```bash
python3 -m compileall -q cognitive_evolve_runtime tests
uv run --extra test pytest -q
# 506 passed, 1 skipped
```

## Explicitly not done

- No provider, prompt, UI/API, or M5/M6 proof-gate rewrite.
- No destructive migration of persisted run outputs.
- No Git commit/push/PR.
- No long self-evolve/model run.

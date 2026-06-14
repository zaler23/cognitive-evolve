# M6.2 Theory Integration Full Landing Plan

Date: 2026-06-08
Status: implementation-ready plan after six rounds of model discussion
Source artifacts: sanitized multi-model discussion summary; raw local review artifacts are not part of the public source tree.

## 0. Decision

Do **not** broadly refactor the project before M6.2.

Implement M6.2 as an additive, disabled-by-default, advisory-only theory layer. The only allowed preparatory refactor is a minimal seam for optional advisory ranking features, stable sidecar keying, isolated theory telemetry, and mechanical import-boundary tests.

M6.2 must not rewrite Nexus, providers, UI, genome schema, prompt views, M5 certificates, M6 gates, archive manager live paths, classification plumbing, pruning, or eligibility logic.

## 1. Non-negotiable boundaries

1. Theory outputs are advisory-only.
2. Theory objects must not encode or persist pass/fail, promotion, certificate, gate, proof, verdict, accepted, or rejected semantics.
3. M5 verified improvement certificates and M6 closure gates remain the only final proof gates.
4. Model-driven classification remains authoritative; no hardcoded domain routing.
5. Theory telemetry lives only under a separate advisory namespace.
6. Theory telemetry must never be embedded in M5 certificate payloads, M6 gate payloads, canonical candidate proof records, or prompt views in the first landing.
7. No M5/M6 module may import `cognitive_evolve_runtime.theory` or read theory telemetry.
8. `cognitive_evolve_runtime/theory/` may import only stable public contracts/dataclasses and standard library code. It must not import `nexus/`, ranking internals, evolution loop internals, archive manager live objects, or M5/M6 gate modules.
9. Theory reads only finalized immutable snapshots/representations, never live mutable handles or in-flight state.
10. Theory producers are pure and idempotent: same representation + config => same advisory features.
11. Theory producers cannot consume previous theory outputs as input.
12. Theory failures, timeouts, and cancellations return empty advisory features and never block runtime.
13. Disabled theory must be behavior-identical and artifact-identical for persisted outputs, prompts, ranking decisions, ledgers, archives, and certificates. It must not consume RNG, mutate candidates, alter timestamps, emit prompt text, or write telemetry unless explicitly excluded from a dedicated test.

## 2. Active scope

Active M6.2 implementation covers packages a-d only:

- M6.2a: foundation + inert runtime boundary.
- M6.2b: MDL producer.
- M6.2c: BOED / Active Inference pre-rank advisor.
- M6.2d: observer / sequential diagnostics.

Later packages are planned but not created as empty modules:

- M6.3 causal estimation.
- M6.4 archive geometry / optimal transport.
- M6.5 cellular field consuming advisory features.
- M6.6 bandit / budget allocation.
- M6.7 stability, viability, and type-contract diagnostics.

## 3. Files to add

### M6.2a foundation

Add:

- `cognitive_evolve_runtime/theory/__init__.py`
- `cognitive_evolve_runtime/theory/config.py`
- `cognitive_evolve_runtime/theory/errors.py`
- `cognitive_evolve_runtime/theory/signals.py`
- `cognitive_evolve_runtime/theory/representations.py`
- `cognitive_evolve_runtime/theory/telemetry.py`
- `cognitive_evolve_runtime/theory/aggregator.py`
- `cognitive_evolve_runtime/theory/layer.py`

Responsibilities:

- `__init__.py`: export only stable public symbols; no side-effect imports.
- `config.py`: frozen `TheoryConfig`; `enabled=False`; producer toggles false; weights `0.0`; per-producer timeout; total timeout; clamp bounds; optional cache bound.
- `errors.py`: `TheoryTimeout`, `TheoryCancelled`, `TheoryProducerError`; all caught inside theory layer.
- `signals.py`: `TheorySignal`, `AdvisoryRankingFeatures`, optional `TheoryObservation`, forbidden-key scanner, JSON-safety validator, finite-number validator, immutable meta handling.
- `representations.py`: immutable JSON-safe candidate/population/completed-event snapshots built from stable public fields only.
- `telemetry.py`: separate advisory telemetry namespace; must not write into certificate/gate/proof records.
- `aggregator.py`: deterministic order-invariant merge; finite/clamped values; idempotent aggregation.
- `layer.py`: `TheoryLayer`; lazy producer loading; disabled/error/timeout/cancel => empty features; no exception propagates to Nexus.

### M6.2b MDL

Add:

- `cognitive_evolve_runtime/theory/mdl.py`

Responsibilities:

- Deterministic description-length / complexity prior over `CandidateRepresentation`.
- Emits bounded `rank_prior` advisory values.
- Default weight is `0.0`.
- Cannot filter, prune, alter eligibility, or alter M5/M6 verdicts.

### M6.2c BOED / Active Inference

Add:

- `cognitive_evolve_runtime/theory/boed.py`

Responsibilities:

- Expected information gain / expected free energy advisory signal over immutable candidate/population representations.
- Emits bounded `plan_value` advisory values.
- Hard per-producer and total layer timeout.
- Cannot prune candidate set or remove full reachability.

### M6.2d observer / sequential diagnostics

Add:

- `cognitive_evolve_runtime/theory/observer.py`

Responsibilities:

- Consume completed immutable event snapshots only.
- Emit advisory diagnostics/risk observations to `theory/telemetry.py` only.
- Never wrap live outcomes.
- Never mutate outcome/certificate objects.
- Never feed M5/M6 gates or certificate payloads.

## 4. Existing files to modify

Modify only after inspecting exact current call signatures.

### Required narrow modifications

- `cognitive_evolve_runtime/ranking/parent_selection.py`
  - Add optional input such as:
    - `advisory_features: Mapping[str, AdvisoryRankingFeatures] | None = None`
  - Default `None` must preserve current behavior exactly.
  - Apply advisory features only after base score calculation as bounded additive/tiebreak values.
  - Advisory features must not change eligibility, pruning, final gate checks, or candidate set size.

- `cognitive_evolve_runtime/nexus/loop/`
  - Only if no narrower ranking adapter exists.
  - Construct immutable sidecar snapshots.
  - Call `TheoryLayer` before ranking only when config enables it.
  - Pass sidecar advisory features to the rank entrypoint.
  - Do not add prompt exposure in M6.2a-d.

### Optional minimal seam if missing

- A stable candidate-id helper if current candidate IDs are not stable across reconstruction.
  - Must not change genome schema.
  - Must not use `repr()` or object identity.
  - Prefer existing `CandidateGenome.id` or existing content-addressed candidate identity.

### Explicitly untouched in M6.2a-d

- `cognitive_evolve_runtime/candidates/genome.py`
- `cognitive_evolve_runtime/nexus/prompt_view.py`
- `cognitive_evolve_runtime/archives/manager.py` live mutation/access path
- M5/M6 certificate/gate/proof modules
- provider modules
- UI/API public response contracts except optional internal test-only telemetry if already isolated
- model-driven classification plumbing
- pruning and eligibility logic

## 5. Schema requirements

### `TheorySignal`

Required constraints:

- `source`: typed literal for active producers only in current package.
- `kind`: typed literal, e.g. `rank_prior`, `plan_value`, `risk`, `diversity`, `diagnostic`.
- `target_type`: typed literal, e.g. `candidate`, `lineage`, `population`, `outcome`, `plan`.
- `cycle_id`: non-empty stable string.
- `target_id`: non-empty stable string.
- `value`: finite Python `float`.
- `confidence`: finite Python `float`, clamped/validated in `[0.0, 1.0]`.
- `interval`: optional pair of finite floats with `low <= high`.
- `provenance`: bounded tuple of stable strings.
- `meta`: immutable, JSON-safe, bounded mapping with no live objects.
- `advisory_only`: exactly `True`.

Forbidden normalized structured keys, recursively in `meta` and diagnostics:

```text
pass
fail
passed
failed
promote
promotion
certified
verdict
gate
proof
certificate
cert_ref
certificate_id
gate_result
promotion_decision
accepted
rejected
```

Normalization must be case-insensitive and separator-insensitive, including variants like:

```text
GateResult
gate_result
gate-result
certificateId
promotion decision
```

Apply the forbidden-key scan to structured keys, not arbitrary natural-language text values unless values are nested structured maps.

### `AdvisoryRankingFeatures`

Required fields:

- `candidate_id: str`
- `rank_prior: float = 0.0`
- `plan_value: float = 0.0`
- `risk: float = 0.0`
- `diversity: float = 0.0`
- `provenance: tuple[str, ...] = ()`

Rules:

- All numeric values finite.
- Candidate ID must come from existing stable candidate identity.
- No fallback to object ID or repr.
- Aggregation owns clamping and normalization.
- Producers emit raw advisory signals; aggregator produces final bounded sidecar features.

## 6. Task order

1. Inspect actual rank entrypoint and Nexus ranking call site.
2. Add `theory/config.py` and `theory/errors.py`.
3. Add `theory/signals.py` with validation and forbidden-key scan.
4. Add `theory/representations.py` with immutable snapshots.
5. Add `theory/telemetry.py` advisory namespace.
6. Add `theory/aggregator.py` deterministic merge.
7. Add `theory/layer.py` with disabled/error/timeout/cancel empty fallback.
8. Add `theory/__init__.py` exports.
9. Wire inert M6.2a sidecar into the narrow ranking boundary.
10. Run M6.2a tests and full suite.
11. Add `theory/mdl.py` and ranking advisory weight path, weight `0.0` default.
12. Run M6.2b tests and full suite.
13. Add `theory/boed.py`, bounded and timeout-safe.
14. Run M6.2c tests and full suite.
15. Add `theory/observer.py` and isolated advisory telemetry for completed event snapshots only.
16. Run M6.2d tests and full suite.
17. Write implementation status note and stop before causal/transport/cellular/stability unless explicitly starting the next package.

## 7. Required tests

Add tests under `tests/theory/` unless an existing suite is a better fit.

### Architecture and boundaries

- `tests/theory/test_import_boundaries.py`
  - AST scan: `theory/` imports no `nexus`, ranking internals, evolution loop internals, archive manager live module, M5/M6 gate/proof modules.
  - M5/M6 modules import no `theory`.

- `tests/theory/test_disabled_path_golden.py`
  - Theory disabled preserves persisted outputs, prompts, ranking decisions, ledger/archive/certificate payloads.
  - Disabled path has no RNG/timestamp/prompt side effects.

- `tests/theory/test_weight_zero_equivalence.py`
  - Producers enabled but all weights zero preserve baseline ranking order and selected parents.

### Schema and serialization

- `tests/theory/test_signals_serialization.py`
  - JSON serialization without custom encoders.
  - Reject NaN/Inf.
  - Reject invalid confidence/interval.
  - Reject non-JSON meta and live object refs.

- `tests/theory/test_forbidden_keys.py`
  - Recursive scan covers nested dict/list payloads.
  - Case/separator variants rejected.

- `tests/theory/test_frozen_meta.py`
  - Metadata immutable after construction.

### Aggregation and producer safety

- `tests/theory/test_aggregator_determinism.py`
  - Shuffled inputs produce identical outputs.
  - Repeated aggregation idempotent.

- `tests/theory/test_theory_layer_failure_modes.py`
  - Producer exception returns empty features.
  - Producer timeout returns empty/partial features within wall-clock bound.
  - Cancellation leaves no corrupted partial state.
  - Total layer timeout enforced, not only per-producer timeout.

- `tests/theory/test_theory_layer_memory_bounds.py`
  - If cache exists, cache evicts and repeated new candidates do not grow unbounded memory.

### Runtime noninterference

- `tests/theory/test_advisory_noninterference.py`
  - Advisory features cannot reduce candidate set size.
  - Advisory features cannot change eligibility.
  - Extreme finite advisory values are clamped.
  - Random injected theory signals cannot change M5/M6 verdicts or proof payloads.
  - No theory fields appear in persisted certificate/gate records.

- `tests/theory/test_sidecar_identity.py`
  - Same candidate reconstructed from dict/checkpoint maps to same advisory key.

### Producers

- `tests/theory/test_mdl.py`
  - Deterministic output.
  - Added redundant structure increases or does not reduce description length.
  - Weight zero preserves baseline.

- `tests/theory/test_boed.py`
  - Deterministic bounded output.
  - Timeout/fallback behavior.
  - Does not prune candidates.

- `tests/theory/test_observer.py`
  - Observer consumes completed snapshots only.
  - Observer does not mutate outcome/certificate object identity or fields.
  - Observer writes only to advisory telemetry namespace.

## 8. Stop conditions

Stop M6.2 implementation immediately if any of the following occur:

1. Theory signal needs a pass/fail/certificate/gate/proof field to work.
2. Theory requires modifying `CandidateGenome` schema.
3. Theory requires changing prompt views in the active package.
4. Theory requires reading live archive manager objects rather than immutable snapshots.
5. Theory requires M5/M6 modules to import or read theory telemetry.
6. Disabled or zero-weight equivalence fails and cannot be isolated to non-semantic telemetry excluded from the test.
7. A theory producer can block runtime beyond configured total timeout.
8. Advisory features alter candidate eligibility, pruning, or final proof verdicts.
9. Full suite no longer passes.

## 9. Later packages

Do not create these as empty modules during M6.2a-d.

### M6.3 causal estimation

Goal: read-only intervention attribution from lineage/archive/completed-event snapshots.

Requirements:

- explicit non-identified state;
- no causal conclusion from raw correlation;
- no control effect in first causal landing.

### M6.4 archive geometry / optimal transport

Goal: distributional archive/population coverage metrics.

Prerequisite:

- stable descriptor contract;
- deterministic descriptors;
- no live archive manager access.

### M6.5 cellular field

Goal: graph cellular search field consuming advisory features.

Boundary:

- cellular field may consume sidecar advisory features through the standard interface;
- cellular field must not import theory producers;
- local cell pressure cannot become proof.

### M6.6 bandit / budget allocation

Goal: budget allocation suggestions over operators/cells/niches.

Boundary:

- first emits suggestions only;
- any real budget control needs separate kill-switch and reachability tests.

### M6.7 stability / viability / type-contract diagnostics

Goal: closed-loop stability and safety diagnostics after enough telemetry exists.

Boundary:

- diagnostic/advisory only;
- no automatic closure/gate decisions.

## 10. Verification commands

At minimum:

```bash
cd <repo-root>
./.venv/bin/python -m compileall -q cognitive_evolve_runtime tests
./.venv/bin/pytest tests/theory -q
./.venv/bin/pytest tests/test_m5_1_runtime_integration.py tests/test_m6_closure_gates.py tests/test_m6_full_closure_integration.py tests/test_generation_plan.py tests/test_nexus_adaptive_semantics.py tests/test_nexus_text_evolution_loop.py -q
./.venv/bin/pytest -q
```

Expected bar before handoff: full suite passes, with no reduction from the current baseline (`476 passed, 1 skipped` at planning time).

## 11. Model-discussion evidence

The public source tree keeps only this sanitized summary. Raw discussion exports, local review directories, provider-specific response filenames, and account-specific model routes are intentionally excluded.

Round 6 consensus: no broad refactor before M6.2; proceed additively with only minimal seams.

## 12. Implementation status — M6.2a-d first landing (2026-06-10)

Implemented the first additive M6.2 landing as a disabled-by-default, advisory-only theory layer.

### Added files

- `cognitive_evolve_runtime/theory/__init__.py`
- `cognitive_evolve_runtime/theory/config.py`
- `cognitive_evolve_runtime/theory/errors.py`
- `cognitive_evolve_runtime/theory/signals.py`
- `cognitive_evolve_runtime/theory/representations.py`
- `cognitive_evolve_runtime/theory/telemetry.py`
- `cognitive_evolve_runtime/theory/aggregator.py`
- `cognitive_evolve_runtime/theory/layer.py`
- `cognitive_evolve_runtime/theory/mdl.py`
- `cognitive_evolve_runtime/theory/boed.py`
- `cognitive_evolve_runtime/theory/observer.py`

### Modified files

- `cognitive_evolve_runtime/ranking/parent_selection.py`
  - Added optional `advisory_features` sidecar input.
  - Advisory values can reorder already-eligible parents but do not change eligibility, pruning, candidate set size, final gate, or verdict semantics.
- `cognitive_evolve_runtime/nexus/loop/`
  - Inspected the real reproduction parent-selection entrypoint before wiring.
  - Constructs immutable population representations only when `policy.metadata["theory"]` explicitly enables theory.
  - Calls `TheoryLayer` only in that enabled path.
  - Disabled/default path returns `{}` and does not write telemetry, mutate candidates, alter prompts, or touch proof/certificate payloads.

### Added tests

- `tests/theory/test_signals_serialization.py`
- `tests/theory/test_aggregator_and_layer.py`
- `tests/theory/test_representations_and_parent_selection.py`
- `tests/theory/test_import_boundaries.py`
- `tests/theory/test_producers.py`

### Boundary notes

- Theory producers import no Nexus/ranking/archive internals.
- M5/M6 certificate/gate/proof modules do not import `cognitive_evolve_runtime.theory`.
- Theory signals reject proof/gate/certificate/verdict/promotion/pass/fail structured keys recursively.
- Default config remains disabled and all producer weights default to `0.0`.
- MDL, BOED, and observer emit advisory signals only; they cannot filter, prune, promote, certify, or mark solved.
- No changes were made to `CandidateGenome`, prompt views, providers, UI/API public contracts, M5/M6 gate payloads, or live archive-manager mutation paths.

### Verification

Executed:

```bash
python3 -m compileall -q cognitive_evolve_runtime tests/theory
uv run --extra test pytest tests/theory tests/test_no_parent_repair_fallback.py tests/test_m5_1_runtime_integration.py tests/test_m6_closure_gates.py tests/test_m6_full_closure_integration.py tests/test_generation_plan.py tests/test_nexus_adaptive_semantics.py tests/test_nexus_text_evolution_loop.py -q
# 75 passed
uv run --extra test pytest -q
# 492 passed, 1 skipped
```

### Current status

M6.2a-d first landing is complete at the advisory-sidecar level.  The theory layer is now present, tested, and inert by default.  The next M6.2 follow-up should run a real enabled-policy dry test against a small deterministic population and then decide whether any advisory telemetry should be exposed in a dedicated non-proof diagnostic artifact.

## 13. Implementation status — M6.3-M6.7 advisory follow-up (2026-06-10)

User explicitly requested continuing the remaining follow-up work after M6.2a-d.  The later theory packages were therefore landed as additive, disabled-by-default, advisory-only modules.  This does **not** change the original proof boundary: theory proposes; M5/M6 proves.

### Added files

- `cognitive_evolve_runtime/theory/causal.py`
  - M6.3 non-identifying intervention-attribution advisory summaries over completed immutable event snapshots.
  - Emits diagnostics with `identified=False`, `confidence=0.0`, and no control effect.
- `cognitive_evolve_runtime/theory/geometry.py`
  - M6.4 deterministic population-geometry descriptors and diversity signals over immutable population representations.
  - Does not read live `ArchiveManager` state.
- `cognitive_evolve_runtime/theory/cellular.py`
  - M6.5 local search-cell summaries that consume immutable representations plus already aggregated advisory sidecar features.
  - Local cell pressure remains advisory and cannot become proof.
- `cognitive_evolve_runtime/theory/bandit.py`
  - M6.6 operator/cell/niche budget suggestions only.
  - Suggestions are validated as `advisory_only=True` and do not mutate runtime budgets.
- `cognitive_evolve_runtime/theory/stability.py`
  - M6.7 viability/type-contract-style population diagnostics.
  - Diagnostic-only; no closure/gate decisions.

### Modified files

- `cognitive_evolve_runtime/theory/signals.py`
  - Extended allowed advisory sources to `causal`, `geometry`, `cellular`, `bandit`, and `stability`.
  - Retained forbidden proof/gate/certificate/verdict/pass/fail metadata-key scan.
- `cognitive_evolve_runtime/theory/config.py`
  - Added opt-in producer toggles and weights for the later packages.
  - Defaults remain disabled and weight `0.0`.
- `cognitive_evolve_runtime/theory/layer.py`
  - Added opt-in helper methods for causal, cellular, bandit, and stability advisories.
  - Added geometry to the same candidate-sidecar aggregation path, but only when explicitly enabled in policy metadata.
  - Improved cache keying to include an immutable population fingerprint, not candidate IDs alone.
  - Failure/timeout/cancellation still fail closed to empty advisory outputs.
- `cognitive_evolve_runtime/theory/__init__.py`
  - Exported public stable symbols for the later packages.

### Added tests

- `tests/theory/test_later_advisory_packages.py`
  - Covers causal non-identification, geometry diversity, cellular sidecar consumption, bandit suggestions, stability diagnostics, and opt-in layer methods.
- `tests/theory/test_enabled_policy_dry_run.py`
  - Verifies an explicitly enabled theory policy can reorder already-eligible parents through the sidecar.
  - Verifies theory sidecar does not make Dormant candidates eligible and disabled policy returns `{}`.

### Boundary notes

- No broad refactor was performed.
- No changes were made to `CandidateGenome`, prompt views, providers, UI/API public contracts, M5/M6 certificates, M6 gates, or final verdict logic.
- The live Nexus loop still consumes only optional candidate advisory features for parent selection; later package methods are explicit opt-in helpers and are not hidden gates.
- Advisory outputs cannot encode or persist proof/gate/certificate/verdict/pass/fail semantics.
- Runtime proof remains certificate/gate based; no theory output can mark solved.

### Verification

Executed:

```bash
python3 -m compileall -q cognitive_evolve_runtime tests/theory
uv run --extra test pytest tests/theory -q
# 24 passed
uv run --extra test pytest tests/test_no_parent_repair_fallback.py tests/test_m5_1_runtime_integration.py tests/test_m6_closure_gates.py tests/test_m6_full_closure_integration.py tests/test_generation_plan.py tests/test_nexus_adaptive_semantics.py tests/test_nexus_text_evolution_loop.py -q
# 59 passed
uv run --extra test pytest -q
# 500 passed, 1 skipped
```

### Current status

M6.2a-d plus the advisory-only M6.3-M6.7 follow-up modules are implemented and verified.  The next safe step is not more theory-module scaffolding; it is a bounded real-run or deterministic integration experiment that measures whether enabled advisory features improve parent diversity/recovery without changing eligibility, proof, or final-gate semantics.

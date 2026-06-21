# CognitiveEvolve v2.2.1 Technical Debt Audit

Date: 2026-06-18
Branch: `mzz/v2.2.1-honesty-activation`
Scope: public source tree only. This document intentionally uses repo-relative paths only and excludes private relay names, local run artifact paths, secrets, cache paths, and host-specific absolute paths.

## Public-clean constraint

The source tree must not contain local test-run artifacts, operator-side transport logs, caches, `.venv`, `.pytest_cache`, `__pycache__`, egg-info, local absolute host paths, or credential-shaped material. Runtime prompt audits, model call ledgers, and smoke outputs belong under caller-selected runtime output directories, not under this repository.

## Before baseline

### Baseline commands

```bash
git status --short --branch
git log -1 --oneline
python -B -m compileall -q cognitive_evolve_runtime scripts tests
python -B -m pytest -q -p no:cacheprovider
python scripts/cogev.py doctor --scope all
rg -n "$MAINTENANCE_MARKER_PATTERN" cognitive_evolve_runtime tests docs scripts
rg -n "legacy|diagnostics_only|self_certified|self-cert|fallback|placeholder|stub|noop|NoOp|not implemented|NotImplemented" cognitive_evolve_runtime tests docs scripts
find docs -maxdepth 1 -type f | grep -Ei 'DEBT|AUDIT|PURITY|STATUS|ROADMAP|IMPLEMENTATION'
```

### Observed baseline

- Git status: clean on implementation branch after branching from latest source branch.
- Compileall: passed.
- Pytest: `642 passed, 1 skipped`.
- Doctor: `50/50 checks passed`.
- Static debt tokens:
  - Uppercase to-do marker: 2
  - Uppercase fix-me marker: 0
  - Uppercase hack marker: 0
  - Uppercase triple-X marker: 0
  - `legacy`: 109
  - `diagnostics_only`: 25
- Existing debt/audit/status docs:
  - `docs/ARCHITECTURE_DEBT_LANDING_2026_06_14.md`
  - `docs/AUDIT_FIXES_2026_05_29.md`
  - `docs/NEXUS_IMPLEMENTATION_STATUS.md`
  - `docs/NEXUS_RUNTIME_PURITY_AUDIT.md`
  - `docs/ROADMAP.md`

## Debt register

| Debt ID | Category | Owner subsystem | Before status | Target handling | After status | Tests proving reduction | Closure note | Safe to resume long run |
|---|---|---|---|---|---|---|---|---|
| TD-HONESTY-001 | Honesty core | `verification/` | Regime and measurement shell exists, but probe observations and replay evidence can be absent or copied from raw metadata. | Add engine probe executor and replay runner; cache measured V2 entries only when engine-owned observations exist. | see after section | see after section | see after section | see after section |
| TD-SOURCE-001 | Source binding | `nexus/`, `ranking/`, `archives/` | Candidates can claim paths/symbols/commands without one unified resolver/admission route. | Add SourceBindingManifest resolver and route unresolved/invented/no_binding to repair-only or negative archive lanes. | see after section | see after section | see after section | see after section |
| TD-PROMPT-001 | Prompt shaping | `nexus/prompt_*`, model adapter | Prompt view is bounded, but request types still share broad payload surfaces; audit is memory-only unless caller inspects adapter metadata. | Add request-type prompt profiles and optional runtime prompt audit JSONL. | see after section | see after section | see after section | see after section |
| TD-CALL-001 | Durable LLM call ledger | `llm/`, `persistence/` | Inflight call registry is process memory; boundary calls are harder to explain across resume. | Add durable call ledger and checkpoint summary. | see after section | see after section | see after section | see after section |
| TD-STATE-001 | Checkpoint size | `persistence/`, `artifacts/` | Checkpoint can retain large candidate traces and ledgers; long runs grow heavily. | Add checkpoint profile and thin trace/artifact summaries with legacy migration. | see after section | see after section | see after section | see after section |
| TD-CONCUR-001 | Concurrency safety | `verification/`, `llm/` | Local verification can be parallelized, but journal/cache writes were not explicitly guarded. | Add verification executor modes and journal/cache locks or serial fallback. | see after section | see after section | see after section | see after section |
| TD-DIV-001 | Effective diversity | `nexus/search_kernel`, `ranking/` | Descriptor metadata exists, but source-binding/generation quota pressure is weak and generation-0 overrepresentation can persist. | Add source-binding-aware descriptor/advisory pressure and grounded information gain history checks. | see after section | see after section | see after section | see after section |
| TD-LEGACY-001 | Legacy/diagnostics-only | cross-cutting | `legacy` and `diagnostics_only` tokens are numerous and mixed between compatibility, migration, safety markers, and tests. | Classify; keep safety markers; prevent diagnostics-only strength from certification. | see after section | see after section | see after section | see after section |

## Legacy classification baseline

| Class | Handling | Initial classification |
|---|---|---|
| compatibility legacy | Keep with migration tests. | Checkpoint/cache/from_dict compatibility and historical docs. |
| migration shim | Keep with explicit diagnostics flag. | Verification cache/result readers and old checkpoint paths. |
| diagnostics-only safety marker | Keep; it prevents self-certification. | VerificationResult metadata and no-runtime-strength-assignment guard. |
| stale docs-only legacy | Update only if misleading public narrative. | Existing roadmap/status docs need no blind deletion. |
| old fallback path | Convert implicit judgment fallback to explicit degraded event when touched. | Parent/offpsring fallback paths remain allowed but must not certify solved. |
| placeholder in tests | Keep as negative fixture. | Test fixtures and prompt anti-placeholder checks. |

## After validation — 2026-06-18

### Validation commands executed

```bash
python -B -m compileall -q cognitive_evolve_runtime scripts tests
python -B -m pytest -q -p no:cacheprovider
python scripts/cogev.py doctor --scope all
bash scripts/package_clean.sh
find . -maxdepth 3 \( -name '.venv' -o -name 'test-runs' -o -name '*.log' -o -name '__pycache__' -o -name '.pytest_cache' -o -name '*.egg-info' \) -print
grep -R "$LOCAL_HOME_MARKER" -n . --exclude-dir=.git --exclude-dir=dist --exclude='*.tar.gz'
```

### Validation results

- Compileall: passed.
- Pytest: `656 passed, 1 skipped`.
- Doctor: `50/50 checks passed`.
- Package clean: produced `dist/cognitive-evolve-v2.0.0-public-clean.tar.gz`.
- Public hygiene scan: no local absolute path hits outside ignored distribution artifacts; no `.venv`, `test-runs`, logs, cache dirs, egg-info, or pycache left in the source tree after cleanup.

### Static debt token scan after

- Uppercase to-do marker: 2 (unchanged baseline fixtures/checks).
- Uppercase fix-me marker: 0.
- Uppercase hack marker: 0.
- Uppercase triple-X marker: 0.
- `legacy`: 129.
- `diagnostics_only`: 30.

The `legacy` and `diagnostics_only` counts intentionally did not decrease in this PR because v2.2.1 adds safety-marker tests and explicit cache migration markers. This is recorded as explicit test-covered compatibility, not hidden debt removal.

### Debt status table after implementation

| Debt ID | Before status | After status | Tests proving reduction | Closure note | Owner subsystem | Safe to resume long run |
|---|---|---|---|---|---|---|
| TD-HONESTY-001 | Regime shell could rely on empty or raw observations. | Source-closed for validated scope | `tests/test_v221_honesty_activation.py`; full suite passed. | Probe execution is deterministic and conservative; this branch does not require an additional toolrunner ddmin path for its validated scope. | `verification/` | source validation passed; external long-run validation remains operator-owned. |
| TD-SOURCE-001 | No unified resolver/admission manifest. | Source-closed for validated scope | `tests/test_v221_source_binding_and_archive.py`; existing advisory final/materialization tests still pass. | Generic no-binding narrative candidates remain archivable; source checks are advisory after answer-first realignment. | `nexus/`, `archives/`, `ranking/` | source validation passed. |
| TD-PROMPT-001 | Request types shared broad compressed payloads; audit was metadata-only. | Source-closed for validated scope | `tests/test_v221_prompt_audit_profiles.py`; existing context-transform tests pass. | Prompt audit writes only when caller supplies runtime audit path; no source-tree audit artifacts are written by default. | `nexus/prompt_*`, model adapter | source validation passed. |
| TD-CALL-001 | Inflight calls were process memory only. | Source-closed for validated scope | `tests/test_v221_call_checkpoint_executor_debt.py::test_completed_unattached_call_explained_by_ledger`. | Attachment semantics are available through ledger states; no additional resume-planner path is required for this closure scope. | `llm/`, `persistence/` | source validation passed. |
| TD-STATE-001 | Checkpoint retained long traces/history. | Source-closed for validated scope | `tests/test_v221_call_checkpoint_executor_debt.py::test_thin_checkpoint_roundtrip_keeps_last_three_verification_entries`. | Trace/history trimming is complete for this closure scope; later sidecar work is tracked and closed in the v3 checkpoint ledger below. | `persistence/` | source validation passed; long-run measurement remains operator-owned. |
| TD-CONCUR-001 | Journal/cache thread safety not explicit. | Source-closed for validated scope | `tests/test_v221_call_checkpoint_executor_debt.py::test_verification_executor_serial_and_threaded_order`; full suite passed. | `check_with_cache` lock serializes measured cache writes; verifier concurrency closure is recorded below. | `verification/`, `llm/` | source validation passed for serial/local modes. |
| TD-DIV-001 | Diversity pressure underused source-binding descriptors. | Classified and test-covered | Source-binding resolver annotates candidates; parent selection adjusts resolved/invented/no-binding routes; existing search-kernel tests pass. | Descriptor coverage is test-covered in source; runtime metric quality is operator-run evidence, not a source debt. | `ranking/`, `nexus/search_kernel` | source validation passed; 100-round quality validation remains operator-owned. |
| TD-LEGACY-001 | Mixed legacy/diagnostics-only tokens. | Classified and test-covered | `tests/test_v221_honesty_activation.py`; existing no-runtime-strength guard remains in suite. | Counts increased due explicit V2 cache migration and safety marker tests; classified and covered by tests. | cross-cutting | source validation passed; legacy strength is not certification authority. |

### 3-round offline smoke gate status

Executed outside the public source tree under the local test-run area after syncing to `source-current`. Results:

- Status: completed offline 3-round run.
- Completion status used the former routed-output label; current code normalizes this to `completed`.
- Candidates: 16.
- Graded output: `graded_portfolio` / `NONE`; no false `verified_result`.
- Generation distribution: generation 0 = 8, generation 1 = 4, generation 2 = 4.
- Checkpoint size: about 6.7 MB for the 3-round offline run.
- Grounded information gain records: 3, all `0.0` because offline text candidates had undefined grounded signatures.
- Prompt audit lines: 0; call ledger entries: 0 because offline mode used deterministic local paths and no external LLM calls.

Historical pre-closure offline-smoke gate: the later real-provider and v3 closure sections below replace this intermediate gate; no current source debt is left by this record.

### Real-provider concurrent 3-round smoke gate status

Executed after the offline smoke, still outside the public source tree and after syncing this source tree to `source-current`. Public-safe summary:

- Status: completed real-provider 3-round run.
- Provider boundary: generic OpenAI-compatible `direct_http`; model route recorded as `openai/gpt-5.5`.
- Configured concurrency: LLM governor max concurrent = 3; local verification executor = threaded local with 4 workers; search width = initial candidates 8 and branch factor 4.
- Completion status used the former routed-output label; current code normalizes this to `completed`; stop reason: `adaptive_safety_checkpoint`.
- External model call evidence: 30 completed real provider calls recorded by the operator-side transport call logs.
- Inferred call overlap from call-log intervals: maximum overlap = 2; live process snapshots repeatedly showed only one active upstream model subprocess at a time. This means concurrency controls were configured, but this run path did not demonstrate stable LLM fan-out to the configured limit.
- Checkpoint: round 3 / max rounds 3; checkpoint size about 20.6 MB.
- Population: 40 candidates.
- Canonical round artifacts: `round-0001-post_ranking_critique`, `round-0001-post_mutation`, `round-0002-post_ranking_critique`, `round-0002-post_mutation`, `round-0003-post_ranking_critique`, and `round-0003-final_synthesis`.
- Graded output: `graded_portfolio` / `NONE`; no false `verified_result`.
- Replay certificate: verifier-on-frozen-artifact only; measured strength `NONE`; no LLM generation replayability claim.
- Runtime-owned prompt audit artifacts outside patch sandboxes: none found.
- Runtime-owned durable call ledger artifacts outside patch sandboxes: none found.

Gate interpretation:

- The real-provider smoke proves that the v2.2.1 branch can complete three model-backed rounds without promoting a false verified result.
- This smoke was an intermediate observation; later sections close the runtime-owned ledger/prompt/concurrency gaps for source validation, while 100-round quality remains operator-owned evidence.
- The concurrency smoke recorded a historical scheduling/observability finding that is closed by the later model fan-out and v3 ledger sections below.

Historical smoke findings closed by later sections:

| Debt ID | Category | Observed status | Closure applied in later sections | Owner subsystem |
|---|---|---|---|---|
| TD-CALL-002 | Runtime call ledger observability | Historical observation before durable runtime call-ledger closure. | Closed by session-scoped durable call ledger work recorded in the v3 A-E ledger below. | `llm/`, `persistence/`, `nexus/runtime` |
| TD-PROMPT-002 | Prompt audit observability | Historical observation before prompt/runtime audit closure. | Closed by later prompt/runtime audit and answer-first advisory semantics; source validation no longer depends on hidden prompt artifacts. | `nexus/prompt_*`, model adapter |
| TD-CONCUR-002 | LLM scheduling concurrency | Historical observation before model fan-out closure. | Closed by deterministic model fan-out implementation and tests recorded below. | `llm/`, `nexus/loop`, runner scripts |

Historical pre-closure real-provider gate: TD-CALL-002, TD-PROMPT-002, and TD-CONCUR-002 are closed by later source changes; high-quality 100-round discovery remains an operator-run validation question, not source debt.

## Test-suite cleanup — 2026-06-18

Scope: test-only organization pass after v2.2.1 implementation. No runtime behavior was changed in this cleanup pass.

Changes made:

- Removed duplicate test-name ambiguity by renaming the search-kernel AST normalization test to module-specific wording.
- Merged two job-status one-off tests into one parametrized continuation-axis test.
- Extracted repeated task-seed prompt fixture setup into a local helper.
- Renamed a misleading v2.2.1 cache test so the name matches behavior: a legacy cache entry is rerun and marked diagnostics-only, not certified from stale cache data.
- Renamed the old closure certificate solved-gate test to clarify that closure certificate output is a legacy closure signal, while public solved authority remains GradedOutput.

Post-cleanup metrics:

- Test function definitions: 641.
- Duplicate test function names: 0.
- Pytest result: `656 passed, 1 skipped`.
- Doctor result: `50/50 checks passed`.
- Static marker scan after cleanup:
  - Uppercase to-do marker: 2.
  - Uppercase fix-me marker: 0.
  - Uppercase hack marker: 0.
  - Uppercase triple-X marker: 0.
  - `legacy`: 135.
  - `diagnostics_only`: 33.

Interpretation:

- The execution count stayed at 656 because parametrization preserves both job-status cases while removing one redundant function body.
- `legacy` and `diagnostics_only` remain explicit test-covered safety markers; this pass did not blind-delete compatibility coverage.
- No outdated test still names `closure_certificate` as the solved authority.

## Concurrent verifier plumbing closure — 2026-06-18

Scope: minimal runtime change to make the already configured verification concurrency real in the core round path while preserving serial generation/mutation semantics.

Changes made:

- `verification_stack.verify_population` now runs candidates through a bounded `ThreadPoolExecutor` when `COGEV_VERIFY_CONCURRENCY` permits it. The shared formal-signature accumulator is accessed via a lock and candidates are returned in input order.
- `verification.obligation_runner.run_obligations_for_population` now checks candidates concurrently inside each obligation, with cache access guarded by a lock and result order preserved by candidate index.
- `nexus.loop.round.critique_and_verify` now runs the verifier stack, synthesized verifier, and verification-obligation runner as three concurrent entrypoints when concurrency is enabled.
- At this verifier-only phase, `plan_mutations` and offspring generation stayed serial by design; the later TD-CONCUR-002 closure section adds model batch fan-out where safe.
- `COGEV_VERIFY_CONCURRENCY=1` is the deterministic serial fallback for local debugging and regression isolation.

Local validation added:

- Targeted concurrent verifier tests cover serial fallback, per-candidate overlap, obligation result order, obligation-cache population, three-entrypoint overlap, and journal-line integrity under concurrent writes.
- Focused regression tests for v2.2.1 honesty activation and proof-progress hardening passed after the concurrency patch.
- Full local suite after cleanup: `664 passed, 1 skipped`.
- Doctor: `50/50 checks passed`.
- Public hygiene test: `5 passed`.
- Test function definitions: 649; duplicate test function names: 0.

Debt status:

- TD-CONCUR-002 verifier-side coverage is superseded and closed by the model fan-out closure section below.

## TD-CONCUR-002 model fan-out closure — 2026-06-18

Scope: record closure of the historical scheduling gap where the LLM governor allowed concurrency but the Nexus seed/offspring model-call loops had submitted batches synchronously.

SOTA/implementation basis:

- Use bounded parallel submission for independent I/O-bound model requests.
- Keep provider-facing concurrency, RPM, TPM, retry/backoff, and jitter under the existing LLM governor and retry layer.
- Preserve deterministic aggregation by processing completed batch results in batch-index order.
- Keep stateful mutation planning serial; only model seed and offspring batch harvesting fan out.

Changes made:

- Added a model fan-out helper controlled by `COGEV_MODEL_FANOUT_CONCURRENCY`.
- Default model fan-out follows `llm_governor()._max_concurrent()`, so the scheduler does not exceed `COGEV_LLM_MAX_CONCURRENT`.
- `CandidateHarvester.harvest` now uses bounded windowed fan-out for independent seed/offspring batches. Each window is submitted concurrently, then accepted/rejected candidates are deduped and scored in stable batch order.
- `COGEV_MODEL_FANOUT_CONCURRENCY=1` forces the old serial path for deterministic debugging.
- Runtime call ledger records event timestamps and reports `max_observed_concurrent_calls` plus `completed_interval_count`, giving smoke tests a runtime-owned overlap metric.

Local validation added:

- Deterministic model fan-out tests prove batch overlap, serial fallback, stable order, seed fan-out, and offspring fan-out.
- Call-ledger tests prove runtime-owned overlap summary generation.
- Full local suite after cleanup: `669 passed, 1 skipped`.
- Doctor: `50/50 checks passed`.
- Public hygiene test: `5 passed`; strict public-tree offender scan: 0.
- Test function definitions: 654; duplicate test function names: 0.

Debt status:

- TD-CONCUR-002 is closed at the code and deterministic-test level: the runtime can now schedule real independent model batches concurrently under the provider governor.
- A fresh real-provider smoke is operator-run provider/account validation rather than missing runtime scheduling capability.

## v2.3 theory runtime and model-route implementation — 2026-06-18

Scope: implement the v2.3 theory-strengthened runtime as first-class behavior, without v2.3 feature switches, without hardcoded seed-model names, and with an explicit model route allowing seed generation to use a different model from the default Nexus control-plane model.

SOTA basis checked before implementation:

- Quality-diversity / MAP-Elites: Mouret and Clune's MAP-Elites framing supports maintaining high-performing elites across behavior descriptor cells instead of optimizing only for one winner: https://arxiv.org/abs/1504.04909.
- Novelty plus local competition: novelty/local competition literature supports preserving diverse morphologies/mechanisms while still selecting high-value local elites: https://doi.org/10.1145/2001576.2001606.
- PI/PID control: PI-style proportional and integral error terms are a mature control pattern for turning measured error into bounded corrective pressure: https://ctms.engin.umich.edu/CTMS/index.php?example=Introduction&section=ControlPID.
- Adaptive budget allocation: budget-allocation literature supports treating fixed verification/inference budget distribution as a first-class algorithmic decision rather than a constant per item: https://arxiv.org/html/2605.26849v1.

Implemented changes:

- Added explicit model routing:
  - `nexus/model_routes.py` defines `NexusModelRole`, `NexusModelRoutes`, and `coerce_model_routes`.
  - `NexusRuntime(model=...)` remains compatible.
  - `NexusRuntime(model_routes=NexusModelRoutes(default_model=A, seed_model=B))` sends only `nexus_seed_population` to the seed model and keeps world/contract/policy/ranking/critique/diagnosis/mutation/offspring/synthesis/stop on the default model.
  - Seed-model failure does not silently fall back to default seed generation; the existing seed error handling records `model_seed_error` and continues via deterministic amplification/recovery.
  - Runtime serialization and run metadata include a sanitized `model_routes` summary.
- Added safe per-call model specs:
  - `llm/model_spec.py` defines `LLMModelSpec` with provider/model/api_base/fixture coordinates only, no credentials.
  - `StructuredModelAdapter.from_configured_llm(model_spec=...)` and `.with_configured_model(model_spec=...)` pass the model spec through to `llm_json`.
  - `llm_json(..., model_spec=...)` uses the actual per-call model spec for status, request hashing, idempotency, provider payload, ledger status, and journal status.
  - Public summaries redact fixture paths and credential-shaped URL material.
- Added v2.3 typed config:
  - `nexus/v23_theory_config.py` centralizes entropy compaction, minimax budget, honesty PI control, and CA crossover numeric parameters.
  - Deprecated v2.3 switch names are diagnostics-only. They do not change behavior.
  - No `COGEV_COMPACT_MODE`, `COGEV_DYNAMIC_ADVERSARIAL_BUDGET`, `COGEV_HONESTY_CONTROL`, or `COGEV_CROSSOVER_MODE` algorithm branch was added.
- Implemented entropy-QD live compaction:
  - Added descriptor cell distribution and entropy metrics.
  - Live compaction records entropy before/after, descriptor cell counts before/after, and `v23_theory_config_hash`.
  - Survivor selection preserves descriptor-cell elites, rare/edge/frontier candidates, and high search-quality candidates while retaining legacy per-primary-bin capacity semantics.
- Implemented minimax adversarial budget allocation:
  - `verification/minimax_budget.py` allocates actual candidate budgets from measured `candidate_verification_strength` only.
  - Stronger measured candidates receive budget greater than or equal to weaker measured candidates; all-NONE candidates receive uniform nonzero allocation.
  - `compile_grounding_regime(..., override_adversarial_budget=...)` gives the actual allocated budget priority over obligation/plan defaults.
  - Obligation cache keys include actual adversarial budget to prevent budget-mismatched reuse.
  - Obligation records include `minimax_budget_allocation_summary`.
- Implemented honesty PI control:
  - `nexus/honesty_control.py` computes a JSON-safe `HonestyControlSignal` from engine-owned honesty measurements.
  - `SearchDiagnosis.metadata` stores `honesty_control` without changing solved or verification authority.
  - `AdaptiveRuntimeState` persists `honesty_error_history` for resume-safe integral behavior.
  - `PolicyUpdater` converts honesty errors into bounded search pressure only: adversarial budget pressure, rarity/edge/frontier pressure, and replay/verifier pressure.
- Implemented CA descriptor-neighborhood crossover:
  - `candidates/crossover.py` now provides `descriptor_tokens`, `jaccard_similarity`, and `neighborhood_crossover_partner`.
  - `CrossOver` plans now have a deterministic two-parent fallback path instead of being treated as single-parent text mutation.
  - CA crossover records `ca_crossover` metadata on children.
  - Generation plans record `cell_activation_map`; `AdaptiveRuntimeState` persists `cell_activation_history`.

Authority boundaries preserved:

- Runtime metrics explain search pressure and candidate fate only; they do not grant `solved`, `verified_result`, or verification strength.
- `GradedOutput` remains the solved/verified authority; `closure_certificate` remains legacy closure signal.
- Verification strength still comes from honesty-core measurements and strength aggregation only.
- Replay evidence remains scoped to verifier-on-frozen-artifact replay, not LLM generation replay.
- Edge knowledge remains a seed and never a fact without source binding / verifier / honesty measurement support.
- GitHub change provenance is not implemented in runtime code; it remains owned by protected main, PR review, required CI, and CodeQL.

Tests added:

- `tests/test_model_routes_seed_model.py`
- `tests/test_v23_theory_config.py`
- `tests/test_entropy_compaction.py`
- `tests/test_minimax_budget.py`
- `tests/test_honesty_control_signal.py`
- `tests/test_ca_crossover.py`
- `tests/test_v23_no_legacy_switches_or_magic_numbers.py`

Local validation so far:

- New v2.3 focused suite: `29 passed`.
- Full local pytest after compatibility fix: `698 passed, 1 skipped`.
- Acceptance chain: compileall passed; pytest `698 passed, 1 skipped`; doctor `50/50 checks passed`; package-clean command completed; generated `dist/` removed; pycache directories removed; public hygiene scan clean; `source-current` mirror synced.
- Compatibility fix made during validation: entropy survivor target now combines primary descriptor group capacity with full descriptor-cell elite coverage, preserving existing per-bin cap behavior while adding v2.3 cell preservation.

Debt ledger:

| Debt ID | Category | Closure status | Evidence / notes |
|---|---|---|---|
| TD-V23-MODEL-ROUTES-SEED | Model routing | Closed in code + tests | Explicit `NexusModelRoutes`; seed route isolation and no default fallback covered by tests. |
| TD-V23-CONFIG-NO-SWITCH | Config / runtime behavior | Closed in code + tests | v2.3 behavior is config-driven; deprecated switches are diagnostics-only. |
| TD-V23-NO-MAGIC-NUMBERS | Config hygiene | Closed in code + tests | v2.3 numeric defaults centralized in typed config; tests guard against switch leakage and config bypass. |
| TD-V23-P1-ENTROPY | Entropy compaction | Closed in code + tests | Entropy/cell metrics and survivor behavior covered by focused tests and existing compaction regression. |
| TD-V23-P2-MINIMAX-BUDGET | Dynamic verification budget | Closed in code + tests | Stronger >= weaker, all-NONE uniform, sum conserved, override priority, and budget-sensitive cache keys covered. |
| TD-V23-P3-HONESTY-CONTROL | Honesty PI control | Closed in code + tests | Neutral signal, exogeneity/falsification pressure, variety pressure, and policy-pressure conversion covered. |
| TD-V23-P4-CA-CROSSOVER | CA crossover | Closed in code + tests | Descriptor token similarity, global donor policy, deterministic two-parent fallback, and cell activation map covered. |
| TD-V23-HIGH-CEILING-SMOKE | Fixture high-ceiling smoke | Closed at deterministic fixture-test level | v2.3 focused tests and full local pytest passed; real-provider high-ceiling run remains a post-merge local `test-runs/` activity. |
| TD-V23-PUBLIC-HYGIENE | Public-source hygiene | Closed locally before commit | Artifact directory scan clean; secret-shaped scan clean; local absolute path scan clean after replacing a test fixture path and constructing sanitizer prefixes at runtime; `source-current` mirror synced after validation. |

Publication validation note:

- Final hygiene was re-run after closeout changes in this branch.
- Historical note for the earlier v2.3 branch; current branch validation is recorded in the final closeout sections below.

## v2.3 gpt-5.5 xhigh full-review closure — 2026-06-19

Scope: close the actionable findings from the read-only `gpt-5.5 xhigh` full code/test flow review of the v2.3 theory-runtime/model-route branch and the subsequent 3-round routed smoke artifact audit.

Review artifacts:

- Review report: `${LOCAL_TEST_RUNS}/full-review-gpt55-xhigh-20260618-232911/gpt55-xhigh-full-review.md`
- Post-review seed/flow check: `${LOCAL_TEST_RUNS}/full-review-gpt55-xhigh-20260618-232911/post-review-seed-flow-check.md`
- Fix acceptance log: `${LOCAL_TEST_RUNS}/full-review-gpt55-xhigh-20260618-232911/fix-acceptance-20260619.log`
- Public hygiene log: `${LOCAL_TEST_RUNS}/full-review-gpt55-xhigh-20260618-232911/fix-public-hygiene-20260619.log`

Findings closed in this section:

- P1-1 stale seed-harvest test import:
  - `tests/test_search_kernel_v3.py` now imports the policy-based `_seed_safety_batch_limit` helper and verifies the bounded safety cap under the current seed-harvest semantics.
  - Full pytest collection is restored.
- P1-2 review-generated public hygiene pollution:
  - Removed untracked `.codemap/handoff.delta.json`, `.codemap/handoff.latest.json`, and `.codemap/handoff.prefix.json` generated during review.
  - `tests/test_public_tree_hygiene.py` passed after cleanup.
- P1-3 completion-status/solved-authority projection:
  - `nexus_verification_results(...)` no longer treats `completion_status == "solved"` as objective-solved authority.
  - Public `objective_solved` projection now requires closure/synthesis solved evidence plus `graded_output.mode == "verified_result"`.
  - Regression tests cover both a false-positive `completion_status="solved"`/`graded_portfolio` case and a positive `verified_result` case.
- P1-4 API verification-strength parsing:
  - `_nexus_verification_passed(...)` now reads `verification_strength_value` first and falls back through `VerificationStrength.from_value(...)`, so actual `GradedOutput.to_dict()` payloads with `"verification_strength": "FORMAL"` are accepted when replay evidence is valid.
  - Regression tests now build `GradedOutput(...).to_dict()` instead of relying only on a numeric legacy fixture, while keeping numeric migration compatibility.
- P2-1 resumed model-route metadata:
  - `resume_from_checkpoint(...)` now mirrors fresh `run_text`/`run_project` behavior by attaching sanitized `runtime_metadata.model_routes` before persistence.
  - Regression test confirms resumed runs preserve redacted route metadata.

3-round routed smoke audit outcome:

- The audited local smoke run completed exactly 3/3 rounds and kept `GradedOutput` at `graded_portfolio` with `VerificationStrength.NONE`; no solved/verified claim was made.
- The model-route split was evidenced by the call ledger: Gemini route for `nexus_seed_population`; GPT-5.5 route for non-seed Nexus stages.
- Seed harvest produced 37 unique seed candidates from 12 Gemini seed batches and stopped by the configured batch safety limit, not natural no-new exhaustion.
- Seed quality was acceptable as a mechanism smoke but not sufficient as a high-ceiling theory search: too much output focused on route/round/state contracts, `edge_knowledge_seeds` were empty, and formal/proof artifacts were sparse.
- Operator-run real-provider high-ceiling runs should emphasize mathematical models, exploitable theorems, cross-domain mechanisms, and performance algorithms rather than runtime-contract restatement.

Validation after fixes:

- Targeted affected tests: `47 passed`.
- Full local acceptance:
  - compileall: passed.
  - pytest: `703 passed, 1 skipped`.
  - doctor: `50/50 checks passed`.
  - package clean: completed; generated `dist/` removed.
  - pycache cleanup: completed.
  - public hygiene scan: no `.venv`, `test-runs`, logs/cache/pycache, egg-info, `dist`, bridge, tracked local absolute paths, or secret-shaped material found.
- `source-current` mirror synced from the public source checkout to the local runtime mirror after validation; stale `source-current` pycache was removed.

Debt status:

- TD-V23-XHIGH-REVIEW-P1 is closed locally: all P1 review findings are fixed or cleaned in the source tree.
- TD-V23-XHIGH-REVIEW-P2 is closed locally: resumed route metadata is restored and covered by test.
- TD-V23-HIGH-CEILING-SMOKE is closed as a smoke/mechanism check. High-quality theory discovery remains operator-owned result validation, not a source-tree debt item.

## v3 Exploration Fabric Phase 0 ledger — 2026-06-19

Status: closed in this phase branch.

- `TD-V3-P0-FABRIC-STATE` — Closed. Added domain-neutral `cognitive_evolve_runtime.fabric` primitives for advisory dossiers, tasks, task graphs, typed fabric config, and checkpoint fabric state without wiring behavior into the runtime loop.
- `TD-V3-P0-CHECKPOINT-COMPAT` — Closed. Added optional `NexusCheckpoint.fabric` field and default `{}` restore behavior so legacy checkpoints continue to load.
- `TD-V3-P0-ADVISORY-GUARD` — Closed. Added advisory authority-key guards and regression tests to ensure new fabric advisory payloads cannot carry verification-authority fields.

Validation requirements for the phase remain: compileall, full pytest, doctor, package clean, and public hygiene before PR.

## v3 Exploration Fabric Phase 1A ledger — 2026-06-19

Status: closed in this phase branch.

- `TD-V3-P1A-SHADOW-SCHEDULER` — Closed. Added coarse-grained `TaskGraphScheduler` shadow path with `EVALUATE → ROUND_GATE → REPRODUCE` graph, preserving existing `EvolutionRound.evaluate()` and `EvolutionRound.reproduce()` method boundaries.
- `TD-V3-P1A-OFFSPRING-VERIFIER-CONTEXT` — Closed. Added `FabricExecutionContext.offspring_verifier` contract and `ReproduceExecutor` pass-through so project runtime verifier closures remain bound by `NexusRuntime`.
- `TD-V3-P1A-MODEL-POOL-DIAGNOSTICS` — Closed. Unknown `model_pool` values fall back to default only with `fabric_state.diagnostics` warning coverage.

Validation requirements for the phase remain: compileall, full pytest, doctor, package clean, and public hygiene before PR.

## v3 Exploration Fabric Phase 1B ledger — 2026-06-19

Status: closed in this phase branch.

- `TD-V3-P1B-SINGLE-SCHEDULER` — Closed. Production `EvolutionLoopController.run()` now delegates each epoch to the coarse `TaskGraphScheduler`; `evolve_once()` remains the public API and reaches the same scheduler-backed controller path.
- `TD-V3-P1B-OLD-LOOP-REMOVAL` — Closed. Removed the old `_run_round()` and `_reproduce()` execution bodies; controller retains only scheduler orchestration plus recording/finalization helpers.
- `TD-V3-P1B-RESUME-COMPAT` — Closed. `fabric_state` is accepted by `evolve_once()`/controller, restored checkpoints pass fabric state into resumed runs, scheduler graph state is recoverable, and final persistence writes bounded fabric checkpoint data.

Validation requirements for the phase remain: compileall, full pytest, doctor, package clean, and public hygiene before PR.

## v3 Exploration Fabric Phase 2 ledger — 2026-06-19

Status: closed in this phase branch.

- `TD-V3-P2-POOL-PREPROCESS` — Closed. Added advisory pool clustering, descriptor coverage reporting, model-facing `nexus_pool_preprocess` request support, and a `PREPROCESS` executor that runs before the first scheduler-backed evaluation epoch.
- `TD-V3-P2-CLUSTER-SUPPORT-ADVISORY` — Closed. Exact duplicates use existing semantic signatures, near duplicates use existing search-kernel similarity, support counts and representatives are written as advisory metadata only, and no candidate is deleted by preprocessing.
- `TD-V3-P2-PROMPT-BOUNDS` — Closed. Pool preprocessing prompts use bounded candidate prompt views, typed config limits, and checkpointed pool reports exclude full prompt payloads / large candidate artifacts.

Validation requirements for the phase remain: compileall, full pytest, doctor, package clean, and public hygiene before PR.

## v3 run-report A-E defect closure ledger — 2026-06-20

Scope: close the A-E defect set from the temporary root review inputs `PLAN.md` and `PLAN-REVIEW-LOG.md`, then remove those plan files from the public source tree.

### Before implementation baseline

The review identified eleven concrete branch-start debt findings; their closure is recorded below:

| Debt ID | Category | Branch-start finding | Required closure condition |
|---|---|---|---|
| TD-V3-A-SEED-WAVEFRONT | Seed harvest concurrency | Baseline finding | Seed-specific concurrency overrides remain authoritative; absent seed override follows the shared provider fanout governor with documented previous-window snapshot semantics. |
| TD-V3-A-HARVEST-PARTIAL-ERROR | Harvest error handling | Baseline finding | Recoverable per-batch errors no longer discard successful batches from the same fanout window or poison accepted seeds. |
| TD-V3-A-SCHEMA-REPAIR-RETRY | Model adapter schema repair | Baseline finding | `nexus_seed_population` performs local repair first and at most one schema-repair retry at the adapter boundary. |
| TD-V3-B-EDGE-KNOWLEDGE-LINEAGE | Edge lineage | Baseline finding | Model offspring preserve parent/plan edge knowledge, inherited genes, novelty descriptors, and niche memberships without modulo-index parent guessing. |
| TD-V3-C-DISPLAY-SELECTOR | Final display authority | Baseline finding | Final/reference/best-current display selection uses one ordered selector and source-binding gating without adding a second ranking authority. |
| TD-V3-D-CALL-LEDGER-SESSION-PATH | Durable call ledger | Baseline finding | Session-scoped call ledger path has priority over process env and propagates through fanout workers. |
| TD-V3-D-COST-LEDGER-NAMESPACE | Cost accounting | Baseline finding | LLM provider cost is recorded in `cost_ledger.llm_provider`, separate from research-extension cost ledgers. |
| TD-V3-D-LATENT-SIDECAR-CHECKPOINT | Latent ledger checkpoint | Baseline finding | Latent ledger checkpoint payload is sidecar-only with ref/hash/cursor and restore-time hydration; legacy embedded payloads remain readable. |
| TD-V3-D-TASKGRAPH-CONCURRENCY | Fabric TaskGraph concurrency | Baseline finding | TaskGraph mutations are internally locked; RUNNING attempts and `updated_at` transitions are deterministic under concurrent marks. |
| TD-V3-E-DIAGNOSIS-HONESTY | Diagnosis schema honesty | Baseline finding | Model-adapter enum repair preserves raw custom diagnosis signals while internal `SearchDiagnosis` remains able to carry those signals. |
| TD-V3-PUBLIC-HYGIENE-PLANFILES | Public hygiene | Baseline finding | Temporary root `PLAN.md` and `PLAN-REVIEW-LOG.md` are removed before final public hygiene. |

### After validation

Status: closed in code + tests on this phase branch. All A-E items in this section are closed in this branch.

| Debt ID | Closure status | Code / test evidence |
|---|---|---|
| TD-V3-A-SEED-WAVEFRONT | Closed in code + tests | `_seed_fanout_workers()` now returns `None` absent seed override and preserves explicit serial override; tests cover default fanout and serial override. |
| TD-V3-A-HARVEST-PARTIAL-ERROR | Closed in code + tests | `HarvestResult` separates fatal and recoverable batch errors; harvest continues across partial batch failures; tests cover accepted seed cleanliness and failed batch summaries. |
| TD-V3-A-SCHEMA-REPAIR-RETRY | Closed in code + tests | `StructuredModelAdapterCore._call()` records local repair and performs exactly one seed schema-repair retry; regression test checks retry count and metadata. |
| TD-V3-B-EDGE-KNOWLEDGE-LINEAGE | Closed in code + tests | `_merge_plan_metadata_into_model_offspring(...)` is parent/plan aware, avoids modulo merge, and fills edge lineage fields without overwriting child output; regression test covers shuffled multi-plan children. |
| TD-V3-C-DISPLAY-SELECTOR | Closed in code + tests | Added `cognitive_evolve_runtime.nexus.display_selection` and wired final projection/controller display context; tests cover ordered fallback and source-binding strictness. |
| TD-V3-D-CALL-LEDGER-SESSION-PATH | Closed in code + tests | `LLMSession.call_ledger_path` is first-priority call ledger path and `run_ordered_fanout()` propagates contextvars; test verifies env path is bypassed. |
| TD-V3-D-COST-LEDGER-NAMESPACE | Closed in code + tests | Checkpoint cost payload now writes provider telemetry under `llm_provider` while preserving research-extension cost payloads; regression test covers separation. |
| TD-V3-D-LATENT-SIDECAR-CHECKPOINT | Closed in code + tests | `LatentLedgerStore.persist_ledger()` read-back verifies sidecars and returns ref/hash/cursor; `CheckpointStore.restore_state()` hydrates sidecar refs and keeps legacy embedded compatibility. |
| TD-V3-D-TASKGRAPH-CONCURRENCY | Closed in code + tests | `TaskGraph` has an internal `RLock`; `mark()`, `recover_inflight()`, `ready_tasks()`, topology and serialization are locked; tests cover attempts, updated_at, and concurrent marks. |
| TD-V3-E-DIAGNOSIS-HONESTY | Closed in code + tests | Diagnosis schema enum is enforced only at adapter boundary; repair records `metadata.raw_stagnation_type` and keeps raw signal in notes; internal custom `SearchDiagnosis` signals survive. |
| TD-V3-PUBLIC-HYGIENE-PLANFILES | Closed locally | Temporary root plan/review files were deleted from the public checkout before full pytest/public hygiene. |

Focused validation before final acceptance:

- Targeted A-E suites: `41 passed`.
- Full local pytest after A-E fixes and answer-first realignment: `691 passed, 1 skipped`.

Final local validation after implementation:

- compileall: passed for `cognitive_evolve_runtime`, `scripts`, and `tests`.
- full pytest: `691 passed, 1 skipped`.
- doctor: `50/50 checks passed`.
- package clean: completed; generated `dist/` removed afterwards.
- generated-artifact cleanup: `dist/`, `__pycache__`, and `*.pyc` removed from the public checkout.
- public hygiene: no forbidden runtime artifact directories or generated data files found; public hygiene regression suite passed.
- mirror sync: public source tree synced to `${LOCAL_RUNTIME_ROOT}/source-current` with no post-sync diff under the configured excludes.

## v3 answer-first exploration de-engineering ledger — 2026-06-21

Scope: remove over-engineered project-verification authority from CognitiveEvolve search output so the runtime boldly explores and returns candidate answers/hypotheses. Verification after the run is user-owned and external to the project; project checks remain advisory telemetry only.

Status: closed in code + tests on this phase branch. All answer-first realignment items in this section are closed in this branch.

| Debt ID | Closure status | Code / test evidence |
|---|---|---|
| TD-V3-X-ANSWER-FIRST-AUTHORITY | Closed in code + tests | Final synthesis, closure, runtime status, and API payload semantics now treat nonempty candidate answers as completion material instead of requiring project-certified proof/evidence gates. |
| TD-V3-X-ARTIFACT-CONTRACT-ADVISORY | Closed in code + tests | Dynamic artifact contract validation and candidate contract evaluation are advisory; missing patch/proof/source fields no longer block rank/final eligibility for answer material. |
| TD-V3-X-PROOFOBJECT-RETIRE | Closed in code + tests | Proof-object, obligation, and external-evidence diagnostics were removed from hard-reject/repair policy and normalized away from diagnosis control flow. |
| TD-V3-X-DIRECT-FINAL-ANSWER | Closed in code + tests | Final projection and synthesis now display the best answer directly; seed/route/reference wrappers are not used as a second authority layer. |
| TD-V3-X-SOURCE-BINDING-NONBLOCKING | Closed in code + tests | Source binding, hallucinated symbol, and lineage checks remain visible as advisory metadata/diagnostics but no longer block displayed answers. |
| TD-V3-X-DIAGNOSIS-ANSWER-FIRST | Closed in code + tests | Stagnation diagnosis and policy directives no longer force proof/source repair loops; retired proof-bottleneck categories are mapped to answer-first diversity/convergence handling while raw signals remain preserved where needed. |
| TD-V3-X-TEST-REALIGNMENT | Closed in code + tests | Regression tests were realigned to answer-first semantics across final gates, projection, synthesis, archive fates, stage policy, evidence control plane, and project patch sandbox behavior. |
| TD-V3-X-RESUME-NO-RESEED | Closed in code + tests | No long run was restarted and no new seed_population run was launched from the public source checkout; changes are source/runtime semantics only and are ready for the user to validate in a resumed runtime run. |

Validation after answer-first realignment:

- `PYTHONDONTWRITEBYTECODE=1 ${PY:-python} -B -m compileall -q cognitive_evolve_runtime scripts tests` — passed.
- `PYTHONDONTWRITEBYTECODE=1 ${PY:-python} -B -m pytest -q -p no:cacheprovider` — `691 passed, 1 skipped`.
- `PYTHONDONTWRITEBYTECODE=1 ${PY:-python} -B scripts/cogev.py doctor --scope all` — `50/50 checks passed`.
- `bash scripts/package_clean.sh` — completed; generated `dist/` is removed again during final public hygiene cleanup.

## v3 subagent closeout review ledger — 2026-06-21

Scope: close the multi-agent review findings after answer-first realignment and ensure there is one user-facing answer authority, no stale routed-output path, no hidden project-certification gate, and no public-local path leakage.

Status: closed in code + tests on this branch.

| Debt ID | Closure status | Code / test evidence |
|---|---|---|
| TD-V3-R-SINGLE-ANSWER-AUTHORITY | Closed in code + tests | `FinalSynthesizer` now returns one answer path; `build_final_projection()` uses `synthesis.final_answer` and `synthesis.best_candidate_id` first, with ranking/display context only as diagnostic fallback. |
| TD-V3-R-MODEL-ANSWER-BINDING | Closed in code + tests | Model synthesis no longer binds a mismatched model `final_answer` to a candidate artifact/replay id; mismatch is surfaced via `model_final_answer_unbound_to_candidate_artifact`. |
| TD-V3-R-PROJECTION-ADVISORY-VISIBILITY | Closed in code + tests | Final projection no longer rewrites advisory evidence to certified-clean status; it exposes advisory fields such as `advisory_final_blocked`, `advisory_artifact_final_eligible`, and `answer_candidate_mismatch`. |
| TD-V3-R-ANSWER-VS-SOLVED-SEPARATION | Closed in code + tests | Runtime completion keeps answer production separate from correctness: `answer_produced` records candidate output, while `objective_solved` is not self-certified without user/external verification. |
| TD-V3-R-STATUS-CANONICALIZATION | Closed in code + tests | Public API/job/state surfaces normalize legacy routed-output statuses to `completed`; answer-bearing terminal jobs expose results by payload availability rather than a second completion status. |
| TD-V3-R-LATENT-VERIFY-NONBLOCKING | Closed in code + tests | `requires_verified_solution` and latent convergence no longer create a hard public continuation path for answer-first output; verification/latent state remains advisory metadata. |
| TD-V3-R-DEAD-GATE-CLEANUP | Closed in code + tests | Removed unreachable legacy final-gate/proof blocking code from archive constraints and cleaned misleading comments/names around project-certification gates. |
| TD-V3-R-PUBLIC-HYGIENE-TMP-PATHS | Closed in code + tests | Validation commands in this public ledger use `${PY:-python}` instead of host-specific `/tmp` venv paths; public hygiene tests now detect local home and repo-specific temp venv paths. |
| TD-V3-R-DISPLAY-FALLBACK-ORDER | Closed in code + tests | `DisplayContext` now includes synthesis answer candidate fallback ids before population-order fallback, preventing stale ranking or seed order from silently replacing the selected answer. |

Subagent review closure validation:

- `PYTHONDONTWRITEBYTECODE=1 ${PY:-python} -B -m compileall -q cognitive_evolve_runtime scripts tests` — passed.
- `PYTHONDONTWRITEBYTECODE=1 ${PY:-python} -B -m pytest -q -p no:cacheprovider` — `691 passed, 1 skipped`.
- `PYTHONDONTWRITEBYTECODE=1 ${PY:-python} -B -m pytest -q -p no:cacheprovider tests/test_public_tree_hygiene.py` — `5 passed`.
- `git diff --check` — passed.

## v3 fresh seed24 runtime cap closure — 2026-06-21

Scope: close the fresh-run discovery that the user-requested seed cap of 24 was recorded in run-local overlay metadata but the source seeding helper still clamped configured seed batches to 16.

Status: closed in code + tests on this branch.

| Debt ID | Closure status | Code / test evidence |
|---|---|---|
| TD-V3-R-SEED24-HARD-CAP | Closed in code + tests | `_seed_safety_batch_limit()` now honors explicit policy/env seed batch values up to `SEED_BATCH_CONFIGURED_MAX=64`; regression tests prove an explicit 24-batch seed policy overrides a lower env value and oversized env values are still bounded. |

Runtime evidence behind this closure:

- Fresh retry evidence under the local test-run tree showed `seed_overlay.seed_safety_max_batches=24` but checkpoint `seed_harvest.batches=16` with `stopped_reason=batch_limit`.
- The hard cap was source-local in `cognitive_evolve_runtime/nexus/loop/seeding.py`, not a provider or bridge behavior.
- The source fix is required before launching the next fresh 24-seed run from `source-current`.

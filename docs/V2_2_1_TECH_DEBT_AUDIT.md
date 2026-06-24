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
| TD-V3-R-SEED24-HARD-CAP | Closed in code + tests | `_seed_safety_batch_limit()` honors explicit policy/env seed batch values; the later self-bootstrap loop closure removed the stale low fixed seed cap entirely while preserving the explicit 24-batch regression. |

Runtime evidence behind this closure:

- Fresh retry evidence under the local test-run tree showed `seed_overlay.seed_safety_max_batches=24` but checkpoint `seed_harvest.batches=16` with `stopped_reason=batch_limit`.
- The hard cap was source-local in `cognitive_evolve_runtime/nexus/loop/seeding.py`, not a provider or bridge behavior.
- The source fix is required before launching the next fresh 24-seed run from `source-current`.

## v3 NextGen CBT-PCBG landing ledger — 2026-06-22

Scope: land the NextGen CognitiveEvolve CBT-PCBG review plan while preserving answer-first exploration semantics. The implementation removes pre-existing exploration hard gates before adding Critical Branching soft budget signals, keeps verification/authority walls advisory for final claims only, avoids new heavy dependencies, and keeps the public source tree clean.

Status: closed in code + tests on this branch.

| Debt ID | Closure status | Code / test evidence |
|---|---|---|
| TD-V3-NG-PR0-HARVEST-RESERVOIR-SOFT-REJECT | Closed in code + tests | `CandidateHarvester` supports `reservoir_mode`; low-relevance and semantic-duplicate seed outputs enter `HarvestResult.reservoir` with `candidate_budget_decision` soft traces. Covered by `tests/test_nextgen_cbt_pcbg_landing.py`. |
| TD-V3-NG-PR0-OFFSPRING-PLAN-DEDUPE-SOFT | Closed in code + tests | Offspring semantic duplicates and plan signature duplicates are retained with soft trace metadata; exact structural blockers remain the only hard exclusion lane. Covered by nextgen and reproduction regressions. |
| TD-V3-NG-PR0-FATE-NONTERMINAL-EXPLORATION | Closed in code + tests | Verifier/project/archive/compaction paths now keep non-structural source-free, docs-only, narrative-only, and repairable failures as Active/Incubating/Dormant advisory material. Covered by archive, population, dormant repair, failure classifier, and nextgen regressions. |
| TD-V3-NG-PR0-BUDGET-ELIGIBLE-LANE | Closed in code + tests | `budget_eligible_candidates()` includes dormant/reserve/reservoir material and excludes only structural or safety failures; parent/ranking/diagnosis paths use this lane. |
| TD-V3-NG-PR0-CANDIDATE-BUDGET-TRACE | Closed in code + tests | Former hard gates write capped `candidate_budget_decisions` events through `record_candidate_budget_decision()`, including harvest, archive, parent, compaction, verifier, and repair reactivation paths. |
| TD-V3-NG-PR1-SUBSTRATE-REGRESSION | Closed in code + tests | Existing A-E substrate tests remain green; stage-budget strict env and `llm_provider` cost namespace were preserved by the full regression suite. |
| TD-V3-NG-PR2-CANONICAL-FAMILY-ID | Closed in code + tests | Compatible `metadata.nextgen` identity fields are populated; unknown search-space families become provisional singleton families instead of falling back to the first configured family. |
| TD-V3-NG-PR2-PRODUCTIVE-OBSERVATION-NONBLOCKING | Closed in code + tests | `ProductiveChildObservation` is observation-only and has no `.passed` or `must_not_block` runtime branch field. |
| TD-V3-NG-PR2-CONSUMER-ALLOWLIST | Closed in code + tests | Static consumer tests scan archive, parent, ranking, verifier, repair, synthesis, final-gate, and verification-stack files to prevent productive/CBT soft signals from becoming hard gate consumers. |
| TD-V3-NG-PR3-LLM-PROFILE-IDENTITY | Closed in code + tests | `LLMCallIdentity` and `LLMModelSpec.profile_id` flow through breaker keys, idempotency, journal, call ledger, telemetry, and public route summary while preserving legacy provider/model fields. |
| TD-V3-NG-PR4-SEED-RESERVOIR-CHECKPOINT | Closed in code + tests | Seed reservoir summaries are carried in seed metadata and checkpoint-safe population serialization; source acceptance does not require a real-provider rerun. |
| TD-V3-NG-PR4-PROMPT-PROTECTED-FRONTIER | Closed in code + tests | Prompt view compression preserves protected candidate objects by shrinking detail before dropping frontier candidates; tiny-budget fallback keeps protected ids and summaries. |
| TD-V3-NG-PR5-CBT-SOFT-QUOTA | Closed in code + tests | CBT Critical Branching is implemented as a soft trace/score adjustment with floor `1`; singleton, rare, novel, and edge branches are protected rather than blocked. |
| TD-V3-NG-PR5-FALSE-CULL-NOVELTY-MONITOR | Closed in code + tests | Prompt population stats expose false-cull and boundary-loop counters so throttle can be defanged when exploration quality worsens. |
| TD-V3-NG-PR6-COST-OPT-IN-ONLY | Closed in code + tests | Ranking reverse-order audit is controlled by `COGEV_RANK_ORDER_AUDIT=off|sample|always`; transport cost metadata stays in provider ledgers and prompt limits only shrink payloads. |
| TD-V3-NG-PR7-PUBLIC-USABILITY | Closed in code + tests | README and provider docs are fixture-first and profile-aware, with no host-local paths, private relay facts, or runtime artifacts added. |
| TD-V3-NG-PUBLIC-HYGIENE-MIRROR | Closed in code + tests | Full local acceptance passed; final cleanup and mirror sync are performed after `package_clean.sh` and public hygiene scans. |

Validation after NextGen CBT-PCBG landing:

- `PYTHONDONTWRITEBYTECODE=1 ${PY:-python} -B -m compileall -q cognitive_evolve_runtime scripts tests` — passed.
- `PYTHONDONTWRITEBYTECODE=1 ${PY:-python} -B -m pytest -q -p no:cacheprovider` — `704 passed, 1 skipped`.
- `PYTHONDONTWRITEBYTECODE=1 ${PY:-python} -B scripts/cogev.py doctor --scope all` — `50/50 checks passed`.
- Subagent closeout review found no blocking old-gate, dual-authority, public-hygiene, or over-engineering issues; one static-test coverage suggestion was closed by expanding the consumer allowlist test.

## v3 Final Best Direction + Resurrection cleanup ledger — 2026-06-22

Scope: make answer-first output total over all non-structural/non-safety candidates, add a minimal loser-pool resurrection lane inspired by Swiss/non-elimination and double-elimination loser brackets, and close the small review debts without adding a second selector, tournament framework, database, or new dependency.

Status: closed in code + tests on this branch.

| Debt ID | Closure status | Code / test evidence |
|---|---|---|
| TD-V3-NG-FINAL-BEST-CURRENT | Closed in code + tests | `SynthesizedResult` and `FinalProjection` expose `best_current_direction` for the best non-structural candidate; failed/source-free candidates can produce an answer while `objective_solved=false`, and structural/stage hard rejects remain excluded. Covered by `tests/test_nextgen_cbt_pcbg_landing.py`, `tests/test_nexus_audit_regressions.py`, and adaptive projection regressions. |
| TD-V3-NG-RESURRECTION-LANE | Closed in code + tests | `ParentSelector` adds a bounded soft resurrection quota from dormant/reserve/failed/reservoir candidates, writes `resurrection_*` trace metadata, and protects Critical Branching over pure framework noise without adding a bracket scheduler or second selector. |
| TD-V3-NG-RESERVOIR-CAP | Closed in code + tests | `HarvestPolicy.reservoir_limit` defaults to `256`; overflow records `reservoir_truncated_count` and capped summaries instead of appending full objects indefinitely. |
| TD-V3-NG-SOFT-CAP-NAMING | Closed in code + tests | Legacy `max_per_group` repair/reseed config is documented and emitted as `soft_group_hint`; overrepresented groups remain selectable and are not hard-capped. |
| TD-V3-NG-SEED-ENSEMBLE-LINEAGE | Closed in code + tests | `SeedModelEnsembleAdapter` preserves `origin_model_index` for both dict seed outputs and `CandidateGenome` seed outputs while keeping input-order-stable fanout. |
| TD-V3-NG-IDENTITY-NO-PARALLEL-PATH | Closed in code + tests | LLM profile identity regression verifies provider dispatch still uses provider/model, while breaker/idempotency/journal/call-ledger use profile identity for isolation rather than a duplicate provider path. |

Validation after Final Best Direction + Resurrection cleanup:

- `PYTHONDONTWRITEBYTECODE=1 ${PY:-python} -B -m compileall -q cognitive_evolve_runtime scripts tests` — passed.
- `PYTHONDONTWRITEBYTECODE=1 ${PY:-python} -B -m pytest -q -p no:cacheprovider` — `712 passed, 1 skipped` after explicit removal of a pre-existing `.pytest_cache/` runtime artifact.
- `PYTHONDONTWRITEBYTECODE=1 ${PY:-python} -B scripts/cogev.py doctor --scope all` — `50/50 checks passed`.
- `bash scripts/package_clean.sh` — completed; generated `dist/` is removed again during final public hygiene cleanup.
- Independent subagent review findings on structural hard rejects, fallback ordering, display route tests, and public hygiene self-cleaning were addressed before closure.

## v3 NextGen intent binding final-direction correction ledger — 2026-06-22

Scope: close the smoke-run finding that final selection could choose an audit/ledger support object as the best direction when the frozen user goal asked for mechanism/model/framework exploration, and that final summaries could surface advisory material as `verified`.

Status: closed in code + tests on this branch.

| Debt ID | Closure status | Code / test evidence |
|---|---|---|
| TD-V3-NG-INTENT-BINDING-NO-ENUM | Closed in code + tests | `metadata.intent_binding` stores free-text `search_intent`, `candidate_main_claim`, `supporting_claims`, `alignment_rationale`, and continuous `direct_answer_score`; no target-kind enum was added. |
| TD-V3-NG-NO-DOMAIN-HARDCODED-FINAL-SCORE | Closed in code + tests | Final and resurrection scoring now use intent directness plus existing soft signals; static regression tests reject domain-specific final/resurrection scoring constants. |
| TD-V3-NG-SUPPORTING-ARTIFACT-NOT-BEST-DIRECTION | Closed in code + tests | When the frozen goal asks for a search/model direction, direct goal claims outrank support artifacts even if the support artifact has stronger artifact/verifiability scores; the same support artifact can still win when the goal asks for it. |
| TD-V3-NG-USER-FACING-VERIFICATION-HONESTY | Closed in code + tests | `best_current_direction.verification_status` only reports `verified` from graded verified output or replayable final certificate evidence; candidate-local metadata alone is advisory. |
| TD-V3-NG-SUMMARY-ANSWER-PRODUCED-CONSISTENCY | Closed in code + tests | The controller refreshes `best_current_direction` after graded output is known, preserving `objective_solved=false` while keeping answer-first output available. |

Validation after intent binding correction:

- `PYTHONDONTWRITEBYTECODE=1 ${PY:-python} -B -m compileall -q cognitive_evolve_runtime/nexus tests/test_nextgen_cbt_pcbg_landing.py tests/test_failure_classifier.py` — passed.
- `PYTHONDONTWRITEBYTECODE=1 ${PY:-python} -B -m pytest -q -p no:cacheprovider tests/test_nextgen_cbt_pcbg_landing.py tests/test_failure_classifier.py tests/test_nexus_audit_regressions.py tests/adaptive/test_evidence_control_plane.py` — `77 passed`.

## v3 semantic consistency and lightweight boundary closure ledger — 2026-06-23

- `TD-V3-RUNTIME-OPTIONS-RESUME-SEMANTICS` — Closed in code + tests. Checkpoints now persist open `runtime_options`; project resume restores effective `verification.include_tests` instead of falling back to a hard-coded verifier default.
- `TD-V3-FALLBACK-CAPTURE-CONTEXT-MANAGER` — Closed in code + tests. Nexus runtime entrypoints use `capture_fallback_events()` and text-world fallback only catches component-declared fallback exceptions or model-boundary errors.
- `TD-V3-CONTRACT-HASH-LINEAGE-NO-REBASE` — Closed in code + tests. Resumed candidates keep their generation-time `contract_hash`; current verification contract and overlays are recorded in verification summaries.
- `TD-V3-PROJECT-CONTEXT-EXPLICIT-FLOW` — Closed in code + tests. Project context is passed through explicit runtime/Fabric context and can be resolved by downstream components without extra positional parameters.
- `TD-V3-SEED-FAMILY-PRIORITY-OPEN-PLANES` — Closed in code + tests. Seed batches receive model-authored family priority and coverage traces without finite domain enums or hard gates.
- `TD-V3-SERDE-BOUNDARY-LOWER-LAYER` — Closed in code + tests. Stable JSON/hash helpers moved to `core.serialization`; `nexus._serde` remains a compatibility re-export while non-Nexus low-level packages import the core module.
- `TD-V3-LLM-REQUEST-POLICY-NO-NEXUS-TRANSPORT` — Closed in code + tests. Long-context output budgets are selected by explicit `LLMRequestPolicy` from the model adapter, not by Nexus request-name constants inside transport.
- `TD-V3-PUBLIC-HYGIENE-MIRROR` — Closed in code + tests. No run artifacts, local virtualenvs, bridge logs, or plan files are introduced by this change; mirror sync remains part of final acceptance.

## v3 self-bootstrap loop seed coverage and resurrection ledger — 2026-06-24

Scope: close the real self-bootstrap loop findings around low seed caps, seed family coverage, target-perturb continuation, loser-pool factor resurrection, strategy comparison, checkpoint monitor state, route discipline, and capability-preserving efficiency.

Status: closed in code + tests on this branch.

| Debt ID | Closure status | Code / test evidence |
|---|---|---|
| TD-V3-LOOP-SEED-UNCAPPED-COVERAGE | Closed in code + tests | `_seed_safety_batch_limit()` now honors explicit policy/env values without the stale 64 cap; `tests/test_search_kernel_v3.py` proves a high seed batch env is no longer clipped. |
| TD-V3-LOOP-SEED-COVERAGE-ASSESSMENT | Closed in code + tests | `nexus/seed_coverage.py::assess_seed_coverage()` records accepted/reservoir/rejected/family/singleton/origin coverage and is written into policy/candidate metadata; `tests/test_self_bootstrap_loop_controls.py` covers broad vs thin coverage. |
| TD-V3-LOOP-SEED-PARTIAL-FAILURE-STATUS | Closed in code + tests | Seed coverage carries `partial_failure_count`; existing harvester result keeps recoverable failed batch ids without poisoning accepted seeds, and full pytest covers harvester regressions. |
| TD-V3-LOOP-TARGET-PERTURB-SEED | Closed in code + tests | `target_perturb_seed_judgment()` provides evidence-first `not_needed/watch/trigger_recommended` guidance from checkpoint population signals without rerunning initial seed; tests cover the round-10 stuck trigger. |
| TD-V3-LOOP-FACTOR-RESURRECTION-TRACE | Closed in code + tests | `nexus/factor_resurrection.py` extracts advisory factors from dormant/failed/reservoir candidates while keeping source candidates non-terminal; tests cover loser-pool factor extraction. |
| TD-V3-LOOP-FAILURE-FACTOR-VIEW | Closed in code + tests | `archive_prompt_view()` adds `failure_factor_hints` even when `FailureArchive.records` is empty but dormant candidates carry failure lessons; tests cover this exact case. |
| TD-V3-LOOP-RESURRECTION-QUOTA-DATA-SENSITIVE | Closed in code + tests | `resurrection_quota()` now accepts pool size/pressure and can exceed 3 under large loser pools while staying within branch budget; tests prove scaling and floor. |
| TD-V3-LOOP-NO-HARDCODED-GENERAL-SEARCH-WORDLIST | Closed in code + tests | General scaffold stripping and false-cull boundary wordlist checks were removed from search steering; tests prove terms are not stripped/flagged by those generic helpers. |
| TD-V3-LOOP-STRATEGY-COMPARISON-CARRIER | Closed in code + tests | `nexus/strategy_comparison.py` carries open free-text hypotheses/observations into prompt views without architecture enums; tests cover arbitrary free-text strategy observations. |
| TD-V3-LOOP-CHECKPOINT-MONITOR-STATE | Closed in code + tests | `LiveNexusStore` and final persistence carry `search_kernel`, `runtime_options`, and `monitor_state`; tests read checkpoint and round snapshot evidence. |
| TD-V3-LOOP-CODEX-GPT55-HIGH-ROUTING | Closed in protocol + tests | Public LOOP protocol requires all real self-bootstrap roles to resolve to one high-capability profile before provider calls; route values stay run-local rather than embedded as source constants. |
| TD-V3-LOOP-ALGORITHM-EFFICIENCY-METRICS | Closed in code + tests | Seed loop records `algorithm_efficiency` metrics such as accepted-per-batch, reservoir count, and partial failures as measure-only metadata; prompt/checkpoint tests cover persistence. |
| TD-V3-LOOP-MODEL-PARALLEL-EFFICIENCY | Closed in code + tests | Seed loop records `model_parallel_efficiency` and no longer clips explicit seed fanout by stale seed batch cap; model fanout and seed tests pass in full pytest. |
| TD-V3-LOOP-TRANSPORT-SCAFFOLD-HYGIENE | Closed in code + tests | Public LOOP protocol avoids private transport scaffold terms and public hygiene tests continue to guard bridge/proxy/local path leakage. |
| TD-V3-LOOP-SELFBOOTSTRAP-PROTOCOL | Closed in docs + tests | `docs/SELF_BOOTSTRAP_LOOP_PROTOCOL.md` defines the minimal run-audit-modify-rerun protocol without adding a scheduler framework. |
| TD-V3-LOOP-PUBLIC-HYGIENE-MIRROR | Closed in acceptance | `compileall`, full pytest, doctor, and package_clean passed; final cleanup, hygiene scan, and source-current mirror sync are required before push. |

Validation evidence for this closure:

- `PYTHONDONTWRITEBYTECODE=1 ${PY:-python} -B -m compileall -q cognitive_evolve_runtime scripts tests` — passed.
- `PYTHONDONTWRITEBYTECODE=1 ${PY:-python} -B -m pytest -q -p no:cacheprovider` — `733 passed, 1 skipped`.
- `PYTHONDONTWRITEBYTECODE=1 ${PY:-python} -B scripts/cogev.py doctor --scope all` — `50/50 checks passed`.
- `bash scripts/package_clean.sh` — produced the clean package artifact, then `dist/` and bytecode caches were removed.

## v3 minimal-core/full-fusion ablation and efficiency closure ledger — 2026-06-24

Scope: reopen the remaining gaps from the run-result review: minimal active core and full fusion must be represented as runnable comparison profiles, failure theorem and R_eff must exist as source mechanisms, large seed pool must be separated from the active frontier, reservoir sidecar must keep large pools out of checkpoints, and efficiency work must be capability-preserving.

Status: closed in code + tests on this branch.

| Debt ID | Closure status | Code / test evidence |
|---|---|---|
| TD-V3-MINCORE-FOUR-WAY-ABLATION | Closed in code + tests | `nexus/minimal_core.py::run_core_ablation()` replays `score_only`, `Nexus_QD_failure_replay`, `minimal_active_core`, and `full_fusion` on the same candidate pool with zero added provider calls; tests assert all four profiles are compared. |
| TD-V3-MINCORE-R-EFF | Closed in code + tests | `estimate_reproduction_pressure()` records an advisory `r_eff.v1` signal from surviving child pressure, novelty pressure, and failure reactivation without becoming a hard gate. |
| TD-V3-MINCORE-FAILURE-THEOREM | Closed in code + tests | `extract_failure_theorem()`/`extract_failure_theorems()` lift reusable failure lessons into advisory theorem payloads and prompt/checkpoint traces. |
| TD-V3-MINCORE-SINGLE-PROMOTION-GATE | Closed in code + tests | `single_promotion_gate()` separates promotion eligibility from verified-claim permission so exploratory candidates can advance without self-certifying solved. |
| TD-V3-MINCORE-LARGE-POOL-SMALL-FRONTIER | Closed in code + tests | `apply_seed_active_frontier()` keeps large accepted seed pools intact while marking a bounded active frontier and moving overflow to Dormant with trace metadata. |
| TD-V3-MINCORE-SEED-UNBOUNDED-RUNMODE | Closed in code + tests | `COGEV_NEXUS_SEED_BATCH_LIMIT=unbounded` disables the static seed batch cap and harvests until exhaustion/low-gain/budget/operator stop; default bounded behavior remains available for ordinary runs. |
| TD-V3-MINCORE-SEED-RESERVOIR-SIDECAR | Closed in code + tests | Seed reservoir payloads are written to digest-named sidecars and checkpoints store only refs, not large embedded seed pools. |
| TD-V3-MINCORE-OPTIONAL-LAYERS-PAY-RENT | Closed in code + tests | The four-way ablation reports optional-layer marginal gains and recommends minimal core unless full fusion shows enough measured advantage. |
| TD-V3-MINCORE-ALGORITHM-EFFICIENCY-NO-CAPABILITY-LOSS | Closed in code + tests | Seed loop records algorithm-efficiency metrics and active-frontier separation while preserving reservoir/dormant material instead of deleting breadth. |
| TD-V3-MINCORE-MODEL-PARALLEL-EFFICIENCY | Closed in code + tests | Seed fanout remains governed by explicit policy/env concurrency and records parallel-efficiency metadata without clipping seed breadth. |
| TD-V3-MINCORE-GPT55-HIGH-GUARD | Closed in code + tests | `COGEV_LLM_REQUIRED_MODEL` blocks real provider calls whose resolved `COGEV_LLM_MODEL` differs from the operator-selected GPT 5.5 high model id; the self-bootstrap runner exposes `--required-model`. |
| TD-V3-MINCORE-RUN-SUMMARY-FIELDS | Closed in code + tests | Runtime result/checkpoint/prompt paths carry `minimal_core_ablation`, `factor_resurrection_summary`, seed coverage, active frontier, reservoir ref, and efficiency summaries. |

Validation after minimal-core/full-fusion closure:

- `PYTHONDONTWRITEBYTECODE=1 ${PY:-python} -B -m compileall -q cognitive_evolve_runtime scripts tests` — passed.
- `PYTHONDONTWRITEBYTECODE=1 ${PY:-python} -B -m pytest -q -p no:cacheprovider tests/test_search_kernel_v3.py tests/test_model_fanout_concurrency.py tests/test_nextgen_cbt_pcbg_landing.py tests/test_self_bootstrap_loop_controls.py tests/test_security_config_and_stop_decision.py` — `68 passed`.
- `PYTHONDONTWRITEBYTECODE=1 ${PY:-python} -B -m pytest -q -p no:cacheprovider` — `738 passed, 1 skipped`.
- `PYTHONDONTWRITEBYTECODE=1 ${PY:-python} -B scripts/cogev.py doctor --scope all` — `50/50 checks passed`.
- `bash scripts/package_clean.sh` — completed; generated `dist/` and bytecode caches were removed again during final public hygiene cleanup.

## v3 self-bootstrap LOOP artifact-backed closure ledger — 2026-06-24

Scope: close issues confirmed by the GPT 5.5 high unbounded-seed self-bootstrap run artifacts, without adding hardcoded finite categories or sacrificing search breadth.

Status: closed in code + tests on this branch.

| Debt ID | Closure status | Code / test evidence |
|---|---|---|
| TD-V3-LOOP-UNBOUNDED-SEED-HANDOFF | Closed in code + tests | Unbounded seed harvest now hands off after broad coverage and consecutive low marginal family-yield signals derived from target/min-batches/patience, not a static low seed cap; replaying Attempt4 would stop at 524 accepted candidates / 210 families. |
| TD-V3-LOOP-SEED-HARVEST-METADATA-BLOAT | Closed in code + tests | Full seed harvest summary is stored once in policy metadata; each candidate now carries only a compact per-candidate seed_harvest trace while retaining failed-batch and stop evidence. |
| TD-V3-LOOP-INTENT-BINDING-STALE-FALLBACK | Closed in code + tests | Intent binding now refreshes stale no-contract or different-goal bindings once the frozen contract intent is available, and the token matcher now computes real free-text overlap. |
| TD-V3-LOOP-SEED-COVERAGE-PERSISTENCE | Closed in code + tests | PolicyUpdater preserves seed/search-kernel metadata when model-authored policy updates omit those fields, so final checkpoint persistence can still emit coverage/frontier/efficiency refs. |
| TD-V3-LOOP-CAPABILITY-PRESERVING-EFFICIENCY | Closed in code + tests | The fix reduces duplicated metadata and adds dynamic handoff without deleting accepted seeds; active/dormant/reservoir semantics remain intact for later resurrection. |


Validation after artifact-backed LOOP closure:

- `PYTHONDONTWRITEBYTECODE=1 ${PY:-python} -B -m pytest -q -p no:cacheprovider tests/test_search_kernel_v3.py tests/test_model_fanout_concurrency.py tests/test_nextgen_cbt_pcbg_landing.py tests/test_self_bootstrap_loop_controls.py tests/test_nexus_audit_regressions.py tests/adaptive/test_evidence_control_plane.py` — `102 passed`.
- `PYTHONDONTWRITEBYTECODE=1 ${PY:-python} -B -m compileall -q cognitive_evolve_runtime scripts tests` — passed.
- `PYTHONDONTWRITEBYTECODE=1 ${PY:-python} -B -m pytest -q -p no:cacheprovider` — `741 passed, 1 skipped`.
- `PYTHONDONTWRITEBYTECODE=1 ${PY:-python} -B scripts/cogev.py doctor --scope all` — `50/50 checks passed`.
- `bash scripts/package_clean.sh` — completed; generated `dist/` and bytecode caches were removed again during final public hygiene cleanup.

## v3 self-bootstrap LOOP handoff efficiency closure ledger — 2026-06-24

Scope: close the run-confirmed efficiency bug where unbounded seed harvest kept spending GPT 5.5 high calls after coverage was already broad because the handoff floor double-counted `low_gain_patience`.

Status: closed in code + tests on this branch.

| Debt ID | Closure status | Code / test evidence |
|---|---|---|
| TD-V3-LOOP-HANDOFF-PATIENCE-DOUBLE-COUNT | Closed in code + tests | `_unbounded_seed_handoff_exhausted()` now uses `target_size * min_batches` as the broad-pool evidence floor; the caller still applies `low_gain_patience` as consecutive low-gain streak, avoiding duplicate patience. `tests/test_search_kernel_v3.py::test_unbounded_seed_handoff_floor_does_not_double_count_patience` replays this run's 287 accepted / 33-family handoff condition. |

Validation for this closure:

- `PYTHONDONTWRITEBYTECODE=1 ${PY:-python} -B -m pytest -q -p no:cacheprovider tests/test_search_kernel_v3.py` — `11 passed`.

## v3 self-bootstrap LOOP result projection/size closure ledger — 2026-06-24

Scope: close issues confirmed by `subbootstrap-gpt55high-loop2-handofffix-20260624-115052` artifacts without changing the research direction or reducing search breadth.

Status: closed in code + tests on this branch.

| Debt ID | Closure status | Code / test evidence |
|---|---|---|
| TD-V3-LOOP-FINAL-PROJECTION-CANDIDATE-BINDING | Closed in code + tests | `build_final_projection()` now checks `synthesis.best_current_direction.candidate_id` before falling back to an unbound synthesis answer, so adaptive `final-projection.json` can stay bound to the same displayed candidate as `final-answer.md`. |
| TD-V3-LOOP-LATENT-AUDIT-RESULT-BLOAT | Closed in code + tests | `audit_latent_replay_bundle()` now records `active_evidence_id_count` plus a bounded `active_evidence_ids` sample and source hash/preview instead of duplicating thousands of evidence ids per trace result. |

Validation for this closure:

- `PYTHONDONTWRITEBYTECODE=1 ${PY:-python} -B -m pytest -q -p no:cacheprovider tests/test_latent_replay_audit_bundle.py tests/test_nextgen_cbt_pcbg_landing.py` — `33 passed`.
- `PYTHONDONTWRITEBYTECODE=1 ${PY:-python} -B -m pytest -q -p no:cacheprovider tests/test_latent_m5_2_runtime_audit.py tests/adaptive/test_evidence_control_plane.py` — `31 passed`.
- `PYTHONDONTWRITEBYTECODE=1 ${PY:-python} -B -m compileall -q cognitive_evolve_runtime scripts tests` — passed.
- `PYTHONDONTWRITEBYTECODE=1 ${PY:-python} -B -m pytest -q -p no:cacheprovider` — `744 passed, 1 skipped`.
- `PYTHONDONTWRITEBYTECODE=1 ${PY:-python} -B scripts/cogev.py doctor --scope all` — `50/50 checks passed`.
- `bash scripts/package_clean.sh` — completed; generated `dist/` and bytecode caches were removed again during public hygiene cleanup.

## v3 self-bootstrap LOOP post-run persistence closure ledger — 2026-06-24

Scope: close issues confirmed by `subbootstrap-gpt55high-loop3-postresultfix-20260624-134900` artifacts without reducing search breadth or changing the selected research direction.

Status: closed in code + tests on this branch.

| Debt ID | Closure status | Code / test evidence |
|---|---|---|
| TD-V3-LOOP-LATENT-LEDGER-CHECKPOINT-REGRESSION | Closed in code + tests | `build_checkpoint_state()` now uses `contract_payload_for_persistence()` to strip hydrated `metadata.latent_ledger` whenever a sidecar ref exists; restore still hydrates from the ref. |
| TD-V3-LOOP-RUN-RESULT-CONTRACT-PERSISTENCE | Closed in code + tests | `NexusRuntime` now serializes text/project/resume run contracts with `contract_payload_for_persistence()`, so `run-result.json` keeps `latent_ledger_ref` without re-embedding the hydrated ledger. |


Validation for this closure:

- `PYTHONDONTWRITEBYTECODE=1 ${PY:-python} -B -m pytest -q -p no:cacheprovider tests/test_latent_live_store_persistence.py` — `2 passed`.
- `PYTHONDONTWRITEBYTECODE=1 ${PY:-python} -B -m pytest -q -p no:cacheprovider tests/test_latent_live_store_persistence.py tests/test_progress_events_match_checkpoint.py tests/test_runtime_options_and_context.py` — `11 passed`.
- Actual LOOP3 artifact replay through `contract_payload_for_persistence()` reduces the persisted contract payload from ~20.3 MB to ~1.21 MB while retaining `latent_ledger_ref`.
- `PYTHONDONTWRITEBYTECODE=1 ${PY:-python} -B -m compileall -q cognitive_evolve_runtime scripts tests` — passed.
- `PYTHONDONTWRITEBYTECODE=1 ${PY:-python} -B -m pytest -q -p no:cacheprovider` — `745 passed, 1 skipped`.
- `PYTHONDONTWRITEBYTECODE=1 ${PY:-python} -B scripts/cogev.py doctor --scope all` — `50/50 checks passed`.
- `bash scripts/package_clean.sh` — completed; generated `dist/` and bytecode caches were removed again during public hygiene cleanup.

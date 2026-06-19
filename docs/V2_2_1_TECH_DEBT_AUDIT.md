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

| Debt ID | Category | Owner subsystem | Before status | Target handling | After status | Tests proving reduction | Remaining risk | Safe to resume long run |
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

The `legacy` and `diagnostics_only` counts intentionally did not decrease in this PR because v2.2.1 adds safety-marker tests and explicit cache migration markers. This is classified as `accepted_with_test`, not hidden debt removal.

### Debt status table after implementation

| Debt ID | Before status | After status | Tests proving reduction | Remaining risk | Owner subsystem | Safe to resume long run |
|---|---|---|---|---|---|---|
| TD-HONESTY-001 | Regime shell could rely on empty or raw observations. | mitigated | `tests/test_v221_honesty_activation.py`; full suite passed. | Probe execution is deterministic and conservative; richer toolrunner ddmin regressions remain a future enhancement. | `verification/` | smoke only; 100-round deferred pending real-provider smoke. |
| TD-SOURCE-001 | No unified resolver/admission manifest. | mitigated | `tests/test_v221_source_binding_and_archive.py`; existing final-gate/materialization tests still pass. | Generic no-binding narrative candidates remain archivable for backward compatibility; source-required final gate still blocks when required. | `nexus/`, `archives/`, `ranking/` | smoke only. |
| TD-PROMPT-001 | Request types shared broad compressed payloads; audit was metadata-only. | mitigated | `tests/test_v221_prompt_audit_profiles.py`; existing context-transform tests pass. | Prompt audit writes only when caller supplies runtime audit path; no source-tree audit artifacts are written by default. | `nexus/prompt_*`, model adapter | smoke only. |
| TD-CALL-001 | Inflight calls were process memory only. | mitigated | `tests/test_v221_call_checkpoint_executor_debt.py::test_completed_unattached_call_explained_by_ledger`. | Attachment semantics are available through ledger states; full resume planner reuse policy can still be expanded. | `llm/`, `persistence/` | smoke only. |
| TD-STATE-001 | Checkpoint retained long traces/history. | mitigated | `tests/test_v221_call_checkpoint_executor_debt.py::test_thin_checkpoint_roundtrip_keeps_last_three_verification_entries`. | Candidate artifacts are not fully externalized in this PR; profile trims traces/history first. | `persistence/` | smoke only; long-run threshold must be measured. |
| TD-CONCUR-001 | Journal/cache thread safety not explicit. | mitigated | `tests/test_v221_call_checkpoint_executor_debt.py::test_verification_executor_serial_and_threaded_order`; full suite passed. | `check_with_cache` lock serializes measured cache writes; local verifier parallelism can be expanded later. | `verification/`, `llm/` | smoke only with serial/local modes. |
| TD-DIV-001 | Diversity pressure underused source-binding descriptors. | accepted_with_test | Source-binding resolver annotates candidates; parent selection adjusts resolved/invented/no-binding routes; existing search-kernel tests pass. | Quantitative descriptor-coverage improvement requires runtime smoke/long-run metrics. | `ranking/`, `nexus/search_kernel` | smoke required before 100-round resume. |
| TD-LEGACY-001 | Mixed legacy/diagnostics-only tokens. | accepted_with_test | `tests/test_v221_honesty_activation.py`; existing no-runtime-strength guard remains in suite. | Counts increased due explicit V2 cache migration and safety marker tests; classified, not deleted. | cross-cutting | smoke only; no certification from legacy strength. |

### 3-round offline smoke gate status

Executed outside the public source tree under the local test-run area after syncing to `source-current`. Results:

- Status: completed offline 3-round run.
- Completion status: `best_current_route`.
- Candidates: 16.
- Graded output: `graded_portfolio` / `NONE`; no false `verified_result`.
- Generation distribution: generation 0 = 8, generation 1 = 4, generation 2 = 4.
- Checkpoint size: about 6.7 MB for the 3-round offline run.
- Grounded information gain records: 3, all `0.0` because offline text candidates had undefined grounded signatures.
- Prompt audit lines: 0; call ledger entries: 0 because offline mode used deterministic local paths and no external LLM calls.

Long-run resume gate after offline smoke: `deferred_with_reason`. Do not resume a 100-round external-model run until a real-provider smoke produces runtime-owned prompt/call-ledger artifacts and at least one measured-strength candidate, or the contract explicitly accepts portfolio-only exploration.

### Real-provider concurrent 3-round smoke gate status

Executed after the offline smoke, still outside the public source tree and after syncing this source tree to `source-current`. Public-safe summary:

- Status: completed real-provider 3-round run.
- Provider boundary: generic OpenAI-compatible `direct_http`; model route recorded as `openai/gpt-5.5`.
- Configured concurrency: LLM governor max concurrent = 3; local verification executor = threaded local with 4 workers; search width = initial candidates 8 and branch factor 4.
- Completion status: `best_current_route`; stop reason: `adaptive_safety_checkpoint`.
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
- The smoke does **not** unlock a 100-round resume gate because it did not produce a measured-strength candidate and did not emit runtime-owned prompt-audit/call-ledger artifacts outside generated patch sandboxes.
- The concurrency smoke exposed a scheduling/observability gap: configuration-level concurrency exists, but the `run-core-self-evolve-openai.py` path still behaves mostly as a synchronous model-call loop.

New debt discovered by real-provider smoke:

| Debt ID | Category | Observed status | Required handling before long-run resume | Owner subsystem |
|---|---|---|---|---|
| TD-CALL-002 | Runtime call ledger observability | Real external calls were recorded by operator-side transport logs, but no runtime-owned durable call ledger artifact was found outside patch sandboxes. | Route real provider call state into a runtime-owned call ledger or explicitly document transport-only call evidence as insufficient for resume accounting. | `llm/`, `persistence/`, `nexus/runtime` |
| TD-PROMPT-002 | Prompt audit observability | Real provider prompts were sent, but no runtime-owned prompt audit artifact was found outside patch sandboxes. | Ensure the runtime passes and persists a prompt-audit path for model-backed smoke runs, or document why prompt-audit is intentionally opt-in and not a gate artifact. | `nexus/prompt_*`, model adapter |
| TD-CONCUR-002 | LLM scheduling concurrency | LLM governor was configured for concurrency, but live snapshots showed one active upstream model subprocess and inferred overlap peaked at 2 rather than stable configured fan-out. | Add explicit model-call fan-out where safe, or state that current Nexus core loop is width-expanded but synchronously scheduled; include a deterministic concurrency smoke assertion. | `llm/`, `nexus/loop`, runner scripts |

Long-run resume gate after real-provider smoke: `deferred_with_reason`. Do not resume a 100-round external-model run until TD-CALL-002 and TD-PROMPT-002 are closed or deliberately waived, and until at least one measured-strength candidate appears or portfolio-only continuation is explicitly accepted.

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
- `legacy` and `diagnostics_only` remain accepted-with-test safety markers; this pass did not blind-delete compatibility coverage.
- No outdated test still names `closure_certificate` as the solved authority.

## Concurrent verifier plumbing follow-up — 2026-06-18

Scope: minimal runtime change to make the already configured verification concurrency real in the core round path while preserving serial generation/mutation semantics.

Changes made:

- `verification_stack.verify_population` now runs candidates through a bounded `ThreadPoolExecutor` when `COGEV_VERIFY_CONCURRENCY` permits it. The shared formal-signature accumulator is accessed via a lock and candidates are returned in input order.
- `verification.obligation_runner.run_obligations_for_population` now checks candidates concurrently inside each obligation, with cache access guarded by a lock and result order preserved by candidate index.
- `nexus.loop.round.critique_and_verify` now runs the verifier stack, synthesized verifier, and verification-obligation runner as three concurrent entrypoints when concurrency is enabled.
- At this verifier-only follow-up stage, `plan_mutations` and offspring generation remained serial because they relied on ordered stateful search controls; see the TD-CONCUR-002 closure section below for the later model batch fan-out change.
- `COGEV_VERIFY_CONCURRENCY=1` is the deterministic serial fallback for local debugging and regression isolation.

Local validation added:

- Targeted concurrent verifier tests cover serial fallback, per-candidate overlap, obligation result order, obligation-cache population, three-entrypoint overlap, and journal-line integrity under concurrent writes.
- Focused regression tests for v2.2.1 honesty activation and proof-progress hardening passed after the concurrency patch.
- Full local suite after cleanup: `664 passed, 1 skipped`.
- Doctor: `50/50 checks passed`.
- Public hygiene test: `5 passed`.
- Test function definitions: 649; duplicate test function names: 0.

Debt status:

- TD-CONCUR-002 was partially mitigated at the runtime scheduling layer by this verifier-only patch; this status is superseded by the TD-CONCUR-002 model fan-out closure section below.

## TD-CONCUR-002 model fan-out closure — 2026-06-18

Scope: close the remaining scheduling gap where the LLM governor allowed concurrency but the Nexus seed/offspring model-call loops still submitted batches synchronously.

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
- A fresh real-provider smoke is still recommended before a long external-model run, but it is now validation of provider/account behavior rather than missing runtime scheduling capability.

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

Remaining before PR publication:

- Re-run final hygiene immediately before staging/commit if additional files change.
- Open draft PR from `mzz/v2.3-theory-runtime-model-routes` after successful local validation.

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
- Future real-provider high-ceiling runs should strengthen seed prompt/schema pressure toward mathematical models, exploitable theorems, cross-domain mechanisms, and performance algorithms rather than runtime-contract restatement.

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
- TD-V23-HIGH-CEILING-SMOKE remains closed only as a smoke/mechanism check, not as proof of high-quality theory discovery. The next real-provider run should be launched only after PR hygiene/CI and should use stronger seed-field pressure for edge theorem/cross-domain/performance content.

## v3 Exploration Fabric Phase 0 ledger — 2026-06-19

Status: closed in this phase branch.

- `TD-V3-P0-FABRIC-STATE` — Closed. Added domain-neutral `cognitive_evolve_runtime.fabric` primitives for advisory dossiers, tasks, task graphs, typed fabric config, and checkpoint fabric state without wiring behavior into the runtime loop.
- `TD-V3-P0-CHECKPOINT-COMPAT` — Closed. Added optional `NexusCheckpoint.fabric` field and default `{}` restore behavior so legacy checkpoints continue to load.
- `TD-V3-P0-ADVISORY-GUARD` — Closed. Added advisory authority-key guards and regression tests to ensure new fabric advisory payloads cannot carry verification-authority fields.

Validation requirements for the phase remain: compileall, full pytest, doctor, package clean, and public hygiene before PR.

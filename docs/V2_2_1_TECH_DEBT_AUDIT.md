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

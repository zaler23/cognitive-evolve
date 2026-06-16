# CognitiveEvolve Agent System

CognitiveEvolve `2.0.0` is a standalone **Nexus-only model-driven offline evolution runtime** with an OpenAI-compatible API surface.

## Project status

CognitiveEvolve is a **source-installable beta / engineering preview**. Users
can clone the repository, install it from source, run the CLI, and serve the
OpenAI-compatible API locally. The project does not yet promise a packaged
public installer, PyPI release, hosted service, or one-click production
deployment.

Text, research, and project inputs enter one runtime pipeline:

```text
Input Packet
  → World Model
  → Objective Contract
  → Evolution Policy
  → CandidateGenome / ProjectCandidateGenome population
  → Local verification and tool feedback
  → Relative ranking + multihead scores
  → Archives + diagnosis + mutation/crossover
  → Final answer, patch, report, or structured failure analysis
```

The platform fixes evolution mechanics: snapshots, file hashes, candidate lineage, archive fates, local tool protocol, patch sandboxing, persistence, checkpoint replay, and progress events. The model decides task semantics through structured objective contracts, evolution policies, diagnoses, ranking judgments, and mutation plans.

## Runtime architecture

The runtime source of truth is:

```text
cognitive_evolve_runtime/nexus/runtime.py       # NexusRuntime, the single execution authority
cognitive_evolve_runtime/nexus/loop/            # structured evolution loop package
cognitive_evolve_runtime/nexus/difficulty_estimator.py # model/task difficulty and round-budget estimation
cognitive_evolve_runtime/nexus/policy.py        # EvolutionPolicy and policy updates
cognitive_evolve_runtime/nexus/diagnosis.py     # stagnation/control diagnosis
cognitive_evolve_runtime/nexus/model_adapter.py # public adapter facade only
cognitive_evolve_runtime/nexus/model_adapter_core.py    # transport, prompt-view, schema validation core
cognitive_evolve_runtime/nexus/model_adapter_schemas.py # structured response schemas
cognitive_evolve_runtime/nexus/model_adapter_repair.py  # deterministic schema repair
cognitive_evolve_runtime/nexus/model_adapter_facets/    # protocol-specific adapter facets
cognitive_evolve_runtime/nexus/stage_policy/    # stage eligibility package
cognitive_evolve_runtime/nexus/state.py         # runtime-state projections
```

Shared foundations:

```text
cognitive_evolve_runtime/candidates/     # CandidateGenome, ProjectCandidateGenome, mutation, crossover, patch merge
cognitive_evolve_runtime/archives/       # ArchiveManager facade, archive registry, constraints, and fate-specific archives
cognitive_evolve_runtime/ranking/        # RelativeRater, multihead Elo, parent selection, novelty, lineage saturation
cognitive_evolve_runtime/inputs/         # text packets, project snapshots, project maps, context selection
cognitive_evolve_runtime/tools/          # verifier environment, tool runner, patch sandbox, feedback
cognitive_evolve_runtime/persistence/    # population/archives/event/checkpoint/verification stores
cognitive_evolve_runtime/events/         # event bus and progress event schemas
cognitive_evolve_runtime/contracts/      # objective contract definitions
cognitive_evolve_runtime/evidence/       # evidence planning and ledger
cognitive_evolve_runtime/durable/        # file locks and atomic writes
```

There are no alternate runtime, ranking, archive, or candidate-search packages. New development should extend the modules above. Third-party systems can be integrated only through provider/tool/adapter boundaries; they are not declared or selected as replacement runtimes.

## Install

```bash
python3 -m pip install -e .
```

For deterministic tests and demos:

```bash
export COGEV_LLM_PROVIDER=fixture
export COGEV_LLM_FIXTURE="$PWD/tests/fixtures/llm_fixture.json"
```

For real model use, configure a generic provider explicitly. Runtime code talks to `llm.provider_interface.LLMProviderInterface`; supported public modes are `litellm`, `direct_http` for OpenAI-compatible `/v1/chat/completions`, and deterministic `fixture` for tests. Tests default to hermetic mode and never read user-home `.env` files. The public project does not ship a private application model relay or provider-specific local integration.

```dotenv
COGEV_LLM_PROVIDER=litellm
COGEV_LLM_MODEL=provider/model-id
COGEV_LLM_API_BASE=https://your-provider.example/v1
COGEV_LLM_API_KEY=<upstream-provider-key>
COGEV_SERVER_API_KEY=<frontend-service-key>
```

To create a deployment config without committing secrets:

```bash
python3 scripts/cogev.py config init --profile local --output .env
python3 scripts/cogev.py config init --profile production --output /secure/path/cogev.env
python3 scripts/cogev.py config init --profile fixture --print
```

Generated `.env` files are ignored by Git. Only `.env.example`, `.env.production.example`, and `.env.fixture.example` belong in source control.

## CLI

```bash
python3 scripts/cogev.py config init --profile local --output .env
python3 scripts/cogev.py llm status
python3 scripts/cogev.py route "your task"
python3 scripts/cogev.py enhance "your task"
python3 scripts/cogev.py run "your task"
python3 scripts/cogev.py runtime run <task_dir> --all
python3 scripts/cogev.py runtime run <task_dir> --rounds 3
python3 scripts/cogev.py runtime status <task_dir>
python3 scripts/cogev.py eval run <task_dir>
python3 scripts/cogev.py optimize run <task_dir>
python3 scripts/cogev.py doctor --scope all
```

The runtime command has no runtime selector; it always invokes `NexusRuntime`.

## OpenAI-compatible API

```bash
python3 scripts/cogev.py api status
python3 scripts/cogev.py api serve
```

For local development, `scripts/start-cognitive-evolve-api.sh` can prepare an
isolated virtual environment outside the source tree and start the API. The
launcher installs the project non-editably into that external environment so it
does not create `.venv/` or `*.egg-info/` inside the repository. By default it
refuses to stop another process already listening on the service port; set
`COGEV_STOP_EXISTING_PORT=1` only when you intentionally want the launcher to
stop that listener first.

Frontend configuration:

```text
Base URL: http://127.0.0.1:8765/v1
API Key:  <COGEV_SERVER_API_KEY>
Model:    cognitive-evolve-one-shot-deep
```

For long frontend runs, prefer `/v1/cogev/jobs` over holding one chat-completions request open. Streaming chat completions emit progress metadata while Nexus writes durable progress and checkpoint artifacts. API model tiers now select adaptive Nexus policies rather than fixed round/candidate counts. `cognitive-evolve-one-shot-exhaustive` activates an exhaustive policy with safety checkpoints, dynamic seed batching/deduplication, and wider mutation branching. Reaching a safety checkpoint returns `needs_continuation`. The only allowed early-stop statuses before the safety cap are `candidate_ready_for_external_review` and `diminishing_returns_checkpoint`; both produce reviewable candidate output, not a correctness claim. Only a verifier/model `objective_solved` signal marks the objective solved. Legacy `COGEV_NEXUS_PROFILE_*_ROUNDS`, `*_CANDIDATES`, `COGEV_MUTATION_BRANCH_FACTOR`, and `COGEV_ACTIVE_POOL_LIMIT` are ignored by default to prevent stale local `.env` files from pinning or narrowing adaptive runs. API calls bind Nexus to the configured generic LLM adapter rather than the deterministic offline seed fallback.

Compatibility boundary: `/v1/chat/completions` is OpenAI-shaped, not
token-semantic identical. A single request may run adaptive multi-round
candidate evolution before a solved/best-effort/needs-continuation artifact,
and streaming sends progress, heartbeat, and final-answer chunks rather than
raw provider token deltas.

Security defaults: the API refuses to serve on a non-loopback host with
`COGEV_SERVER_REQUIRE_AUTH=false` unless `COGEV_ALLOW_INSECURE_BIND=1` is set
explicitly. CORS defaults to localhost origins; wildcard origins disable
credentials.

## Runtime artifacts

A task run writes Nexus artifacts under the task directory:

```text
runtime-state.json
nexus-runtime/run-result.json
nexus-runtime/final-answer.md
nexus-runtime/population.json
nexus-runtime/archives.json
nexus-runtime/checkpoint.json
nexus-runtime/events.jsonl
nexus-runtime/candidate-journal.jsonl
nexus-runtime/rounds/round-*.json
nexus-runtime/adaptive/adaptive-state.json
nexus-runtime/adaptive/final-certificate.json
nexus-runtime/adaptive/final-projection.json
nexus-runtime/adaptive/spatial-topology.json
nexus-runtime/challenge-memory.json
nexus-runtime/challenge-events.jsonl
nexus-runtime/nexus-runtime-self-check.json
nexus-runtime/nexus-runtime-self-check.md
evaluations/native-eval-report.json
evaluations/native-eval-report.md
```

Progress events distinguish pipeline progress from evolution progress. Checkpoints contain mode, contract, world, policy, diagnosis, population, archives, budget history, and round state so Nexus runs can resume from saved state. Live persistence writes after ranking/critique, after mutation, on interruption, and at final synthesis; initial exploration seeds are marked as seeds and are not returned as final answers unless evolved or synthesized.

Fallbacks are auditable runtime events, not silent logger-only behavior.
`run-result.json` stores `evolution.fallback_events` and
`evolution.fallback_event_count`; `nexus-runtime/events.jsonl` stores the same
sanitized fallback event summaries. These summaries redact local paths and
secret-shaped text and do not include long prompts or provider credentials.

The optional Adaptive Evidence Layer is disabled unless configured by
environment, `.cogev/config.yaml`, or a task-local `task.yaml`. Its public
surface is evidence-oriented: `ArtifactPolicy`, `EvidenceRecord`,
`ChallengeMemory`, `SearchPressure`, optional external evaluator feedback,
observe/advisory spatial telemetry, checkpointable adaptive state, clean final
projection, and a final certificate.

The Evidence Control Plane keeps search and finality separate. Artifact policy
decides whether a candidate may be probed or finalized; evidence records update
search value and repair value; challenge memory turns failures and boundaries
into search pressure for the next mutation round. A configured external
evaluator is treated as objective evidence, while model self-claims such as
"verified" do not by themselves solve the objective. Machine-artifact tasks can
set `adaptive.evidence.machine_artifact_required=true`; natural-language
fallback artifacts may then be probed but are not final-eligible until re-emitted
as clean machine-readable artifacts. The runtime turns configured artifact
policy into a bounded prompt hint so model-backed mutation sees the exact
`artifact_type`, required fields, forbidden aliases, and optional domain
vocabulary. Artifact normalization, semantic-drift diagnostics, and
score-component diagnostics become challenge-memory cases rather than final
claims. When no candidate is certified solved, the final projection still emits
a best-current candidate when one is available, and its JSON projection preserves
structured machine artifacts instead of string-wrapping them.

Adaptive research extensions are implemented as an internal registry under
`AdaptiveRuntimeController`, not as a second runtime or a parallel research
control plane. Extensions emit `ResearchSignal` objects that are applied through
one deterministic applicator with explicit modes: `observe` keeps only metrics
and warnings, `advisory` allows search pressure and parent-selection advisory
without writing extension evidence, and `active` is the only mode that may write
extension evidence records or blocking final-gate directives. Extensions never
own candidate fate, challenge truth, archive state, or final solved authority.
Spatial research selection reuses the existing adaptive spatial population
state, so there is only one candidate-coordinate authority. The optional
research snapshot is consolidated under `nexus-runtime/research/` as
`research-state.json`, `research-events.jsonl`, and `research-metrics.json`.

Research extension authority boundaries are fixed: `NexusRuntime` orchestrates,
`AdaptiveRuntimeController` owns adaptive/research entry, `ArchiveManager` owns
candidate fate, `ChallengeMemory` owns challenge relations, `EvidenceRecord`
helpers own candidate evidence, `ParentSelector` consumes advisory, and
`FinalProjection` owns user-facing output. Pattern memory, immune/necropsy,
budget backpressure, MDL compression, parameter sweep, chaos, BFT quorum,
context pruning, and contract refinement are advisory or gate-directive
extensions only; none can silently mutate the objective contract or mark an
objective solved without clean artifact and high-authority evidence.

## Testing and validation

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
python3 scripts/cogev.py doctor --scope all
```

Hermetic tests stay provider-free. Real provider smoke coverage is opt-in:

```bash
COGEV_RUN_LLM_TESTS=1 python -m pytest -q tests/test_llm_opt_in_integration.py
```

### Nexus prompt budget

Nexus persists full evolution state locally, but sends compressed prompt views to the configured model. Use `COGEV_NEXUS_PROMPT_MAX_CHARS` for ordinary calls and `COGEV_NEXUS_LONG_CONTEXT_MAX_CHARS` for long-context calls. Prompt-view accounting is written to `nexus-runtime/run-result.json` under `evolution.prompt_view_metadata`.

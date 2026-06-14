# Nexus Runtime Design

## Design goal

Nexus is a model-driven offline evolution runtime. It is not a one-shot agent and not a domain-specific workflow engine. The runtime provides stable evolution mechanics while a model supplies task semantics for each run.

## Platform-fixed mechanics

Nexus owns:

- input snapshots and file hashes;
- world-model construction;
- objective-contract storage and hashing;
- candidate genome shape;
- lineage tracking;
- multi-archive storage;
- relative ranking protocol;
- local tool feedback protocol;
- project patch sandboxing;
- budget accounting;
- event logging;
- checkpoint/replay state;
- final artifact writing.

## Model-decided semantics

The model can decide:

- what the task goal means for this input;
- unacceptable substitutes and allowed output forms;
- fitness axes and candidate niches;
- which mutation operators are useful;
- which context slices are needed;
- relative candidate comparisons;
- stagnation diagnosis and policy updates;
- synthesis rules for the final answer, patch, report, or failure analysis.

## CandidateGenome

`candidates.genome.CandidateGenome` is the unit of evolution. It stores artifact, claim, core mechanism, assumptions, missing parts, edge seeds, inherited genes, mutation history, tool results, verification trace, novelty descriptors, niche memberships, failure lessons, multihead scores, and current fate.

`ProjectCandidateGenome` extends it with patch operations, touched files/symbols, expected effects, affected tests, risk notes, patch application results, commands run, and verification results.

## EvolutionPolicy

`nexus.policy.EvolutionPolicy` records candidate niches, fitness axes, mutation operators, archive schema, parent-selection preferences, culling principles, rarity budget, tool preferences, stagnation actions, and synthesis policy. The policy preserves multihead scoring data instead of compressing candidates into one scalar.

## ArchiveManager

`archives.manager.ArchiveManager` routes candidates to answer, mechanism, novelty, rarity, dormant, auxiliary, failure, project-patch, and quality-diversity archives. Low-scoring but rare candidates can remain available through rarity or dormant storage. Failed candidates can still contribute failure lessons and inheritable genes.

Candidate fates have one canonical lifecycle meaning in `nexus/fate_semantics.py`:
`Active` and `Elite` are live material, `Incubating` is a bounded repair lane,
`Dormant` is parked material that can only return through an explicit
reactivation/repair path, `Auxiliary` is support material, and `Culled`/`Failed`
are terminal for the candidate itself. A failed candidate can seed a new repair
offspring only through a separate extraction path; it is not silently treated as
a live parent.

## RelativeRater

`ranking.relative_rater.RelativeRater` compares candidates by final-answer promise, core mechanism strength, mutation value, rare-knowledge value, auxiliary value, dominance, crossover pairings, and dormant/reactivation value. Ranking outputs feed multihead Elo and parent selection.

## MutationPlanner

`candidates.mutation.MutationEngine` supports Deepen, Repair, Simplify, Specialize, Generalize, Invert, Transfer, RareInject, CrossOver, AdversarialPatch, ToolGround, CoreExtraction, ScaffoldRemoval, and Dormant reactivation actions. Operators modify inheritable mechanisms and traces, not only surface text.

## Text input path

```text
TextInputPacket
→ TextWorldModel
→ ObjectiveContract
→ EvolutionPolicy
→ seed population
→ tools/model checks
→ relative ranking
→ archive update
→ diagnosis and mutation
→ synthesis
```

Seeds include direct, known-pattern, edge-knowledge, analogy, inversion, decomposition, tool-grounded, adversarial-critic, counterexample/negative-construction, wildcard, and rare-recall candidates. If a model returns a narrow seed pool, Nexus amplifies it to the active profile width while marking supplemental seeds as `search_seed_not_final`.

## Project input path

```text
ProjectSnapshot
→ ProjectWorldModel
→ ContextPacket
→ ProjectCandidateGenome
→ PatchSandbox
→ ToolRunner / VerificationTrace
→ Relative Project Ranking
→ Patch Mutation / Crossover
→ Final Patch / Report
```

Project input is never fed as one oversized prompt. The model can request files, symbols, or tests; `ContextSelector` returns bounded slices.

## Tool feedback

`tools.feedback.ToolFeedback` separates input evidence, tool evidence, and model hypothesis. Local verification can run compileall, pytest, or other installed adapters. Tool output is written to the candidate's tool results and verification trace.

## Search diagnosis

`nexus.diagnosis.SearchStateDiagnoser` inspects recent population history, archive distribution, tool feedback, failure signatures, mutation history, ranking history, budget use, and current policy. It emits a `SearchDiagnosis` with stagnation type, over/under-explored families, semantic-drift risk, auxiliary-collapse risk, prematurely culled genes, and recommended actions.

## Events and progress

Nexus emits separate pipeline and evolution progress events. Pipeline progress tracks stages. Evolution progress tracks round, population, active/dormant/archive counts, tool calls, current best answer, current auxiliary candidate, diagnosis, and next action.

## Persistence and replay

Population, archives, events, candidate journals, round snapshots, checkpoints, and verification traces are persisted under `nexus-runtime/` using durable writes. Checkpoints include mode, contract, world, policy, diagnosis, population, archives, progress event, and budget history. Live persistence runs after ranking/critique, after mutation, on error checkpoints, and at final synthesis, so interrupted runs still leave a resumable state.

Final snapshot files (`population.json`, `archives.json`, `checkpoint.json`,
`final-answer.md`, and `run-result.json`) are published through
`persistence.transactional_snapshot.NexusSnapshotTransaction`. Event JSONL
remains append-only; the coherent snapshot manifest records file hashes and a
transaction id so readers can distinguish a complete generation from an
interrupted write.

## Avoiding high-frequency loops

Nexus forces search diversity through rare seeds, rarity archives, dormant reactivation, novelty bonuses, underexplored niche bonuses, RareInject, CoreExtraction, and ScaffoldRemoval. Edge knowledge remains a seed until input or local tool evidence supports it.


## Absorbing old adaptive strengths without restoring legacy

Nexus keeps the old runtime's useful exploration pressure as first-class Nexus mechanics: profile-controlled candidate width, branch factor, structured critique, broad mutation palettes, rare/dormant preservation, and mid-loop checkpoints. It does not restore `adaptive_engine`, `candidate_search`, `optimizer`, or `archive` as parallel authorities.

Initial seeds are search instructions, not answers. If no non-seed candidate emerges, synthesis returns a failure/interruption report instead of echoing a `Direct Solver Seed` as the final answer.

## Current package boundary

Runtime behavior belongs in the Nexus packages listed in `docs/ARCHITECTURE_BOUNDARIES.md`. Do not add duplicate ranking/archive/runtime packages and do not add wrapper modules for absent runtime namespaces.


## API model tiers and LLM-backed Nexus

The OpenAI-compatible API is not a separate runtime. It writes the request artifact, activates the request-local model profile, and then calls `EngineOrchestrator`, which resolves the Nexus round budget and model adapter.

Model tiers select adaptive internal search policy:

- API model caps default to `0`, meaning adaptive Nexus policy rather than a fixed profile round count.
- Profile safety limits are checkpoints; they never mean the objective is solved.
- `deep`, `ultra`, and `exhaustive` increase minimum stop depth, safety window, and branch factor, while candidate width is derived from policy diversity or explicit operator floors.
- `completion_status=needs_continuation` is returned when the safety checkpoint is reached without a stop/verifier signal.

For API calls, Nexus uses `StructuredModelAdapter.from_configured_llm()` unless a caller injects a model adapter. This means objective-contract generation, policy generation, seed population, relative ranking, diagnosis, policy update, mutation planning, offspring generation, and final synthesis can all be LLM-backed. Deterministic seed/ranking logic is reserved for hermetic tests and direct offline calls with no configured adapter.

## LLM provider boundary

Nexus runtime code calls the LLM layer through `llm.provider_interface.LLMProviderInterface`. `llm.litellm_provider.LiteLLMProvider` is the default concrete implementation and `llm.mock_provider.MockLLMProvider` is available for deterministic tests. This keeps the transport layer replaceable without adding another runtime path.

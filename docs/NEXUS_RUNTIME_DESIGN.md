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
- isolated temporary patch copies (not a security sandbox);
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

## Offline mutation engine

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

Generation zero crosses model-authored search families with six domain-neutral
cognitive operations: direct mainstream attack, cross-domain transfer, edge
knowledge, counterexample probe, representation shift, and tool probe. The
model chooses the useful disciplines, mechanisms, and representations for the
task. Family/axis labels are coverage receipts, not capability or fitness
claims, and each accepted seed begins an independent lineage root. With a
configured model, Nexus does not deterministically top up a narrow or empty
seed result; deterministic amplification is limited to the explicit offline
path where `model is None`.

## Project input path

```text
ProjectSnapshot
→ ProjectWorldModel
→ ContextPacket
→ ProjectCandidateGenome
→ isolated temporary patch copy
→ ToolRunner / PreliminaryValidationTrace
→ Relative Project Ranking
→ Patch Mutation / Crossover
→ Final Patch / Report
```

Project input is never fed as one oversized prompt. The model can request files, symbols, or tests; `ContextSelector` returns bounded slices.

## Tool feedback

`tools.feedback.ToolFeedback` separates input evidence, tool evidence, and model hypothesis. Preliminary checks can run in-process compilation and static adapters; project tests or other repository code run only after explicit operator opt-in. Tool output is written to the candidate's tool results and preliminary validation trace.

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


## Productive branch allocation

After preliminary outcomes exist, Productive Branch Allocation assigns real
offspring slots by lineage-root UCB. Credit comes only from runtime-observed
productive events such as a new valid outcome cell, full-metric Pareto
improvement, newly resolved challenge, applied and verified patch, or surviving
evaluator/verifier pass. Exact duplicates, failed observations, and unbound
children receive no productive credit. Sibling producers share one observed
event reward so fan-out does not multiply evidence.

Each slot binds one runtime-generated slot ID, lineage root, primary parent, and
semantic directive. Every model-backed round sends all allocated slots in one
runtime-authored lineage envelope. The default `slot` mode makes one
offspring-generation call per allocated slot; explicit `single_batch` mode makes
one call for the complete slot manifest. The normal model-backed path bypasses
mutation planning. Deterministic mutation planning exists only for explicit
offline operation where `model is None`, as a compatibility path.

Initial seeds are search instructions, not proof of completion. Synthesis
returns reviewable best-current answer material when available. Producer-owned
artifacts always keep `objective_solved=false`; an external reviewer owns
correctness and final acceptance.

## Current package boundary

Runtime behavior belongs in the Nexus packages listed in `docs/ARCHITECTURE_BOUNDARIES.md`. Do not add duplicate ranking/archive/runtime packages and do not add wrapper modules for absent runtime namespaces.


## API model tiers and LLM-backed Nexus

The OpenAI-compatible API is not a separate runtime. It writes the request artifact, activates the request-local model profile, and then calls `EngineOrchestrator`, which resolves the Nexus round budget and model adapter.

Model tiers select adaptive internal search policy:

- API model caps default to `0`, meaning adaptive Nexus policy rather than a fixed profile round count.
- Profile safety limits are checkpoints; they never mean the objective is solved.
- `deep`, `ultra`, and `exhaustive` increase minimum stop depth, safety window, and branch factor, while candidate width is derived from policy diversity or explicit operator floors.
- `completion_status=completed` is returned when the safety checkpoint yields answer material; explicit interruption/quota/operator continuation remains separate metadata.

For API calls, Nexus uses `StructuredModelAdapter.from_configured_llm()` unless a caller injects a model adapter. This means objective-contract generation, policy generation, seed population, relative ranking, diagnosis, policy update, offspring generation, and final synthesis can all be LLM-backed. Model-backed reproduction uses the single runtime lineage envelope above rather than a separate planning call; its default slot transport still makes one offspring-generation call per allocated slot. Deterministic seed/ranking logic and mutation planning are reserved for hermetic tests and direct offline calls with no configured adapter.

## LLM provider boundary

Nexus runtime code calls the LLM layer through `llm.provider_interface.LLMProviderInterface`. `llm.litellm_provider.LiteLLMProvider` is the default concrete implementation and `llm.mock_provider.MockLLMProvider` is available for deterministic tests. This keeps the transport layer replaceable without adding another runtime path.

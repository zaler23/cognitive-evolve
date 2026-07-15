# Core Evolve Algorithm — Nexus 2.0

Nexus evolves structured task artifacts rather than free-form answer strings.
Every candidate is a `CandidateGenome` or `ProjectCandidateGenome` carrying its
artifact, lineage, mechanism, assumptions, missing parts, model-chosen niches,
formal or tool evidence, verification trace, archive memberships, failure
lessons, fate, and multihead scores.

## Authority boundary

The runtime owns evolution mechanics: exact input identity, frozen objective
contract, candidate IDs, lineage, branch-slot bindings, runtime observations,
archives, local tool protocol, patch worktrees, events, checkpoints, and replay
boundaries. The model owns task semantics: objective-level candidate families,
useful disciplines and representations, concrete mechanisms, artifacts, and
task-specific variation.

Model output cannot author runtime facts. Model-claimed evaluator results,
verification state, patch results, evidence progress, terminal controls, IDs,
generation, or lineage remain diagnostic claims only. The runtime writes those
fields from observed execution and parent bindings.

All producer-owned checks are preliminary. Nexus may produce and improve an
answer, proof attempt, design, or patch, but it does not issue the external
correctness verdict. Producer-owned projections keep `objective_solved=false`;
independent review or a separately owned verifier remains the final authority.

## Search loop

Nexus is a model-driven iterative search: the runtime allocates and records
branches, while the model authors each task-specific semantic variation.
This is not a claim that every mutation/crossover is a
biological genetic operator.

```text
Input Packet
→ World Model
→ Frozen Objective Contract
→ Model-authored or objective-derived Search Space
→ Family × Cognitive-Axis Seed Portfolio
→ Preliminary Evaluation and Relative Ranking
→ Archive Update and Parent Selection
→ Grounded Productive Branch Allocation
→ One Runtime Lineage Envelope
→ Direct Model Artifact Evolution
→ Preliminary Evaluation of Offspring
→ Synthesis and External-Review Projection
```

An operator may supply concrete generation-zero incumbents. They enter the same
population, count toward seed coverage, and retain independent lineage roots.
When no model is configured, deterministic seed amplification and mutation are
available as explicit offline behavior; they are not mixed into model-backed
generation as hidden top-up or fallback.

## Outcome-ready seed portfolio

The initial population covers each active objective-level candidate family
through six task-neutral cognitive operations:

1. `direct_mainstream` — strongest direct mechanism and its assumptions;
2. `cross_domain_transfer` — a structural transfer from a model-chosen field;
3. `edge_knowledge` — uncommon results, boundary cases, or specialist
   heuristics that materially alter the artifact;
4. `counterexample_probe` — adversarial cases and violated assumptions;
5. `representation_shift` — a different representation, decomposition, scale,
   abstraction, or formalism;
6. `tool_probe` — a concrete calculation, experiment, retrieval, simulation,
   or executable probe.

This is a family-by-operation portfolio, not a hard-coded discipline ontology.
The model chooses relevant domains, lenses, and tools. Slots require materially
distinct artifacts, pairwise non-redundant model-chosen niches, and observable
evaluation dimensions. Each axis also has a typed receipt: a direct-mechanism
claim, transfer source, edge-knowledge seed, challenged assumption,
representation `from`/`to`, or tool-probe plan.

Receipts measure whether the requested portfolio was delivered; they do not
prove usefulness. Missing receipts create a coverage shortfall rather than
inventing capability, and seed labels never earn productive-branch reward.
Every accepted model seed starts an independent lineage root so early portfolio
width is not collapsed into one synthetic ancestry.

## Grounded productive branch allocation

Productive Branch Allocation (PBA) allocates the next round only after runtime
outcomes exist. Its bandit arm is the lineage root, not a prose mechanism label
or a model-selected arm name. Positive credit comes from observed events:

- entering a new categorical evaluator/verification/patch outcome cell;
- a full-metric Pareto improvement over an earlier candidate in the same cell;
- resolving a previously open named challenge;
- producing an applied artifact with passing verification; or
- surviving an observed evaluator or verifier pass.

Exact phenotype duplicates, terminal failures, and failed verification receive
no productive credit and add risk. Candidates created in the same round and
generation form a cohort: siblings that expose the same new event share its
reward, preventing fan-out or iteration order from multiplying one observation.
Seeds establish baselines but do not count as reproduction pulls.

PBA uses lineage-root UCB with an exploration floor. Every selected unobserved
lineage receives a trial before reward-driven deepening when slots permit.
Allocated slots bind a runtime-generated slot ID, lineage root, and primary
parent. A model cannot redirect a slot by claiming another parent or arm, and a
slot is credited at most once.

## Model-backed generation envelope

Every model-backed reproduction round uses one runtime-authored lineage
envelope. The envelope contains the selected parents, the grounded PBA slot
manifest, the frozen task, and bounded evaluator feedback. It does not prescribe
a named mutation operator: the model chooses the concrete semantic strategy and
returns complete evaluator-visible artifact variants for the allocated slots.

The default `slot` mode requests one model response per allocated slot;
explicit `single_batch` mode requests the complete slot manifest in one
generation call. There is no sequential semantic top-up after duplicates or
short output. A schema-valid empty batch is an explicit abstention: an already accepted
preliminary incumbent may continue to the next registered round; without one,
the run checkpoints. Provider, transport, and nonempty invalid-response errors
remain explicit failures rather than activating a second generation path.

The runtime binds each returned child to its primary parent and lineage root,
then assigns candidate ID, generation, lineage, initial `Active` fate, and
runtime-control fields. Artifact content remains model-authored. A bare task
artifact may be preserved when it satisfies the explicit dynamic artifact
schema or when the single lineage envelope plus the frozen task makes the
preliminary evaluator the next authority. Frozen problem text alone is not an
artifact validator, and the runtime never manufactures an artifact from claim
or mechanism prose.

## Input integrity and token efficiency

Complete population, archive, checkpoint, and journal state remains local.
Model requests receive task-specific prompt views: nonessential history,
archive exemplars, repeated traces, and unselected candidates may be summarized
or omitted with hashes and accounting metadata instead of being copied into
every request.

The frozen problem, exact task specification, selected-parent artifacts,
generation envelope, seed-portfolio contract, and explicitly protected source
context are not silently truncated. If protected content plus the structured
response schema cannot fit the effective provider limit, the call fails and the
run checkpoints instead of replacing the input with a lossy excerpt. A positive
transport clamp is included in the effective limit before prompt construction;
an unlimited transport setting does not impose a hidden lower cap.

This boundary improves token efficiency without reducing search width or
changing the local source of truth. For tasks with large artifacts, domain
runners should evolve the smallest useful locus and use a deterministic
materializer for the complete evaluator-visible artifact.

## Proof and evidence progress

For proof-like objectives, narrative elaboration alone is not progress. Search
telemetry requires concrete formal objects—such as equations, constructions,
case analyses, witnesses, counterexamples, derivations, or proof steps—and an
`obligation_delta` against named obligations. Duplicate formal signatures are
deduplicated, while repeated object absence or ledger non-progress adds search
pressure toward a formal artifact, a counterexample, or route refutation.

`proof_progress`, evidence deltas, tool output, preliminary evaluation, and
model judgment guide selection and reproduction only. They do not prove the
objective and cannot become a second final authority. If the run improves a
route without externally closing it, synthesis exposes the best current
direction and its open obligations rather than self-certifying a solution.

## Ranking and archives

Ranking is relative and multihead. Nexus preserves objective alignment, answer
likelihood, mechanism strength, novelty, rarity, verifiability, coherence, tool
progress, robustness, simplicity, transfer potential, auxiliary value, and
deferral risk instead of collapsing every decision into one total score.

When a configured preliminary evaluator has measured candidates, its
pass/metric ordering controls the incumbent. Model ranking cannot replace an
evaluated incumbent with an unevaluated self-scored candidate. The incumbent is
protected through compaction and projected from the exact evaluated artifact
snapshot and content hash; unbound synthesis prose is not promoted as that
artifact.

Archives preserve answer elites, mechanism elites, rare and edge candidates,
dormant repair material, auxiliary candidates, project patches, and useful
failures. Materialized artifact identity is the hard phenotype key. Two
candidates with the same evaluator-visible artifact are one phenotype even if
their labels, niches, or lineage stories differ. Distinct artifacts with the
same categorical outcome remain distinguishable, but prose labels alone do not
create reported diversity.

## Project input

Project runs snapshot files, build a world model, select exact source context,
generate patch candidates, apply each patch in an isolated temporary copy, run
operator-selected preliminary checks, and feed structured outcomes back into
the population. Reproduction refreshes source context from selected-parent
bindings and the current mutation objective. These temporary copies are not a
security sandbox, and test execution remains operator opt-in.

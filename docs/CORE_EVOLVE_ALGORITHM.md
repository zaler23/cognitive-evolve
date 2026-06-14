# Core Evolve Algorithm — Nexus 2.0

Nexus evolves structured candidates instead of free-form answer strings. Every candidate is a `CandidateGenome` or `ProjectCandidateGenome` with lineage, artifact, mechanism, assumptions, missing parts, edge seeds, mutation history, tool results, verification trace, novelty descriptors, archive memberships, failure lessons, fate, and multihead scores.

## Loop

```text
Input Packet
→ World Model
→ Objective Contract
→ Evolution Policy
→ Seed Population
→ Local Verification / Tool Feedback
→ Relative Ranking
→ Archive Update
→ Search Diagnosis
→ Policy Update
→ Parent Selection
→ Mutation / Crossover / Rare Injection
→ Synthesis
```

The platform fixes evolution mechanics: hashing, lineage, archives, local tools, patch sandboxing, events, checkpoints, and progress. The model supplies task-specific semantics through structured objective contracts, policies, rankings, diagnoses, and mutation plans.

## Terminology boundary

`evolution` in CognitiveEvolve means model-driven iterative search over
structured candidate genomes. It is not a claim that every mutation/crossover
operation is a biological genetic operator or that fitness is independent of
model judgment. The fixed runtime guarantees are lineage, candidate structure,
selection pressure, archive retention, verification traces, and resumable
checkpoints. The model-authored parts provide semantic variation, synthesis,
ranking judgments, and task-specific fitness axes. Documentation and APIs should
describe this honestly as model-driven candidate evolution rather than a
separate genetic-algorithm runtime.

## Ranking

Ranking is relative and multihead. The runtime preserves objective alignment, answer likelihood, core mechanism strength, novelty, rarity, verifiability, coherence, tool progress, robustness, simplicity, transfer potential, auxiliary value, and deferral risk instead of collapsing everything into one total score.

## Archives

The archive system preserves answer elites, mechanism elites, rare/edge candidates, dormant candidates, auxiliary candidates, project patches, and useful failures. Candidates that are not current winners can still retain inheritable genes.

## Project input

Project runs snapshot files, build a world model, select context packets, generate project patch candidates, apply each patch in a sandbox, run local verification, and feed structured results back into the genome.

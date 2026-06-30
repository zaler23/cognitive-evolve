# Nexus Exploration and Recovery Hardening

## Why this exists

The old adaptive runtime had one important advantage: it was noisy in a useful way. It generated multiple route families, critiqued candidates, mutated promising and weird branches, and left intermediate checkpoints. Nexus is the single runtime now, so those advantages are implemented as Nexus-native mechanics rather than as a restored legacy path.

## Absorbed old-runtime strengths

Nexus now keeps these old strengths:

- **Wide initial exploration**: a narrow model seed pool is amplified with task-neutral search moves such as direct, known-pattern, negative/counterexample, analogy, inversion/dual, decomposition, tool-grounded, adversarial-critic, rare-recall, and wildcard routes.
- **Generate → critique → mutate**: every round includes a structured critique stage. Critiques are written into candidate verification traces and failure lessons, then used to choose mutation actions.
- **Branching instead of winner-only mutation**: profile budgets carry `initial_candidate_count` and `mutation_branches_per_round`, so exhaustive/deep profiles expand both width and depth.
- **Rare and edge preservation**: edge candidates are marked as seeds, routed into rarity/dormant archives, and can steer RareInject without being treated as facts.
- **Round/phase checkpoints**: Nexus writes population, archives, checkpoint, round snapshots, event log, and candidate journal during the loop, not only after final synthesis.
- **Interruption recovery**: if a provider/quota/model error occurs mid-round, Nexus saves an `error_checkpoint` and returns an interrupted failure/synthesis result instead of dropping the candidate pool.

## Seed safety

Supplemental seeds are marked:

```json
{
  "search_seed_not_final": true,
  "exploration_source": "nexus_exploration_amplifier"
}
```

Final synthesis will not return an unconverted initial seed as the answer. A seed must be evolved, verified, or synthesized by the model before it can become answer material.

## Artifacts

A Nexus run directory now includes:

```text
nexus-runtime/
  population.json
  archives.json
  checkpoint.json
  candidate-journal.jsonl
  events.jsonl
  rounds/
    round-0001-post_ranking_critique.json
    round-0001-post_mutation.json
    round-0001-error_checkpoint.json
  final-answer.md
  run-result.json
```

`checkpoint.json` is updated during the loop and is safe to inspect even when a run is interrupted before final synthesis.

## API profiles

`resolve_nexus_round_budget(...)` now resolves adaptive controls for each model tier:

- `max_rounds` as an explicit cap or safety checkpoint
- `adaptive` / `round_safety_limit` / `completion_requires_stop_signal`
- dynamic `initial_candidate_count` floors when explicitly configured
- `mutation_branches_per_round`

Operators may override them using:

```text
COGEV_NEXUS_PROFILE_EXHAUSTIVE_SAFETY_ROUNDS
COGEV_NEXUS_PROFILE_EXHAUSTIVE_MIN_CANDIDATES
COGEV_NEXUS_BRANCH_FACTOR
COGEV_NEXUS_PROFILE_EXHAUSTIVE_BRANCH_FACTOR
# Legacy COGEV_NEXUS_PROFILE_EXHAUSTIVE_ROUNDS / *_CANDIDATES are ignored by default;
# set COGEV_ACCEPT_LEGACY_PROFILE_ROUNDS=1 or COGEV_ACCEPT_LEGACY_PROFILE_CANDIDATES=1 only for deliberate compatibility.
# Legacy COGEV_MUTATION_BRANCH_FACTOR and COGEV_ACTIVE_POOL_LIMIT are ignored by default and surfaced as config warnings.
```

This keeps API model tiers meaningful without reintroducing the old adaptive engine. Candidate seeding uses repeated model batches plus semantic dedupe until the dynamic target is met or novelty stalls; safety checkpoints can return completed answer-first candidate output; continuation remains reserved for explicit interruption/quota/operator continuation.

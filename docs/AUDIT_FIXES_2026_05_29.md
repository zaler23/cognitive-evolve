# 2026-05-29 audit fixes

This release hardens the Nexus runtime against the negative audit findings from
the 34-point architecture review.

## Runtime invariants now enforced

- Archive membership is a current-state index, not proof of validity.
- A candidate is removed from all stale archive memberships before it is routed
  under a new fate.
- `Failed` and `Culled` candidates are kept as failure lessons only; they do
  not enter answer, rarity, novelty, mechanism, quality-diversity, or patch
  archives.
- Final synthesis must bind to a current eligible candidate.  Model synthesis
  cannot select a missing candidate id, and ineligible candidate ids are
  replaced only when the runtime already has an eligible candidate.
- Candidate score maps reject `NaN`, `Infinity`, and other non-finite values;
  durable JSON writers reject non-standard JSON.
- Project verification failure marks candidates as `Failed` consistently for
  initial candidates and offspring.

## Security and service boundaries

- Patch sandboxes reject path traversal, symlink targets, and symlink ancestors;
  sandbox copy also excludes symlinks and local build/cache directories.
- Local verifier commands run through an allowlist instead of arbitrary command
  execution.
- The API server refuses to bind to a non-loopback host with authentication
  disabled unless `COGEV_ALLOW_INSECURE_BIND=1` is explicitly set.
- CORS defaults to localhost origins.  Wildcard origins disable credentials.
- Background jobs record heartbeats and stale running jobs rehydrated after a
  process restart are marked `interrupted`.

## Architecture and maintenance

- The model protocol is split into capability-specific protocols while the
  complete `NexusModelProtocol` remains available for full adapters.
- `evolve_once()` is reduced to a delegating entrypoint; `EvolutionLoopController`
  owns lifecycle/checkpointing and `EvolutionRound` owns round stages.
- Project checkpoint resume preserves the project world envelope rather than
  collapsing it to a different runtime shape.
- Final result files and run-result JSON use atomic durable writes.
- Event persistence can append many logical events with one replay scan.
- Real-provider integration coverage is available as an explicit opt-in test via
  `COGEV_RUN_LLM_TESTS=1`; hermetic tests remain provider-free by default.

## API compatibility boundary

The `/v1/chat/completions` surface is OpenAI-shaped, not token-semantic
identical.  A request can trigger adaptive multi-round evolution. Streaming emits progress/heartbeat/final-answer chunks, and long-running production use should prefer `/v1/cogev/jobs`. A safety checkpoint produces `completion_status=needs_continuation`, not `completed`.

## Local configuration compatibility boundary

- Built-in API model round caps are `0`, meaning adaptive Nexus control rather
  than a fixed round count.
- Stale local env names that used to shape profile width/depth are ignored by
  default and surfaced through `round_budget.config_warnings`:
  `COGEV_NEXUS_PROFILE_*_ROUNDS`, `COGEV_NEXUS_PROFILE_*_CANDIDATES`,
  `COGEV_MUTATION_BRANCH_FACTOR`, and `COGEV_ACTIVE_POOL_LIMIT`.
- The replacement knobs are explicit: `COGEV_NEXUS_PROFILE_*_SAFETY_ROUNDS`,
  `COGEV_NEXUS_PROFILE_*_MIN_CANDIDATES`, `COGEV_NEXUS_BRANCH_FACTOR`, and
  `COGEV_NEXUS_PROFILE_*_BRANCH_FACTOR`.
- Legacy `COGEV_STOP_POLICY=budget_guarded` is accepted as an alias for
  `adaptive_until_solved`; unknown stop policies fall back to the selected
  profile default instead of the balanced default.

# Nexus proof-progress hardening

This update turns proof-like objectives into a stricter runtime path.  A
candidate may still explore concepts, but it cannot rank as the best final
answer or enter synthesis unless it carries verifier-visible object-level
progress.

Runtime invariants:

- proof-like tasks require concrete `formal_artifacts` such as `equation_set`,
  `construction`, `inequality_proof`, `case_analysis`, `witness`,
  `counterexample`, `lemma_ref`, `derivation`, or `proof_step`;
- candidates must record an `obligation_delta` against named proof obligations;
- duplicate formal signatures are rejected deterministically;
- repeated proof-object absence or ledger non-progress becomes a hard diagnosis
  that forces mutation directives toward formal artifacts or route refutation;
- quota/rate-limit errors pause the run at a checkpoint instead of falling back
  and continuing to call the provider;
- `current_round` is mirrored into checkpoint/runtime metadata so frontends do
  not confuse static budget settings with live progress.

The intended failure mode is now explicit: if a run only deepens terminology
without producing formal objects or closing obligations, synthesis returns
`route_incomplete` rather than a plausible-looking final answer.

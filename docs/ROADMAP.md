# Roadmap

CognitiveEvolve 2.0 is Nexus-only. Future work should deepen the current architecture rather than add alternate runtime paths.

## Near term

- Add more local tool adapters under `tools/`.
- Expand project context selection for large repositories.
- Improve checkpoint replay for long runs.
- Add richer UI/event consumers for `events/` progress logs.

## Guardrails

- Keep `NexusRuntime` as the only runtime.
- Keep `ranking/` as the only ranking system.
- Keep `archives/` as the only archive system.
- Keep candidates structured as `CandidateGenome` or `ProjectCandidateGenome`.

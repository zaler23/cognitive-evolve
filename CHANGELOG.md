# Changelog

## 2.0.0

Nexus runtime release.

- Promoted `NexusRuntime` to the single runtime architecture.
- Standardized candidates on `CandidateGenome` and `ProjectCandidateGenome`.
- Standardized ranking on relative comparison and multihead scores.
- Standardized archives on the Nexus multi-archive manager.
- Added text and project input paths with project snapshots, context selection, patch sandboxing, and local tool feedback.
- Added checkpoint replay, event/progress stores, and hermetic test defaults.
- Removed duplicate runtime/ranking/archive namespace baggage and npm control-plane metadata.

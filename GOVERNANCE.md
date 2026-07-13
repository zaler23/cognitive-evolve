# Governance

## Current model

CognitiveEvolve currently uses maintainer-led governance. The project is still a pre-release protocol workbench, so maintainers prioritize safety, source/runtime separation, and reviewable evolution over broad feature intake.

## Decision principles

- Reuse existing patterns before building new systems.
- Keep source and runtime physically separate.
- Prefer file-first native capabilities before external backends.
- Require explicit complexity-budget justification for new dependencies, services, databases, UI, MCP servers, background jobs, or fallback paths.
- Preserve validation evidence for architecture, eval, prompt, memory, adapter, and workflow changes.

## Maintainer responsibilities

Maintainers are responsible for:

- Reviewing pull requests and issues.
- Protecting security reporting paths.
- Keeping public docs accurate about project maturity.
- Rejecting changes that bypass the source/runtime boundary or create hidden dependency requirements.
- Deciding when the project reaches a public release boundary.

## Changing governance

Governance changes should be proposed through a design review issue and accepted through a pull request that updates this file and any related project policy documents.

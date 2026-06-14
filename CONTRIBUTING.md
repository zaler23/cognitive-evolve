# Contributing to CognitiveEvolve

Thank you for helping improve CognitiveEvolve. This repository is a pre-release, host-agnostic agent-system workbench. Contributions are welcome when they preserve the source/runtime boundary and keep changes reviewable.

## Project status

CognitiveEvolve is a source-installable beta / engineering preview. Users can
install it from source for local CLI and API use, but the project does not yet
promise a packaged public installer, PyPI release, hosted service, or one-click
production deployment. Do not add release packaging, hosted-service setup, or
deployment promises unless a maintainer has accepted a release-boundary
decision.

## Before opening an issue or pull request

1. Read `README.md`, `AGENTS.md`, `LOCAL_RUNTIME_BOUNDARY.md`, and the relevant files under `.cogev/specs/`.
2. Check existing issues and pull requests to avoid duplicates.
3. For architecture, eval, prompt, memory, adapter, or workflow changes, describe the problem contract and the smallest reversible change.
4. Do not include secrets, local credentials, private task artifacts, or runtime cache output.

## Development setup

The source project has a small Python runtime dependency set plus explicit test extras.

```bash
python3 -m pip install -e ".[test]"
PYTHONPYCACHEPREFIX=/tmp/cogev-pycache PYTHONDONTWRITEBYTECODE=1 python3 -B -m compileall -q cognitive_evolve_runtime scripts tests
PYTHONDONTWRITEBYTECODE=1 python3 -B -m pytest -q -p no:cacheprovider
```

For local development on a machine with a configured standalone runtime, also run:

```bash
python3 scripts/cogev.py doctor --scope core
python3 scripts/cogev.py doctor --scope runtime
```

GitHub Actions uses a temporary local runtime root and runs the source-safe subset automatically.

## Pull request expectations

A pull request should include:

- A clear problem statement and non-goals.
- A small, reviewable diff.
- Tests or a clear reason tests are not applicable.
- Documentation/spec updates when behavior changes.
- A complexity-budget note for new dependencies, services, databases, UI, MCP servers, background jobs, or fallback paths.
- A source/runtime boundary note if files under `.cogev/`, `scripts/`, `cognitive_evolve_runtime/`, or `cognitive_evolve_runtime/tools/` change.

## Dependency policy

Required dependencies in `pyproject.toml` should remain limited to the portable alpha runtime surface. Optional local backends belong behind explicit adapter ports and must not require ad hoc API-key wiring.

## Code style

- Prefer small functions and explicit artifact outputs.
- Keep file writes under the active task/runtime root unless the task explicitly edits source files.
- Do not add speculative abstractions or future-proof fallback chains.
- Preserve existing user changes; avoid unrelated cleanup in feature PRs.

## Commit guidance

Split large work into reviewable commits, for example:

1. Specs / policy changes.
2. Runtime implementation.
3. Tests / evals.
4. Documentation / sync audit.

## Security

Report vulnerabilities using `SECURITY.md`. Do not open public issues with exploit details or secrets.

# Pull Request

## Summary

<!-- What changed and why? -->

## Problem contract

- Goal:
- Non-goals:
- Key assumptions:

## Change type

- [ ] Bug fix
- [ ] Documentation
- [ ] Spec / protocol
- [ ] Runtime implementation
- [ ] Adapter boundary
- [ ] Eval / test
- [ ] Security hardening
- [ ] Other:

## Complexity budget

- [ ] No new dependency, service, database, UI, MCP server, background job, or fallback path.
- [ ] New complexity is introduced and justified below.

Justification, if applicable:

## Source/runtime boundary

- [ ] This change does not affect the local runtime boundary.
- [ ] This change affects runtime/source sync and updates documentation or tests.

## Validation

- [ ] `PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/cogev-pycache" PYTHONDONTWRITEBYTECODE=1 python3 -B -m compileall -q cognitive_evolve_runtime scripts tests`
- [ ] `PYTHONDONTWRITEBYTECODE=1 python3 -B -m pytest -q -p no:cacheprovider`
- [ ] `python3 scripts/cogev.py doctor --scope core`
- [ ] Other:

## Security and privacy

- [ ] No secrets, credentials, private runtime artifacts, or sensitive traces are included.
- [ ] Security-sensitive behavior changed and is explained below.

## CheckModel / independent review

- [ ] Not needed for this change.
- [ ] Needed; reviewer notes are included.
- [ ] Skipped with reason:

## Documentation

- [ ] README/docs/specs updated, or not needed.

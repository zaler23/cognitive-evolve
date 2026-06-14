# Release Provenance

This package is produced from the pure Nexus source boundary.

## Validation commands

```bash
python -m compileall -q cognitive_evolve_runtime
python -m pytest -q
```

## Source-of-truth modules

- `cognitive_evolve_runtime/nexus/runtime.py`
- `cognitive_evolve_runtime/nexus/loop/`
- `cognitive_evolve_runtime/nexus/state.py`
- `cognitive_evolve_runtime/candidates/`
- `cognitive_evolve_runtime/archives/`
- `cognitive_evolve_runtime/ranking/`
- `cognitive_evolve_runtime/inputs/`
- `cognitive_evolve_runtime/tools/`
- `cognitive_evolve_runtime/persistence/`
- `cognitive_evolve_runtime/events/`

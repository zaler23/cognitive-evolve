# CognitiveEvolve Source Specifications

These package resources are reference/design specifications for the Nexus-only source project.

Do not assume every YAML/JSON file in this directory is an executable runtime contract. Runtime authority remains in `cognitive_evolve_runtime.nexus.runtime.NexusRuntime`, with validation performed by Python modules and tests.

## Files

- `native-capabilities.json` — reference capability IDs and their Nexus ownership boundaries.
- `extension-ports.json` — allowed adapter/extension boundaries; not alternate runtimes.
- `native-runtime.json` — Nexus runtime architecture summary.
- `native-eval-suite.json` — non-executable reference validation shape; it has drifted from the hard-coded checks in `cognitive_evolve_runtime/nexus/evaluation.py`, which is the runtime authority.

## Boundary

Specs may document intended contracts, but implementation truth is established by source code, tests, doctor checks, and generated Nexus artifacts.

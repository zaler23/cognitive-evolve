# CognitiveEvolve Agent-System Paradigm

## Thesis

CognitiveEvolve is a new agent-system paradigm for tasks where quality depends on judgment, cognition, evaluation, and evolution rather than a single prompt-response turn.

The core system is host-neutral. MCP, LangGraph, DSPy/GEPA, Promptfoo, SkyDiscover, and other tools are adapters or configured local backends. They do not define the project identity, but tasks may use them directly through the runtime when needed.

## Core Loop

1. Understand the user's input and start always-on cognitive intake.
2. Generate and resolve the 1-3 highest-value internal questions without asking the user mid-turn; continue internal question batches only when they change the decision.
3. Route the task by complexity and risk for downstream execution strength, not for deciding whether intake happens.
4. Write a problem contract when the task is non-trivial.
5. Build a cognitive scaffold that separates facts, assumptions, inferences, preferences, recommendations, validation-needed items, and uncertainty.
6. Generate bounded candidates when the task is subjective, architectural, or evolutionary.
7. Score candidates with explicit objectives.
8. Compare candidates with quality-diversity pressure, multihead scoring, and pairwise judgment.
9. Preserve dissent through independent review where available.
10. Record the decision, validation plan, trace, feedback, and memory outcome.

## Why It Differs From A Normal Agent Harness

Most agent harnesses emphasize tool access, orchestration, and execution. CognitiveEvolve emphasizes cognition quality:

- Better problem definition before action.
- Explicit tradeoff structure instead of hidden preference guesses.
- Multi-candidate search for tasks where one answer is fragile.
- Evidence and validation records as first-class artifacts.
- User cognitive gain as a hard quality objective.
- Optional host adapters rather than a fixed runtime.

## Native Capabilities

The current native capabilities are:

- host execution and verification
- project governance
- workflow packets
- task scoping
- cognitive search
- evolution loop
- independent review
- user cognition
- tool boundary
- evaluation runner
- prompt optimizer
- durable execution
- observability

Each capability can work through local files first. External tools are bound later through adapter ports when they solve a validated failure.

## Adapter Principle

An adapter must declare:

- input contract
- output contract
- activation gate
- LLM-required execution path
- rollback path
- decision-record requirements
- verification command

This keeps the paradigm portable across hosts while still allowing practical local execution.

# CognitiveEvolve Quickstart

## One command

```bash
cogev quickstart "Improve this project while keeping patches small"
```

If `.env` is missing, the command asks for provider, model, API key,
concurrency, and rounds, writes the minimal `.env`, then runs the existing
standalone runtime.

## Fixture demo

```bash
cogev config init --profile fixture --output .env --force
cogev run --offline "demo fixture evolution"
```

## Check provider status

```bash
cogev llm status
cogev llm smoke
```

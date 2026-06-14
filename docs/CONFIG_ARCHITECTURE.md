# Configuration Architecture

CognitiveEvolve configuration is loaded through the runtime configuration layer and environment variables. Tests run in hermetic mode by default.

## Core settings

| Setting | Purpose |
|---|---|
| `COGEV_LLM_PROVIDER` | `fixture` for deterministic tests, `litellm` for provider-backed runs, or `direct_http` for OpenAI-compatible HTTP |
| `COGEV_LLM_FIXTURE` | Path to fixture responses for offline tests |
| `COGEV_LLM_MODEL` | Provider model ID for LiteLLM runs |
| `COGEV_LLM_API_BASE` | Optional provider base URL; required for `direct_http` |
| `COGEV_LLM_API_KEY` | Upstream provider key |
| `COGEV_SERVER_API_KEY` | Service key for the OpenAI-compatible API |
| `COGEV_RUNTIME_ROOT` | Standalone runtime root |
| `COGEV_TASKS_ROOT` | Standalone task directory root |
| `COGEV_HERMETIC_TEST` | Blocks user-home `.env` and real-provider fallback during tests |
| `COGEV_API_MAX_REQUEST_BYTES` | Maximum accepted `/v1/*` request body size |
| `COGEV_API_RATE_LIMIT_PER_MINUTE` | Per service-key/client local API request limit; `0` disables |
| `COGEV_API_JOB_TTL_SECONDS` | In-memory terminal job retention window |
| `COGEV_API_MAX_TRACKED_JOBS` | Maximum in-memory job snapshots before oldest terminal jobs are pruned |
| `COGEV_STREAM_MAX_SECONDS` | Optional SSE connection lifetime before clients should poll durable jobs |

Configuration precedence is explicit process environment first, then an
explicit `COGEV_CONFIG_FILE`, then runtime/cwd/repo `.cogev/config.yaml`
defaults for unset keys, then code defaults. Use
`configuration.config_resolution_diagnostics()` for a redacted source trace.

## Environment templates

Source control includes only safe templates:

- `.env.example` for loopback development placeholders.
- `.env.production.example` for shared deployment placeholders.
- `.env.fixture.example` for deterministic fixture tests.

Create a real local config with `python3 scripts/cogev.py config init --profile local --output .env`. The generated `.env` is ignored by Git and must contain deployment-specific secrets only outside the public source boundary. Public templates must remain generic: provider IDs, OpenAI-compatible base URLs, and fixture paths only.

## Runtime artifacts

Task runs write to:

```text
<task-dir>/runtime-state.json
<task-dir>/nexus-runtime/
<task-dir>/evaluations/
```

## Resume

`NexusRuntime.resume_from_checkpoint(...)` resumes from a stored checkpoint containing mode, contract, world, policy, diagnosis, population, archives, progress event, and budget history.

## API configuration

The OpenAI-compatible endpoint uses the service key, configured model list, runtime root, and task root from `api.config`. Frontend keys and upstream model keys remain separate.

The API also enforces a local request-size guard, a sliding-window request
rate guard, bounded executors, optional SSE connection lifetime control, and terminal job registry pruning. Reverse
proxies may add stricter production limits, but the Python service no longer
depends on a reverse proxy for basic resource boundaries.

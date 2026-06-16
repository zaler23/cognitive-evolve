# Provider Matrix — CognitiveEvolve 2.0

CognitiveEvolve requires an explicit model provider for production model calls. The runtime never silently replaces a failed production provider with local heuristics or fixture responses.

| Provider mode | Intended use | Required configuration | Status |
|---|---|---|---|
| `litellm` | Production model calls | `COGEV_LLM_PROVIDER=litellm`, `COGEV_LLM_MODEL`, `COGEV_LLM_API_KEY`; optional `COGEV_LLM_API_BASE` | Supported |
| `direct_http` | Production through a compatible `/v1/chat/completions` endpoint without LiteLLM request shaping | `COGEV_LLM_PROVIDER=direct_http`, `COGEV_LLM_MODEL`, `COGEV_LLM_API_BASE`; optional `COGEV_LLM_API_KEY` | Supported |
| `fixture` | Deterministic tests only | `COGEV_LLM_PROVIDER=fixture`, `COGEV_LLM_FIXTURE=tests/fixtures/llm_fixture.json` | Test-only |
| Missing/failed provider | Error path | No valid provider configuration or exhausted retries | Explicit failure / partial state |

The public provider boundary is `LLMProviderInterface`; implementations must remain generic provider adapters, not private application relays. Concurrency and budget settings are controlled by `COGEV_LLM_MAX_CONCURRENT`, `COGEV_LLM_RPM`, `COGEV_LLM_TPM`, and stage-budget settings.

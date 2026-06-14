# Security Policy

## Supported versions

CognitiveEvolve is currently pre-release. Security fixes target the default branch until stable releases exist.

| Version | Supported |
|---|---|
| Default branch | Yes |
| Tagged releases | Not yet |

## Reporting a vulnerability

Please do not report security vulnerabilities in public issues.

Preferred reporting path:

1. Use GitHub private vulnerability reporting or a repository security advisory if it is enabled for this repository.
2. If private reporting is not enabled, open a public issue titled `Security contact request` with no exploit details, no proof-of-concept, and no secrets. A maintainer should then provide a private reporting path.

Include in a private report:

- Affected component or file path.
- Impact and threat model.
- Reproduction steps or proof-of-concept, when safe to share privately.
- Whether credentials, private task artifacts, or user data could be exposed.
- Suggested mitigation, if known.

## Handling expectations

Maintainers should acknowledge a valid private report, assess severity, prepare a fix or mitigation, and publish a security advisory when appropriate.

## Secrets and local runtime data

Do not include API keys, tokens, `.env` files, private host configuration, local runtime caches, task artifacts containing sensitive user content, or prompt/eval traces with private data in issues or pull requests.

## Runtime hardening defaults

- The OpenAI-compatible API is local-first. Serving on a non-loopback host with
  frontend authentication disabled is refused unless
  `COGEV_ALLOW_INSECURE_BIND=1` is explicitly set for a trusted local network.
- CORS defaults to localhost origins. If `COGEV_CORS_ALLOW_ORIGINS=*` is used,
  credentials are disabled automatically.
- Project patch verification runs inside a copied sandbox. Patch paths reject
  absolute paths, `..`, symlink targets, and symlink ancestors.
- Local verification commands are allowlisted. Add new commands through
  `ToolCommandSpec` and review their read/write/network behavior before use.

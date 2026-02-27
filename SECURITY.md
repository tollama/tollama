# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in tollama, please report it through
[GitHub Security Advisories](https://github.com/yongchoelchoi/tollama/security/advisories/new)
rather than opening a public issue.

### What to include

- Description of the vulnerability
- Steps to reproduce (if applicable)
- Affected component (daemon API, CLI, auth, runner process isolation, etc.)
- Potential impact assessment

### Response timeline

- **Acknowledgment:** within 72 hours of report submission
- **Initial assessment:** within 1 week
- **Fix or mitigation:** timeline communicated after assessment

### Scope

The following areas are in scope for security reports:

- **Daemon HTTP API** (`tollamad`): authentication bypass, authorization flaws,
  injection vulnerabilities, path traversal
- **Runner process isolation:** sandbox escapes, unauthorized cross-runner access
- **Authentication:** API key handling, credential storage, token validation
- **Dependency vulnerabilities:** issues in direct dependencies that affect
  tollama's security posture
- **Information disclosure:** credential leakage in logs, diagnostics, or error
  responses

### Out of scope

- Model weights and their upstream licenses (report to upstream model providers)
- Denial-of-service via large forecast requests (use rate limiting and resource
  controls)
- Vulnerabilities in optional runner dependencies when tollama is not involved

## Operator Hardening

Recommended settings for production and multi-user deployments:

- **Enable API key authentication.** Add one or more keys to `auth.api_keys` in
  `~/.tollama/config.json`. Without keys configured, the API is open to any process
  that can reach the daemon port.
- **Bind to loopback only.** Start the daemon with `tollama serve --host 127.0.0.1`
  (the default) rather than `0.0.0.0`. Only expose on a wider interface if you have
  network-level controls in place.
- **Do not set `TOLLAMA_DOCS_PUBLIC=1` in production.** This flag removes authentication
  from the interactive API docs (`/docs`, `/redoc`, `/openapi.json`), exposing your API
  schema to unauthenticated callers.
- **Restrict CORS origins.** The daemon's CORS configuration defaults to permissive
  settings for local development. Review and restrict `allowed_origins` for any
  deployment where the daemon is reachable from a browser context.
- **Review rate limit defaults.** The built-in rate limiter protects against request
  floods. Confirm the default limits are appropriate for your environment; lower them
  for shared or public-facing instances.
- **Runner process isolation.** Each runner family runs as a separate subprocess.
  Avoid running the daemon as root. Use OS-level controls (cgroups, `ulimit`) to
  cap runner memory and CPU if needed.

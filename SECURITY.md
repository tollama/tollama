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

# OpenClaw Gateway Runbook

This runbook describes end-to-end operation of `skills/tollama-forecast/` when
OpenClaw executes commands through a gateway runtime.

## 1) Topology assumptions

- OpenClaw execution happens on a gateway host, not on the user workstation.
- Tollama daemon may run:
  - on the same gateway host, or
  - on a separate host reachable from the gateway network.
- `127.0.0.1` is valid only when daemon is on the gateway host itself.

## 2) Base URL policy

- Use explicit, gateway-reachable daemon URLs.
- Preferred order:
  - CLI `--base-url`
  - `TOLLAMA_BASE_URL`
  - default `http://127.0.0.1:11435`
- Avoid loopback defaults when daemon is remote.

Recommended:

```bash
export TOLLAMA_BASE_URL="http://<daemon-host>:11435"
```

## 3) Network, proxy, and TLS checks

1. DNS/route check:

```bash
curl -s "$TOLLAMA_BASE_URL/api/version"
```

2. Health check:

```bash
bash skills/tollama-forecast/bin/tollama-health.sh --base-url "$TOLLAMA_BASE_URL"
```

3. Proxy policy checks (if gateway enforces proxy):

- Ensure daemon host is correctly covered by proxy bypass rules where required.
- Verify `HTTP_PROXY`/`HTTPS_PROXY`/`NO_PROXY` do not block daemon access.

4. TLS (for HTTPS daemon endpoints):

- Use gateway trust store/cert chain accepted by `curl`.
- Validate certificate hostname matches daemon URL host.

## 4) Smoke tests

```bash
bash scripts/validate_openclaw_skill_tollama_forecast.sh
bash skills/tollama-forecast/bin/tollama-models.sh installed --base-url "$TOLLAMA_BASE_URL"
bash skills/tollama-forecast/bin/tollama-pull.sh --model mock --base-url "$TOLLAMA_BASE_URL"
bash skills/tollama-forecast/bin/tollama-forecast.sh \
  --model mock \
  --input skills/tollama-forecast/examples/simple_forecast.json \
  --base-url "$TOLLAMA_BASE_URL" \
  --timeout 300
```

Metrics smoke:

```bash
bash skills/tollama-forecast/bin/tollama-forecast.sh \
  --model mock \
  --input skills/tollama-forecast/examples/metrics_forecast.json \
  --base-url "$TOLLAMA_BASE_URL" \
  --metrics mape,mase,mae,rmse,smape \
  --mase-seasonality 1 \
  --timeout 300
```

## 5) Failure triage

- Exit code `3`:
  - first suspect gateway-to-daemon routing, firewall, or URL mismatch.
- Exit code `6`:
  - raise timeout for first-run model load and runner bootstrap.
- Exit code `5`:
  - confirm gated model license acceptance policy.

For machine-readable diagnostics:

```bash
export TOLLAMA_JSON_STDERR=1
```

## 6) Rollback checklist

If new gateway rollout fails, revert to last known-good state:

1. Restore previous `TOLLAMA_BASE_URL` and proxy environment configuration.
2. Revert skill link to previous validated revision (or remove updated link).
3. Re-run:
   - `tollama-health.sh`
   - `tollama-models.sh installed`
   - `tollama-forecast.sh` with `examples/simple_forecast.json`
4. Confirm exit code contract remains `0/2/3/4/5/6/10`.

## 7) Operational guardrails

- Keep skill validation (`scripts/validate_openclaw_skill_tollama_forecast.sh`)
  in CI before gateway rollout.
- Prefer daemon URL config as code for reproducible gateway environments.
- Re-run smoke tests after daemon upgrades and network policy changes.

## 8) Performance expectations

Approximate first-request and steady-state latency per model family (single series, horizon 10):

| Family | First request | Steady state | Notes |
|---|---|---|---|
| `mock` | < 1 s | < 1 s | No model loading |
| `torch` (Chronos-2) | 10–60 s | 1–5 s | Weight load + torch JIT on first use |
| `timesfm` | 2–5 min | 5–30 s | XLA/JAX compilation on first request |
| `uni2ts` (Moirai) | 30–90 s | 3–10 s | ONNX or torch weight load |
| `sundial` | 30–120 s | 3–15 s | Transformer weight load |
| `toto` | 30–90 s | 3–10 s | Transformer weight load |

These are estimates on CPU-only hardware with no pre-warmed cache. GPU acceleration
reduces steady-state by 5–10×. Set `--timeout` accordingly for first-run requests.

## 9) Monitoring signals

Endpoints to poll from your monitoring system:

| Endpoint | Method | Purpose | Alert condition |
|---|---|---|---|
| `/v1/health` | GET | Daemon liveness | Non-200 → daemon down |
| `/api/usage` | GET | Per-key request counts and error rates | Error rate spike |
| `/metrics` | GET | Prometheus metrics (optional dep) | Runner restart counter, P95 latency |

Exit codes to alert on when received from skill scripts:

| Exit code | Alert action |
|---|---|
| `3` (DAEMON_UNREACHABLE) | Page on-call — daemon or network failure |
| `10` (INTERNAL_ERROR) | Investigate runner process crash |
| `6` (TIMEOUT) | Review timeout config and model load time |

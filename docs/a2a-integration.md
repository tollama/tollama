# A2A Integration (Latest Spec)

- Discovery: `GET /.well-known/agent-card.json`
- JSON-RPC endpoint: `POST /a2a`
- Implemented methods:
  - `message/send`
  - `message/stream` (SSE)
  - `tasks/get`
  - `tasks/query`
  - `tasks/cancel`
- Current capability flags in Agent Card:
  - `streaming=true`
  - `pushNotifications=false`
- When API keys are configured, discovery and `/a2a` calls require
  `Authorization: Bearer <key>` (authenticated discovery default).

Minimal outbound A2A client is available:

```python
from tollama.a2a import A2AClient

client = A2AClient()
card = client.discover(base_url="http://127.0.0.1:11435")
print(card["name"])
```

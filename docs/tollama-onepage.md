# Tollama — Ollama for Time Series Foundation Models

> 시계열 파운데이션 모델(TSFM)을 하나의 플랫폼에서 실행, 비교, 서빙하는 통합 TSFM 플랫폼

---

## 왜 필요한가?

TSFM이 LLM처럼 범용 모델로 등장하면서, 도메인별 커스텀 모델 없이도 시계열 예측이 가능해졌다. 그런데 이 모델들을 **실제로 쓰기는 여전히 어렵다.**

| 문제 | 현실 |
|------|------|
| 설치 파편화 | Chronos는 PyTorch 특정 버전, TimesFM은 자체 패키지, Uni2TS는 Python 3.11 필수 |
| API 비통일 | 모델마다 입력 포맷, 예측 방식, covariate 처리가 제각각 |
| 의존성 충돌 | 두 모델을 동시에 설치하면 패키지 버전 충돌 |
| 통합 불편 | AI 에이전트나 서비스에서 TSFM을 도구로 쓰려면 모델별 래퍼를 직접 작성 |

## 목표

**파편화된 TSFM들을 하나의 통합 인터페이스로 묶어, 개발자와 AI 에이전트 모두가 쉽게 시계열 예측을 활용할 수 있는 TSFM 플랫폼을 만든다.**

---

## 개발자를 위한 가치 — API, SDK, Dashboard

예측 기반 서비스를 **쉽고 빠르게** 만들고, **유지보수 부담 없이** 운영할 수 있다.

```python
from tollama import Tollama

t = Tollama()

# 3줄로 예측 완료
result = t.forecast(model="chronos2", series=my_data, horizon=30)
df = result.to_df()

# 모델 자동 선택 + 앙상블
best = t.auto_forecast(series=my_data, horizon=30, strategy="ensemble")

# 체이닝 워크플로우: 분석 → 예측 → What-if 시나리오
flow = t.workflow(my_data).analyze().auto_forecast(horizon=30).what_if(scenarios)
```

| 제공 인터페이스 | 설명 |
|---------------|------|
| **HTTP API** | 42개 엔드포인트 — 예측, 분석, 비교, What-if, 리포트 |
| **Python SDK** | `Tollama` 클래스 16개 메서드, DataFrame 변환, 체이닝 workflow |
| **CLI** | `tollama pull` → `tollama run` — Ollama 스타일 워크플로우 |
| **Dashboard** | 웹(Chart.js) + TUI(Textual) — 모델 모니터링, 예측 시각화 |

## AI 에이전트를 위한 가치 — MCP, A2A, Skills

AI 에이전트가 **예측이 필요한 순간에 바로 TSFM을 도구로 호출**할 수 있다.

| 연동 경로 | 설명 |
|----------|------|
| **MCP Server** | 15개 도구 — forecast, analyze, compare, what-if, report 등 |
| **A2A 프로토콜** | JSON-RPC 기반 에이전트 간 통신, 태스크 큐 |
| **LangChain** | 13개 도구 네이티브 통합 |
| **CrewAI / AutoGen / Smolagents** | 각 프레임워크별 어댑터 제공 |
| **OpenClaw Skill** | OpenAI 도구 스키마 + 셸 스크립트 |

## TSFM Platform Dashboard

웹 대시보드와 터미널 TUI 대시보드를 통해 플랫폼 전체를 한눈에 관리한다.

- 모델 상태 모니터링 (설치/로드/실행 중)
- 예측 결과 시각화 및 모델 간 비교
- 실시간 이벤트 스트리밍 (SSE)
- 사용량 메트릭 (Prometheus 연동)

---

## 아키텍처

```
┌────────────────────────────────────────────────────────┐
│  개발자: CLI / SDK / HTTP API / Dashboard              │
├────────────────────────────────────────────────────────┤
│  AI 에이전트: MCP (15 tools) / A2A / LangChain / ...   │
├────────────────────────────────────────────────────────┤
│  TSFM Platform Daemon (tollamad)                       │
│  예측 · 분석 · 비교 · What-if · 파이프라인 · 리포트      │
│  인증 · Rate Limiting · Prometheus · SSE               │
├──────┬──────┬──────┬──────┬──────┬──────┬──────────────┤
│      │ stdio JSON-lines protocol      │              │
│      ▼      ▼      ▼      ▼      ▼      ▼              │
│   torch  timesfm  uni2ts  sundial  toto  mock          │
│   (Chronos, Granite)                                   │
│   각 패밀리별 독립 venv — 의존성 충돌 원천 차단           │
└────────────────────────────────────────────────────────┘
```

## 지원 모델

| 모델 | 제공처 | Covariates |
|------|--------|:----------:|
| Chronos-2 | Amazon | Past + Future |
| Granite TTM R2 | IBM | Past + Future |
| TimesFM 2.5-200M | Google | Past + Future |
| Moirai 2.0-R Small | Salesforce | Past + Future |
| Sundial Base 128M | THUML | Target only |
| Toto Open Base 1.0 | Datadog | Past only |

---

## To-Do

| 항목 | 현재 상태 | 목표 |
|------|:--------:|------|
| **모델 간 자동 비교/선택** | ✅ 기본 구현 (`/api/compare`, `/api/auto-forecast`) | Best model routing 고도화 — 데이터 특성별 자동 라우팅 |
| **데이터 전처리 자동화** | ❌ 미구현 | 결측치 보간, 리샘플링, 이상치 제거를 플랫폼에서 자동 처리 |
| **파인튜닝 / 앙상블** | ⚠️ 앙상블만 구현 (`ensemble.py`) | 도메인 적응 파인튜닝 워크플로우 추가 |
| **로컬 + 클라우드 실행** | ⚠️ Dockerfile 존재, 로컬 중심 | K8s 매니페스트, docker-compose, 클라우드 배포 가이드 |

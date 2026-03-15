# Tollama 기획-구현 매핑

> tollama/tollama 레포 기준으로, 처음 기획 포인트("TSFM용 Ollama", unified interface, 개발자/에이전트/대시보드 benefit)를 현재 구현물에 맞춰 정리한 문서입니다.

---

## 1. Tollama 한 줄 정의

**Tollama = "Time Series Foundation Model(TSFM)을 Ollama처럼 `pull` / `run` / `serve`로 로컬에서 실행·서빙하는 forecasting daemon + CLI + Python SDK + Agent/Dashboard 통합"**

기본 데몬 주소: `http://127.0.0.1:11435`

---

## 2. 문제 정의(기획) ↔ 현재 구현(코드) 매핑

### (A) 문제점: TSFM 모델 설치/사용법 파편화

구현에서는 이를 **"모델 레지스트리 + runner family 분리 + 단일 API/CLI"**로 흡수한다.

- `model-registry/registry.yaml`에 모델 메타(소스, 라이선스, covariate capability 등)를 선언
- 데몬이 이를 기반으로 `pull`/`forecast`/`show`를 통일된 방식으로 제공
- 패밀리별 독립 venv(`~/.tollama/runtimes/<family>/venv/`)로 의존성 충돌을 원천 차단

### (B) 목표: single interface로 TSFM을 쉽게 사용

| 인터페이스 | 설명 |
|-----------|------|
| **CLI** | `tollama serve`, `tollama pull`, `tollama run`, `tollama list`/`show`/`ps`/`rm` 등 27개 커맨드 |
| **HTTP API** | `/api/*`(Ollama 스타일) + `/v1/*`(stable) — 총 42개 엔드포인트 |
| **Python SDK** | `Tollama` 클래스 16개 메서드 + workflow 체이닝 + DataFrame 변환 |

### (C) 개발자 benefit: 예측 기반 서비스 빠른 구축 + 유지보수 편의

- API/SDK/CLI를 동일 스키마(`core/schemas.py`)로 엮어 요청/응답 일관성 확보
- 모델 설치/삭제/상태를 동일한 lifecycle(`pull`/`list`/`show`/`ps`/`rm`)로 관리
- API Key 인증(`daemon/auth.py`, Bearer 토큰 + HMAC 검증) + 사용량 집계(`/api/usage`, SQLite 기반)
- 이벤트 스트림(`/api/events`, SSE) + Prometheus 메트릭(`/metrics`) 지원

### (D) AI agent benefit: 예측이 필요할 때 "툴처럼" 호출

| 연동 경로 | 도구 수 | 설명 |
|----------|:------:|------|
| **MCP Server** (`tollama-mcp`) | 15개 | Claude Desktop/Code에서 네이티브 도구로 연결 |
| **LangChain** | 13개 | `get_tollama_tools(...)` 래퍼 |
| **CrewAI** | - | `get_crewai_tools(...)` |
| **AutoGen** | - | `get_autogen_tool_specs()` + `register_autogen_tools()` |
| **smolagents** | - | `get_smolagents_tools(...)` |
| **A2A** (JSON-RPC) | 8개 오퍼레이션 | `/.well-known/agent-card.json` 디스커버리 + `/a2a` 호출 + SSE 스트리밍 |
| **OpenClaw Skill** | 6개 스크립트 | `skills/tollama-forecast/` — 표준 exit code 계약(v2) |

### (E) Dashboard: monitoring, management and more

- **웹 대시보드**: `/dashboard` (Alpine.js + HTMX + Chart.js)
- **TUI 대시보드**: `tollama dashboard` (Textual 기반)
- 모델 설치/삭제/조회, forecast 실행 및 차트 렌더링, compare, usage/events 등 "운영/실험" 기능을 탭으로 제공

---

## 3. 현재 구현된 제품 구성요소(컴포넌트)

### 3.1 Daemon (`tollamad` / `tollama serve`)

FastAPI 기반 데몬이 HTTP API 표면 + runner lifecycle 관리를 담당한다.

**주요 진단/관측 엔드포인트:**

| 엔드포인트 | 용도 |
|-----------|------|
| `/api/info` | 진단 정보 1-shot |
| `/api/usage` | 키별 사용량 (SQLite 기반) |
| `/api/events` | SSE 실시간 이벤트 스트리밍 |
| `/metrics` | Prometheus 메트릭 export |
| `/api/version` | 버전 정보 |

### 3.2 Runner family (모델별 런타임 분리)

`mock`, `torch`, `timesfm`, `uni2ts`, `sundial`, `toto` 6개 런너가 "모델 패밀리" 단위로 분리되어 있다.

- **데몬 ↔ 런너 통신**: stdio JSON-lines 프로토콜(`core/protocol.py`)
  - NDJSON 형식, `id`/`method`/`params` 요청 + `id`/`result`/`error` 응답
  - 지원 메서드: `capabilities`, `load`, `unload`, `forecast`, `ping`, `hello`
- **프로세스 슈퍼비전**: `daemon/supervisor.py` — 런너 프로세스 수명주기 관리, 재시도(최대 2회), 기본 타임아웃 10초
- 무거운 의존성은 runner extra로만 설치(`pyproject.toml`의 optional extra)하여 daemon/core/cli는 가볍게 유지

### 3.3 Registry + Local manifest

- **레지스트리**: `model-registry/registry.yaml`에 7개 모델 선언
- **로컬 저장**: `~/.tollama/` (또는 `$TOLLAMA_HOME`) 하위에 manifest/스냅샷 메타 저장
  - `~/.tollama/models/<name>/manifest.json` — 모델별 매니페스트
  - `~/.tollama/runtimes/<family>/venv/` — 패밀리별 독립 가상환경
  - `~/.tollama/modelfiles/` — TSModelfile 정의
  - `~/.tollama/config.json` — 데몬 설정
- HuggingFace 소스 모델은 스냅샷 pull, 라이선스 수락이 필요한 모델은 `--accept-license` / `accept_license=true`로 처리

### 3.4 CLI / SDK / Dashboard / Agent 통합

CLI 치트시트에 핵심 플로우가 정리되어 있고, 웹/TUI 대시보드, MCP/A2A/LangChain 도구들이 같은 베이스 URL(`http://localhost:11435`)을 공유한다.

---

## 4. "지금 당장" 되는 기능들 (핵심 기능 인벤토리)

### 4.1 모델 lifecycle (Ollama 스타일)

| 기능 | CLI | HTTP API |
|------|-----|----------|
| 설치 | `tollama pull <model>` | `POST /api/pull` |
| 목록 조회 | `tollama list` | `GET /api/tags` |
| 상세 정보 | `tollama show <model>` | `POST /api/show` |
| 로드 상태 | `tollama ps` | `GET /api/ps` |
| 삭제 | `tollama rm <model>` | `DELETE /api/delete` |

### 4.2 Forecasting (단일 계약)

| 엔드포인트 | 설명 |
|-----------|------|
| `POST /api/forecast` | JSON 또는 NDJSON 스트림 (`stream=true`) |
| `POST /v1/forecast` | Stable JSON |
| `POST /api/auto-forecast` | 제로 설정 모델 선택 (전략: `auto`/`fastest`/`best_accuracy`/`ensemble`) |
| `POST /api/forecast/progressive` | SSE 단계별 스트리밍 예측 |
| `POST /api/forecast/upload` | Multipart 파일 업로드 + 예측 |

### 4.3 Covariates(공변량) 표준 계약 + 호환성 처리

- `past_covariates`, `future_covariates`, `static_covariates` 구조 표준화
- `parameters.covariates_mode`:
  - `best_effort` (기본): 미지원 covariates를 무시하고 경고 반환
  - `strict`: 미지원 covariates에 HTTP 400 에러
- 수치형 + 범주형 covariate 모두 지원
- 모델별 호환성은 데몬이 자동 판단 (`daemon/covariates.py`)

### 4.4 정확도 메트릭 계산 (8종)

요청에 `series[].actuals`와 `parameters.metrics.names`를 주면 아래 메트릭을 계산하여 응답에 포함한다.

| 메트릭 | 설명 |
|--------|------|
| **MAPE** | Mean Absolute Percentage Error |
| **MASE** | Mean Absolute Scaled Error |
| **MAE** | Mean Absolute Error |
| **RMSE** | Root Mean Squared Error |
| **SMAPE** | Symmetric Mean Absolute Percentage Error |
| **WAPE** | Weighted Absolute Percentage Error |
| **RMSSE** | Root Mean Squared Scaled Error |
| **Pinball** | Quantile Loss (분위수 기반) |

### 4.5 Data ingest (CSV/Parquet)

- `data_url` 기반 ingest 또는 multipart upload로 CSV/Parquet를 series payload로 정규화
- 엔드포인트:
  - `POST /api/ingest/upload` — 파일 인제스트 (예측 없이)
  - `POST /api/forecast/upload` — 파일 업로드 + 바로 예측
- Parquet는 optional dependency(`.[ingest]`)로 분리

### 4.6 Structured Intelligence + Generative Planning

단순 forecast 외에 다음 고급 분석 엔드포인트들이 포함되어 있다.

| 엔드포인트 | 설명 |
|-----------|------|
| `POST /api/analyze` | 주파수, 계절성, 추세, 이상치, 정상성, 데이터 품질 진단 |
| `POST /api/generate` | 통계 프로필 기반 합성 시계열 생성 |
| `POST /api/compare` | 다중 모델 비교 (동일 데이터, 병렬 예측) |
| `POST /api/what-if` | 시나리오 분석 (multiply/add/replace 변환) |
| `POST /api/counterfactual` | 개입 기반 반사실적 궤적 비교 |
| `POST /api/scenario-tree` | 재귀적 분위수 기반 확률적 시나리오 트리 |
| `POST /api/report` | 분석 + 추천 + 예측 + 내러티브 종합 보고서 |
| `POST /api/pipeline` | analyze → recommend → pull → auto-forecast 자동화 |
| `POST /api/recommend` | 데이터 특성 기반 모델 추천 |

### 4.7 Dashboard (Web + TUI)

| 항목 | 웹 | TUI |
|------|-----|------|
| 접근 방법 | `/dashboard` 또는 `tollama open` | `tollama dashboard` (optional `.[tui]`) |
| 기술 스택 | Alpine.js + HTMX + Chart.js | Textual 프레임워크 |
| 기능 | Overview(모델/이벤트/사용량), Forecast(입력/차트/Export), Models(pull/delete), Compare, Help |
| 인증 | API key 설정 시 401 → 로그인 다이얼로그 | 동일 |

---

## 5. 지원 TSFM 모델/런너 (레지스트리 기준)

현재 레지스트리에 아래 모델들이 "이름으로 바로" 관리된다.

| 모델 | 패밀리 | HuggingFace Repo | 라이선스 | `--accept-license` 필요 | pyproject.toml extra |
|------|:------:|-------------------|:--------:|:----------------------:|---------------------|
| `mock` | `mock` | (로컬 내장) | MIT | No | (기본 포함) |
| `chronos2` | `torch` | `amazon/chronos-2` | Apache-2.0 | No | `runner_torch` |
| `granite-ttm-r2` | `torch` | `ibm-granite/granite-timeseries-ttm-r2` | Apache-2.0 | No | `runner_torch` |
| `timesfm-2.5-200m` | `timesfm` | `google/timesfm-2.5-200m-pytorch` | Apache-2.0 | No | `runner_timesfm` |
| `moirai-2.0-R-small` | `uni2ts` | `Salesforce/moirai-2.0-R-small` | CC BY-NC 4.0 | **Yes** | `runner_uni2ts` |
| `sundial-base-128m` | `sundial` | `thuml/sundial-base-128m` | Apache-2.0 | No | `runner_sundial` |
| `toto-open-base-1.0` | `toto` | `Datadog/Toto-Open-Base-1.0` | Apache-2.0 | No | `runner_toto` |

> **참고**: TimesFM은 첫 실행 시 컴파일/준비로 시간이 걸릴 수 있어 기본 타임아웃을 늘렸고, 더 느린 머신에서는 추가 `timeout`이 필요할 수 있다. 또한 `uni2ts`와 `timesfm`은 Python < 3.12 제약이 있다.

---

## 6. 사용 플로우 (개발자 관점)

### 가장 빠른 시작 (Quickstart)

```bash
pip install tollama
tollama serve          # 데몬 실행 (기본 11435)
tollama quickstart     # pull + forecast 데모 + next steps
```

### 모델 설치/예측

```bash
tollama pull chronos2
tollama run chronos2 --input examples/chronos2_request.json --no-stream
```

### Python SDK (서비스 코드에 임베딩)

```python
from tollama import Tollama

t = Tollama()

# 기본 예측
result = t.forecast(model="chronos2", series=my_data, horizon=30)
df = result.to_df()

# 자동 모델 선택 + 앙상블
best = t.auto_forecast(series=my_data, horizon=30, strategy="ensemble")

# 체이닝 워크플로우
flow = t.workflow(my_data).analyze().auto_forecast(horizon=30).what_if(scenarios)

# 파일에서 직접 예측
result = t.forecast_from_file("data.csv", model="chronos2", horizon=7)

# 고급 분석
analysis = t.analyze(series=my_data)
report = t.report(series=my_data, horizon=30)
cf = t.counterfactual(model="chronos2", series=my_data, intervention_index=10)
tree = t.scenario_tree(model="chronos2", series=my_data, horizon=4, depth=2)
```

**SDK 전체 메서드 목록:**

| 카테고리 | 메서드 |
|---------|--------|
| 시스템 | `health()`, `models()`, `pull()`, `show()` |
| 예측 | `forecast()`, `auto_forecast()`, `forecast_from_file()`, `compare()` |
| 분석 | `analyze()`, `generate()`, `counterfactual()`, `scenario_tree()` |
| 시나리오 | `what_if()`, `report()`, `pipeline()` |
| 워크플로우 | `workflow()` → `TollamaSeriesWorkflow` 체이닝 |
| 결과 처리 | `to_df()`, `then_compare()`, `then_what_if()` |

---

## 7. 에이전트/툴 통합 ("예측을 도구화")

### MCP (Claude Desktop/Code)

```bash
pip install "tollama[mcp]"
tollama-mcp              # 또는 python -m tollama.mcp
```

자동 설치 스크립트: `scripts/install_mcp.sh`

**등록 도구 (15개):**

| 도구 | 설명 |
|------|------|
| `tollama_health` | 데몬 상태 확인 |
| `tollama_models` | 모델 목록 (installed/loaded/available) |
| `tollama_forecast` | 비스트리밍 예측 |
| `tollama_auto_forecast` | 제로 설정 자동 예측 |
| `tollama_analyze` | 시계열 진단 (주파수, 계절성, 추세, 이상치) |
| `tollama_generate` | 합성 시계열 생성 |
| `tollama_counterfactual` | 개입 반사실적 궤적 |
| `tollama_scenario_tree` | 확률적 시나리오 트리 |
| `tollama_report` | 종합 보고서 (analyze + recommend + forecast) |
| `tollama_what_if` | 시나리오 분석 |
| `tollama_pipeline` | 자동화 파이프라인 |
| `tollama_compare` | 다중 모델 비교 |
| `tollama_pull` | 모델 설치 |
| `tollama_show` | 모델 메타데이터 |
| `tollama_recommend` | 모델 추천 |

### LangChain / CrewAI / AutoGen / smolagents

| 프레임워크 | 진입점 | 도구 수 |
|-----------|--------|:------:|
| **LangChain** | `get_tollama_tools(base_url, timeout)` | 13개 Tool 클래스 |
| **CrewAI** | `get_crewai_tools(base_url, timeout)` | 동적 생성 |
| **AutoGen** | `get_autogen_tool_specs()` + `register_autogen_tools(caller, executor)` | 동적 생성 |
| **smolagents** | `get_smolagents_tools(base_url, timeout)` | 동적 생성 |

### A2A (에이전트 간 호출)

| 항목 | 상세 |
|------|------|
| 디스커버리 | `GET /.well-known/agent-card.json` |
| 레거시 alias | `GET /.well-known/agent.json` |
| 호출 | `POST /a2a` (JSON-RPC 2.0) |
| 프로토콜 버전 | A2A 1.0 |
| 지원 오퍼레이션 | forecast, auto_forecast, analyze, generate, compare, what_if, pipeline, recommend |
| 스트리밍 | SSE (EventStream) |
| 태스크 관리 | 상태 머신 (pending → working → completed/failed/canceled) |

### OpenClaw Skill

`skills/tollama-forecast/` 디렉터리에 스킬 구성요소가 정리되어 있다.

| 파일 | 용도 |
|------|------|
| `SKILL.md` | 스킬 문서 |
| `openai-tools.json` | OpenAI 도구 스키마 정의 |
| `bin/tollama-health.sh` | 헬스 체크 |
| `bin/tollama-models.sh` | 모델 목록/관리 |
| `bin/tollama-forecast.sh` | 예측 실행 |
| `bin/tollama-pull.sh` | 모델 설치 |
| `bin/tollama-rm.sh` | 모델 삭제 |
| `bin/tollama-info.sh` | 데몬 정보 |
| `bin/_tollama_lib.sh` | 공유 유틸리티 |
| `examples/*.json` | 예제 페이로드 (단순/다중 시리즈, covariates, metrics) |

**Exit code 계약 (v2):**

| 코드 | 카테고리 |
|:----:|---------|
| `0` | 성공 |
| `2` | `INVALID_REQUEST` |
| `3` | `DAEMON_UNREACHABLE` |
| `4` | `MODEL_MISSING` |
| `5` | `LICENSE_REQUIRED` / `PERMISSION_DENIED` |
| `6` | `TIMEOUT` |
| `10` | `INTERNAL_ERROR` |

---

## 8. 대시보드 (운영/실험) 범위

| 항목 | 웹 GUI | TUI |
|------|--------|-----|
| 접근 | `/dashboard` 또는 `tollama open` | `tollama dashboard` |
| 탭 구성 | Overview · Forecast · Models · Compare · Help | 동일 구조 |
| Overview | 설치/로드 모델 현황, 이벤트 로그, 사용량 | 동일 |
| Forecast | 입력 폼, 차트 렌더링, 결과 Export | 동일 |
| Models | pull/delete, 모델 상세 | 동일 |
| Compare | 다중 모델 비교 | 동일 |
| 인증 | API key 설정 시 401 → 로그인 다이얼로그 | 동일 |

---

## 9. 아키텍처 관점 요약 ("왜 유지보수가 쉬운가")

```
┌──────────────────────────────────────────────────────────────┐
│                        사용자 접점                             │
│  CLI (27 cmds)  /  SDK (16 methods)  /  HTTP (43+ endpoints) │
│  Web Dashboard (Alpine.js + HTMX + Chart.js)                 │
│  TUI Dashboard (Textual)                                     │
├──────────────────────────────────────────────────────────────┤
│                    에이전트 연동 레이어                         │
│  MCP Server (15 tools)  │  A2A (JSON-RPC + SSE)              │
│  LangChain (13 tools)   │  CrewAI / AutoGen / smolagents     │
│  OpenClaw Skill (6 scripts)                                  │
├──────────────────────────────────────────────────────────────┤
│              FastAPI Daemon (tollamad)                        │
│              http://localhost:11435                           │
│  인증(API Key) · Rate Limiting · Prometheus · SSE             │
│  Covariates 호환성 · 메트릭 계산(8종) · 내러티브 생성           │
├──────┬───────┬───────┬───────┬───────┬───────┬───────────────┤
│      │  stdio JSON-lines protocol              │             │
│      ▼       ▼       ▼       ▼       ▼       ▼               │
│    mock    torch   timesfm  uni2ts  sundial  toto            │
│            (Chronos, Granite)                                │
│    각 패밀리별 독립 venv — 의존성 충돌 원천 차단                 │
└──────────────────────────────────────────────────────────────┘
```

**핵심 설계 원칙:**

1. **daemon/core/cli는 가볍게** — 무거운 ML 의존성은 runner extra로 격리
2. **runner family 단위 분리** — 데몬은 라우팅/검증/라이프사이클/에러 매핑에 집중
3. **per-family venv bootstrap** — `tollama runtime install <family>`로 선제 설치, 또는 첫 사용 시 자동 부트스트랩
4. **stdio JSON-lines 프로토콜** — 프로세스 격리 + 언어 독립성 확보
5. **단일 스키마** — `core/schemas.py` 기반으로 CLI/SDK/HTTP/MCP 전체가 동일 요청/응답 구조 공유

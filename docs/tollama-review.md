# Tollama 프로젝트 검토

> **Ollama for Time Series** — 시계열 파운데이션 모델을 단일 API로 실행하고 서빙하는 플랫폼
>
> GitHub: https://github.com/tollama/tollama | License: MIT

---

## 1. 프로젝트 개요

Tollama는 파편화된 TSFM(Time Series Foundation Model)들을 **Ollama 스타일의 통합 인터페이스**로 묶어주는 로컬 우선(local-first) 예측 플랫폼이다. 현재 14개 forecast 모델 패밀리와 테스트용 `mock` 런너를 지원하며, FastAPI 데몬 + CLI + Python SDK + 웹/TUI 대시보드 + MCP/A2A 에이전트 연동까지 갖춘 **풀스택 시계열 예측 플랫폼**으로 구현되어 있다.

### 핵심 컨셉

| 비유 대상 | Ollama | Tollama |
|-----------|--------|---------|
| 대상 모델 | LLM (Llama, Mistral 등) | TSFM (Chronos, TimesFM 등) |
| 핵심 명령어 | `ollama pull`, `ollama run` | `tollama pull`, `tollama run` |
| 서빙 방식 | 로컬 데몬 + REST API | 로컬 데몬 + REST API |
| 모델 격리 | 모델별 관리 | **패밀리별 venv 런타임 격리** |
| SDK | 공식 SDK (Python, JS) | **Python SDK (`from tollama import Tollama`)** |
| 에이전트 연동 | 없음 | **MCP Server + A2A + LangChain/CrewAI/AutoGen/Smolagents** |

---

## 2. 현재 구현 상태 (코드 기반)

### 2.1 아키텍처

```
┌──────────────────────────────────────────────────────────────┐
│                        사용자 접점                             │
│  CLI (tollama)  /  Python SDK  /  HTTP Client  /  Dashboard  │
└────┬─────────────────┬──────────────────┬────────────────────┘
     │                 │                  │
     ▼                 ▼                  ▼
┌──────────────────────────────────────────────────────────────┐
│              에이전트 / 프레임워크 연동 레이어                   │
│  MCP Server (22 tools)  │  A2A (JSON-RPC)  │  OpenClaw Skill │
│  LangChain (13 tools)   │  CrewAI │ AutoGen │ Smolagents     │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│              FastAPI Daemon (tollamad)                        │
│          http://localhost:11435                               │
│                                                              │
│  ┌─ 시스템 / 운영 ──────────────────────────────────────┐   │
│  │ /v1/health  /api/info  /api/version  /api/events(SSE) │   │
│  │ /api/validate  /api/usage  /metrics  /api/dashboard/state ││
│  └───────────────────────────────────────────────────────┘   │
│  ┌─ 모델 관리 ───────────────────────────────────────────┐   │
│  │ /api/tags  /api/show  /api/pull  /api/delete  /api/ps │   │
│  └───────────────────────────────────────────────────────┘   │
│  ┌─ 예측 ────────────────────────────────────────────────┐   │
│  │ /api/forecast  /v1/forecast  /api/forecast/upload      │   │
│  │ /api/forecast/progressive (SSE)  /api/auto-forecast    │   │
│  └───────────────────────────────────────────────────────┘   │
│  ┌─ 고급 분석 ───────────────────────────────────────────┐   │
│  │ /api/analyze    /api/compare      /api/what-if         │   │
│  │ /api/counterfactual  /api/scenario-tree                │   │
│  │ /api/generate   /api/pipeline     /api/report          │   │
│  │ /api/explain-decision  /api/reconcile  /api/conformal  │   │
│  └───────────────────────────────────────────────────────┘   │
│  ┌─ XAI / Trust ─────────────────────────────────────────┐   │
│  │ /api/xai/*  (explain, trust, model-card, alerts,      │   │
│  │ dashboard/history, cache controls, calibration)       │   │
│  └───────────────────────────────────────────────────────┘   │
│  ┌─ 데이터/프로필 ───────────────────────────────────────┐   │
│  │ /api/ingest/upload  /api/modelfiles (CRUD)             │   │
│  └───────────────────────────────────────────────────────┘   │
│  ┌─ 대시보드/A2A ────────────────────────────────────────┐   │
│  │ /dashboard  /.well-known/agent-card.json  /a2a         │   │
│  └───────────────────────────────────────────────────────┘   │
│  ┌─ 미들웨어 ────────────────────────────────────────────┐   │
│  │ API Key 인증  │  Rate Limiting  │  CORS                │   │
│  └───────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
    stdio JSON-lines protocol
    mock / torch / timesfm / uni2ts / sundial / toto /
    lag_llama / patchtst / tide / nhits / nbeatsx /
    timer / timemixer / forecastpfn
```

**핵심 설계 특징:**

- **패밀리별 런타임 격리**: 각 모델 패밀리가 `~/.tollama/runtimes/<family>/venv/`에 독립 가상환경을 가짐. 의존성 충돌 문제를 근본적으로 해결한 설계.
- **stdio JSON-lines 프로토콜**: 데몬과 러너가 표준 입출력을 통한 JSON 라인으로 통신. 프로세스 격리와 언어 독립성 확보.
- **Ollama 스타일 라이프사이클**: `pull` → `run` → `ps` → `rm`의 익숙한 워크플로우.
- **다층 접근 경로**: CLI, SDK, HTTP, MCP, A2A, 프레임워크 어댑터 등 다양한 진입점 제공.

### 2.2 지원 모델 (14개 forecast 패밀리 + 테스트용 `mock`)

| 모델 | 러너 패밀리 | HuggingFace Pull | E2E 테스트 | Covariates 지원 |
|------|------------|:-----------------:|:----------:|----------------|
| Mock (테스트용) | `mock` | N/A | ✅ Pass | 전체 지원 (테스트 목적) |
| Chronos-2 | `torch` | ✅ | ✅ Pass | Past(수치+범주), Future(수치+범주) |
| Granite TTM R2 | `torch` | ✅ | ✅ Pass | Past(수치), Future(수치) |
| TimesFM 2.5-200M | `timesfm` | ✅ | ✅ Pass | Past(수치), Future(수치) |
| Moirai 2.0-R Small | `uni2ts` | ✅ | ✅ Pass | Past(수치), Future(수치) |
| Sundial Base 128M | `sundial` | ✅ | ✅ Pass | ❌ 없음 (target only) |
| Toto Open Base 1.0 | `toto` | ✅ | ⚠️ Skipped | Past(수치)만 |
| Lag-Llama | `lag_llama` | ✅ | ⚠️ Runner/adapter 중심 | ❌ 없음 (target only) |
| PatchTST | `patchtst` | ✅ | ⚠️ Runner/adapter 중심 | ❌ 없음 (target only) |
| TiDE | `tide` | ✅ | ⚠️ Runner/adapter 중심 | Past(수치), Future(수치), Static |
| N-HiTS | `nhits` | ✅ | ⚠️ Runner/adapter 중심 | ❌ 없음 (target only) |
| N-BEATSx | `nbeatsx` | ✅ | ⚠️ Runner/adapter 중심 | ❌ 없음 (target only) |
| Timer Base | `timer` | ✅ | ⚠️ Runner/adapter 중심 | ❌ 없음 (target only) |
| TimeMixer Base | `timemixer` | ✅ | ⚠️ Runner/adapter 중심 | ❌ 없음 (target only) |
| ForecastPFN | `forecastpfn` | ✅ | ⚠️ Runner/adapter 중심 | ❌ 없음 (target only) |

### 2.3 Covariates 지원 (통합 계약)

Tollama의 차별점 중 하나로, 모델마다 다른 covariate 지원 현황을 **통합 계약**으로 관리한다.

- `past_covariates` + `future_covariates`의 일관된 인터페이스
- `best_effort` 모드(기본): 미지원 covariates를 무시하고 경고 반환
- `strict` 모드: 미지원 covariates에 HTTP 400 에러
- 모델별 호환성은 자동으로 데몬이 판단

### 2.4 주요 기능 구현 현황

| 기능 | 상태 | 설명 |
|------|:----:|------|
| CLI 라이프사이클 | ✅ | pull, run, list, show, ps, rm, info, doctor, quickstart, explain, open, dashboard, benchmark, export, quantize + `config/runtime/routing/modelfile/dev/xai` |
| FastAPI HTTP API | ✅ | Ollama-style `/api/*`, stable `/v1/*`, dashboard, XAI/trust, A2A 표면까지 구현 |
| HuggingFace Pull | ✅ | NDJSON 스트리밍 진행률, 오프라인 모드, 라이선스 수락 처리 |
| 런타임 격리 | ✅ | 패밀리별 자동 venv 부트스트랩, Python 버전 제약 관리 |
| Covariates 통합 | ✅ | 14개 forecast 패밀리별 호환성 매트릭스와 best_effort/strict 계약 |
| 프록시/토큰 설정 | ✅ | config, env, CLI 플래그 모두 지원 |
| 진단 엔드포인트 | ✅ | `/api/info`, `tollama info`, `tollama doctor` |
| 테스트/CI | ✅ | Ruff + Pytest + GitHub Actions, 76+ 테스트 파일 |
| Python SDK | ✅ | `Tollama` 클래스 — forecast, compare, what_if, analyze, generate 등 15+ 메서드, 체이닝 workflow API, DataFrame 변환 |
| 웹 대시보드 | ✅ | Alpine.js + HTMX + Chart.js 기반 — `/dashboard`에서 모델 모니터링, 예측 시각화, 이벤트 스트리밍 |
| TUI 대시보드 | ✅ | Textual 프레임워크 — `tollama dashboard` 명령어로 터미널 내 대화형 대시보드 |
| MCP Server | ✅ | FastMCP 기반 22개 도구 — forecast/orchestration 15개 + XAI/trust 7개 |
| A2A 프로토콜 | ✅ | JSON-RPC 라우터, 태스크 큐, 에이전트 카드 (`/.well-known/agent-card.json`) |
| 모델 비교/앙상블 | ✅ | `/api/compare` (다중 모델 비교), `/api/auto-forecast` (자동 선택/앙상블), `ensemble.py` (가중 평균/중앙값 머지) |
| 자동 모델 선택/추천 | ✅ | `auto_select.py`, `recommend.py` — 데이터 특성 기반 모델 순위화 및 추천 |
| What-If 시나리오 | ✅ | `/api/what-if` — 목표/공변량에 multiply/add/replace 변환 적용 |
| 반사실적 분석 | ✅ | `/api/counterfactual` — 개입 기반 궤적 비교 |
| 시나리오 트리 | ✅ | `/api/scenario-tree` — 재귀적 분위수 분기 |
| 합성 데이터 생성 | ✅ | `/api/generate` — 통계 프로필 기반 합성 시계열 |
| 파이프라인 워크플로우 | ✅ | `/api/pipeline` — 분석 → 추천 → pull → 예측 자동화 |
| 리포트 생성 | ✅ | `/api/report` — 분석 + 추천 + 예측 + 내러티브 종합 보고서 |
| TSModelfile | ✅ | 예측 프로필 관리 (CRUD via `/api/modelfiles`) |
| 데이터 인제스트 | ✅ | CSV/Parquet 업로드 및 파싱 (`/api/ingest/upload`, `/api/forecast/upload`) |
| 프레임워크 연동 | ✅ | LangChain(13개 도구), CrewAI, AutoGen, Smolagents 어댑터 |
| OpenClaw Skill | ✅ | 셸 스크립트 + OpenAI 도구 스키마 (`skills/tollama-forecast/`) |
| API Key 인증 | ✅ | Bearer 토큰 인증 (`TOLLAMA_API_KEY`) |
| Rate Limiting | ✅ | 토큰 버킷 방식 요청 제한 |
| Prometheus 메트릭 | ✅ | `/metrics` 엔드포인트 |
| SSE 이벤트 스트리밍 | ✅ | `/api/events` 실시간 이벤트 버스 |
| Progressive Forecast | ✅ | `/api/forecast/progressive` — 단계별 SSE 스트리밍 예측 |
| 설명 가능성 | ✅ | `tollama explain` — 모델 능력, 한계, 라이선스, 사용 사례 |
| 내러티브 생성 | ✅ | 분석/예측/비교/파이프라인에 대한 자연어 설명 자동 생성 |

### 2.5 프로젝트 구조

```
src/tollama/
├── sdk.py           # Python SDK 파사드 (Tollama 클래스)
├── cli/             # Typer CLI (tollama 명령어)
│   ├── main.py      #   전체 CLI 명령어 정의
│   ├── client.py    #   CLI용 HTTP 클라이언트 래퍼
│   ├── dev.py       #   개발자 명령어 (scaffold 등)
│   └── info.py      #   진단 정보 수집
├── client/          # 공유 HTTP 클라이언트
│   ├── http.py      #   TollamaClient 구현
│   └── exceptions.py#   클라이언트 예외 계층
├── core/            # 비즈니스 로직 (45+ 파일)
│   ├── schemas.py   #   Pydantic 요청/응답 모델
│   ├── config.py    #   설정 관리
│   ├── registry.py  #   모델 레지스트리
│   ├── ensemble.py  #   앙상블 머지
│   ├── auto_select.py#  자동 모델 선택
│   ├── recommend.py #   모델 추천
│   ├── scenarios.py #   What-if 시나리오
│   ├── counterfactual.py # 반사실적 분석
│   ├── scenario_tree.py  # 시나리오 트리
│   ├── pipeline.py  #   파이프라인 오케스트레이션
│   ├── report.py    #   리포트 생성
│   ├── synthetic.py #   합성 데이터 생성
│   ├── narratives.py#   자연어 내러티브
│   ├── modelfile.py #   TSModelfile 프로필
│   ├── ingest.py    #   CSV/Parquet 인제스트
│   ├── progressive.py#  단계별 예측
│   ├── hf_pull.py   #   HuggingFace 모델 다운로드
│   └── ...          #   기타 (storage, protocol, metrics 등)
├── daemon/          # FastAPI 데몬
│   ├── app.py       #   전체 API 라우트
│   ├── supervisor.py#   러너 프로세스 슈퍼비전
│   ├── auth.py      #   API Key 인증
│   ├── rate_limiter.py#  Rate Limiting
│   ├── metrics.py   #   Prometheus 메트릭
│   ├── sse.py       #   SSE 이벤트 스트리밍
│   ├── covariates.py#   Covariate 호환성
│   └── dashboard_api.py# 대시보드 상태 API
├── dashboard/       # 웹 대시보드
│   ├── routes.py    #   HTML 파셜 서빙
│   └── static/      #   Alpine.js + HTMX + Chart.js 프론트엔드
├── tui/             # TUI 대시보드 (Textual)
│   ├── app.py       #   TUI 애플리케이션
│   ├── screens/     #   대시보드/예측/모델 상세 화면
│   └── widgets/     #   차트/폼/이벤트로그/모델테이블
├── mcp/             # MCP Server (forecast/orchestration + XAI/trust 도구)
│   ├── server.py    #   FastMCP 등록
│   ├── tools.py     #   도구 핸들러
│   └── schemas.py   #   도구 입력 스키마
├── a2a/             # A2A 프로토콜
│   ├── server.py    #   JSON-RPC 엔드포인트
│   ├── agent_card.py#   에이전트 카드
│   ├── message_router.py # 메시지 라우팅
│   └── tasks.py     #   태스크 상태 머신
├── skill/           # 프레임워크 연동
│   ├── langchain.py #   LangChain 도구 (13개)
│   ├── crewai.py    #   CrewAI 어댑터
│   ├── autogen.py   #   AutoGen 어댑터
│   └── smolagents.py#   Smolagents 어댑터
└── runners/         # 모델별 러너 구현
    ├── mock/        #   테스트용
    ├── torch_runner/#   Chronos, Granite TTM
    ├── timesfm_runner/# TimesFM 2.5
    ├── uni2ts_runner/# Moirai 2.0
    ├── sundial_runner/# Sundial
    ├── toto_runner/ #   Toto
    ├── lag_llama_runner/
    ├── patchtst_runner/
    ├── tide_runner/
    ├── nhits_runner/
    ├── nbeatsx_runner/
    ├── timer_runner/
    ├── timemixer_runner/
    └── forecastpfn_runner/

skills/               # OpenClaw 스킬 (셸 스크립트 + OpenAI 도구 스키마)
scripts/              # 검증/설치 스크립트 (MCP 설치, E2E 테스트 등)
model-registry/       # 모델 메타데이터 레지스트리
examples/             # 요청 JSON 예제 + Jupyter 튜토리얼 노트북
tests/                # 단위 + 통합 테스트 (76+ 파일)
docs/                 # API 레퍼런스, CLI 치트시트, 대시보드 가이드, 트러블슈팅
```

---

## 3. 강점 분석

### 3.1 런타임 격리 설계가 탁월하다

TSFM 생태계의 가장 큰 고통은 의존성 충돌이다. Chronos는 PyTorch 특정 버전, TimesFM은 자체 패키지, Uni2TS는 Python 3.11 필수 등 제각각인데, Tollama는 패밀리별 venv + stdio 프로토콜로 이를 깔끔하게 해결했다. 이것은 단순 API 래퍼를 넘어서는 아키텍처적 가치다.

### 3.2 Covariates 통합 계약이 실용적이다

모델마다 covariate 지원이 제각각인 현실을 `best_effort`/`strict` 모드로 풀어낸 것이 좋다. 사용자는 동일한 요청 포맷으로 모델을 바꿔가며 테스트할 수 있고, 미지원 항목은 경고로 알려준다. 실무에서 매우 유용한 설계다.

### 3.3 Ollama와의 개념적 일관성이 높다

`pull` → `run` → `ps` → `rm`의 워크플로우, localhost 데몬, 환경변수 설정 등 Ollama 사용 경험이 있는 개발자가 즉시 적응할 수 있다. API 포트도 11435로 Ollama(11434)와 나란히 운용 가능하다.

### 3.4 코드 품질이 괜찮다

PEP 621 패키징, src 레이아웃, Ruff 린팅, Pytest, CI — 오픈소스 표준을 잘 따르고 있다. 76+ 테스트 파일에 문서화 수준도 높다.

### 3.5 SDK와 Workflow API 설계가 잘 되어 있다

`Tollama` 클래스가 깔끔한 파이썬 파사드를 제공한다. `to_df()`로 DataFrame 변환, `workflow()` 체이닝 API (`analyze().forecast().what_if()`), `then_compare()`/`then_what_if()` 연속 메서드, pandas Series/DataFrame 입력 지원 등 개발자 경험이 좋다.

```python
from tollama import Tollama

t = Tollama()
result = t.forecast(model="chronos2", series=[{"target": [10, 11, 12], "freq": "D"}], horizon=7)
df = result.to_df()

# 체이닝 workflow
with Tollama() as sdk:
    flow = sdk.workflow(series).analyze().auto_forecast(horizon=3)
```

### 3.6 고급 분석이 단순 예측 도구를 넘어선다

What-if 시나리오, 반사실적 궤적, 확률적 시나리오 트리, 합성 데이터 생성, 다단계 파이프라인, 종합 리포트 + 자연어 내러티브까지 갖추고 있다. 단순 모델 서빙 도구가 아닌 **시계열 분석 플랫폼** 수준이다.

### 3.7 에이전트 연동 범위가 포괄적이다

MCP Server(22개 도구), A2A 프로토콜(JSON-RPC + 태스크 큐), LangChain(13개 도구), CrewAI, AutoGen, Smolagents 어댑터, OpenClaw 스킬까지 — 주요 AI 에이전트 프레임워크를 거의 모두 커버한다. AI 에이전트가 시계열 예측을 "도구"로 사용할 수 있는 인프라가 이미 완비되어 있다.

### 3.8 웹/TUI 이중 대시보드로 운영 가시성을 확보했다

웹 대시보드(Alpine.js/HTMX/Chart.js)는 모델 모니터링, 예측 시각화, 이벤트 스트리밍을 브라우저에서 제공하고, TUI 대시보드(Textual)는 터미널 환경에서 동일한 기능을 대화형으로 제공한다. 서버/로컬 환경 모두에서 운영 상태를 확인할 수 있다.

---

## 4. 개선 필요 사항

### 4.1 GPU 관리가 체계적이지 않다

`TOLLAMA_TOTO_INTEGRATION_CPU=1` 같은 환경변수로 개별 모델의 디바이스를 제어할 수 있지만, GPU 할당, 멀티 GPU, VRAM 메모리 관리에 대한 체계적인 API가 없다. TSFM은 모델 크기에 따라 GPU 메모리가 중요한 이슈이므로 다음이 필요하다:

- 디바이스 선택 API (`device: "cuda:0"`, `device: "cpu"`)
- VRAM 메모리 제한 설정
- 멀티 GPU 할당 전략
- 런타임 GPU 사용량 모니터링

### 4.2 배치 처리 패턴의 문서화가 부족하다

파일 업로드 엔드포인트(`/api/forecast/upload`)와 CSV/Parquet 인제스트(`/api/ingest/upload`)가 존재하지만, 대량 시계열 배치 처리에 대한 문서와 가이드라인이 부족하다. 프로덕션에서는 수천 개 시계열을 한번에 예측하는 것이 일반적인데, 병렬 인퍼런스, 청킹, 진행률 추적 등 배치 오케스트레이션 패턴이 명시적으로 문서화되어야 한다.

### 4.3 클러스터링/멀티 인스턴스 지원이 없다

데몬이 단일 노드(localhost) 설계다. 로드 밸런싱, 분산 인퍼런스, 멀티 인스턴스 조율 기능이 없어 대규모 동시 요청 처리 시 수평 확장이 불가하다.

### 4.4 파인튜닝 워크플로우가 없다

모든 모델을 사전 학습된 상태로만 사용한다. 도메인 특화 시계열 패턴에 대한 파인튜닝, 적응, 전이 학습 파이프라인이 없다. 특정 도메인(금융, 에너지, IoT 등)에서 정확도를 높이려면 파인튜닝 지원이 필요하다.

### 4.5 클라우드 배포 옵션이 없다

Dockerfile은 존재하지만, docker-compose, Kubernetes 매니페스트, Helm 차트가 없다. 로컬 우선 설계가 강점이지만, 팀 서버나 클라우드 환경 배포를 위한 컨테이너화 가이드가 필요하다.

### 4.6 모델 레지스트리가 정적이다

`model-registry/registry.yaml`에 모델 메타데이터가 정의되어 있지만, 동적 모델 등록이나 자동 신규 모델 탐색 기능이 없다. 커뮤니티 기여 모델 추가에 코드 변경이 필요하므로, 외부 레지스트리 연동이나 동적 등록 API가 있으면 생태계 확장에 유리하다.

---

## 5. 경쟁 구도에서의 포지셔닝

| 경쟁 제품 | 특징 | Tollama 대비 |
|-----------|------|-------------|
| **Nixtla TimeGPT** | 클라우드 API 기반 상용 서비스 | Tollama는 로컬 우선 + 오픈소스 + SDK/대시보드/MCP/A2A 완비 |
| **NeuralForecast** | Python 라이브러리, 전통 DL 모델 포함 | Tollama는 TSFM 특화 + 서빙 레이어 + 에이전트 연동 + 고급 분석 |
| **AutoGluon-TimeSeries** | AWS 기반 AutoML | Tollama는 벤더 독립적, 경량, 다중 프레임워크 연동 |
| **HuggingFace Pipeline** | 범용 모델 허브 | Tollama는 시계열 특화 UX + 런타임 격리 + What-if/반사실적 분석 |
| **GluonTS** | Amazon의 시계열 툴킷 | Tollama는 파운데이션 모델 특화 + 에이전트 생태계 연동 |

**Tollama의 고유 포지션**: "여러 TSFM을 로컬에서 런타임 격리와 함께 단일 API로 서빙"하면서, **SDK + 이중 대시보드 + MCP/A2A + 4대 에이전트 프레임워크 연동 + 고급 분석(What-if, 반사실적, 시나리오 트리, 파이프라인, 리포트)**까지 갖춘 조합은 현재 경쟁 제품에 없는 독보적인 포지션이다.

---

## 6. 제안 로드맵

### 구현 완료 항목 (기존 로드맵 대비)

- [x] Python SDK 출시 (`from tollama import Tollama`)
- [x] 모델 비교/벤치마크 (`/api/compare`, `/api/auto-forecast`)
- [x] 자동 모델 추천/라우팅 (`auto_select.py`, `recommend.py`, `/api/auto-forecast`, `/api/pipeline`)
- [x] MCP Server (22개 도구)
- [x] Dockerfile 작성
- [x] A2A 프로토콜 (JSON-RPC)
- [x] OpenAPI 스펙 자동 생성 (`/docs`, `/redoc`, `/openapi.json`)
- [x] 대시보드 (웹 + TUI)
- [x] 모델 앙상블 (`ensemble.py`, auto-forecast 앙상블 전략)

### Phase 1: 프로덕션 강화 (단기)

- [ ] GPU 리소스 관리 (디바이스 선택, VRAM 제한, 멀티 GPU)
- [ ] 배치 처리 문서화 및 오케스트레이션 패턴
- [ ] 클라우드 배포 지원 (Dockerfile, docker-compose, K8s 매니페스트)
- [ ] 성능 벤치마킹 스위트 (레이턴시, 처리량, 메모리)

### Phase 2: 확장성 (중기)

- [ ] 멀티 인스턴스 / 클러스터링 지원
- [ ] 파인튜닝 워크플로우 (도메인 적응)
- [ ] 동적 모델 레지스트리 (커뮤니티 모델 기여)
- [ ] 모델 버전 관리 및 A/B 테스트

### Phase 3: 생태계 확장 (장기)

- [ ] 데이터 드리프트 감지 및 모니터링
- [ ] 자동 재학습 트리거
- [ ] 다국어 SDK (TypeScript/Go)
- [ ] 커뮤니티 모델/모델파일 마켓플레이스

---

## 7. 종합 평가

**현재 상태**: Tollama는 핵심 아키텍처(데몬 + 런타임 격리 + CLI)를 넘어 **Python SDK, 이중 대시보드(웹+TUI), MCP Server(22개 도구), A2A 프로토콜, 4대 에이전트 프레임워크 연동, 고급 분석(What-if, 반사실적, 시나리오 트리, 파이프라인, 리포트), XAI/trust 레이어**까지 갖춘 **기능적으로 성숙한 시계열 예측 플랫폼**이다. 초기 비전에서 제시한 SDK, Dashboard, MCP, A2A 등이 모두 구현되어 있다.

**비전 달성도**: 초기 컨셉에서 제시한 "개발자 benefit(API, SDK, Dashboard)"과 "AI 에이전트 benefit(MCP, A2A, Skills)"이 **모두 구현 완료**되었다. 현재 남은 과제는 기능 구현이 아닌 **운영 성숙도**(GPU 관리, 클러스터링, 클라우드 배포, 파인튜닝)에 있다.

**가장 임팩트 있는 다음 스텝**: GPU 리소스 관리 + 클라우드 배포 지원. 이 두 가지가 있으면 로컬 개발 도구에서 **팀/조직 수준의 프로덕션 시계열 예측 인프라**로 도약할 수 있다.

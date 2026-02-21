# tollama Todo List (Ollama for TSFM)

기준 시점: 2026-02-21 (repo 현재 구현 반영)

우선순위:
- `P0` = 필수(MVP)
- `P1` = 제품화 핵심
- `P2` = 확장/최적화
- `R` = 리서치/조사

상태:
- `[x]` 구현됨
- `[~]` 부분 구현됨
- `[ ]` 미구현

---

## 0) 배경 및 목표 정리

- [~] (R) TSFM 후보 모델군 정리 및 비교표 유지
  - 대상: Chronos2, Moirai(Uni2TS), TimesFM, IBM(Granite/TTM/PatchTST 계열), Toto, Sundial
  - 현재: 모델 매트릭스/라이선스/capabilities는 `model-registry/registry.yaml`, `README.md`, `docs/covariates.md`, `docs/how-to-run.md`에 반영됨
  - TODO: 입력 방식/아키텍처/배포 조건까지 포함한 심화 비교표로 확장

---

## 1) Ollama 벤치마킹 핵심 5요소 -> tollama Todo

### 1) 모델 설정 추상화: "Modelfile" 개념 도입

- [x] (P0) TSModelfile 스펙 정의 (YAML)
  - 필수 필드(예시):
    - `model`, `horizon`, `quantiles`, `options`, `parameters`, `covariate_mappings`, `preprocessing`
- [x] (P1) TSModelfile 등록/조회/삭제 API + CLI 지원
  - 현재: daemon `GET/POST/DELETE /api/modelfiles*`, CLI `tollama modelfile create/list/show/rm` 구현
- [ ] (P1) 모델별 최적 전처리 프리셋을 Modelfile 템플릿으로 제공
  - 목표: 사용자는 내부 로직 몰라도 `forecast(data)`만으로 맞는 전처리+추론 실행

### 2) 실행 런타임 격리: Dependency-Free / Conflict-Free Inference

- [x] (P0) Worker-per-model-family(런너 분리) 기본 구조 유지/강화
  - 현재 family: `torch`, `timesfm`, `uni2ts`, `sundial`, `toto` (+ `mock`)
- [~] (P1) 런너별 종속성 충돌 방지 정책(버전 pin, lockfile, 독립 venv) 확립
  - 현재: optional extras 분리 + family별 독립 runtime venv 자동화 구현(`~/.tollama/runtimes/<family>/venv/`, `tollama runtime install/list/update/remove`)
  - TODO: 버전 pin/lockfile 정책과 설치 재현성 보강
- [ ] (P2) "Python 의존 최소화" 장기 트랙 정의
  - ONNX/TensorRT/LibTorch(C++)/Rust/Go 코어화 가능성 검토
  - (선택) 특정 모델용 컴파일 엔진 기반 고성능 런타임

### 3) 통합 데이터 어댑터: Unified "Tokenizer layer" for Time Series

- [x] (P0) 표준 입력 스키마 확정
  - 현재: `ForecastRequest`/`SeriesInput`에 `history(target+timestamps+freq)`, `past_covariates`, `future_covariates`, `static_covariates` 반영
- [~] (P1) Unified Data Adapter 계층 구현
  - 현재: family별 adapter에서 변환 구현(Chronos binning/tokenization 계열, TimesFM/Moirai patch/window 계열, Sundial 샘플 기반, Toto variate 기반)
  - 현재: `src/tollama/core/ingest.py`로 CSV/Parquet -> canonical `SeriesInput` 변환 경로 추가
  - TODO: 공통 계층으로 완전 통합, covariates/전처리 규칙까지 일원화
- [~] (P1) 전처리 공통 라이브러리화
  - 현재: covariates 정규화/strict-best_effort는 daemon 공통 처리
  - 현재: `freq="auto"` 기본 추론(타임스탬프 기반)은 daemon 공통 처리
  - TODO: 리샘플링/결측/스케일링/윈도잉/타임존 처리의 공통 유틸 정리

### 4) 하드웨어 가속 자동 감지: Backend Selection

- [~] (P0) 디바이스 탐지 및 우선순위 정책 수립 (`cpu`/`cuda`/`mps` 등)
  - 현재: 일부 runner에서 `cpu`/`cuda` 감지 및 옵션 기반 선택 제공
  - TODO: 전 family 공통 정책 + `mps` 포함 통합 기준
- [ ] (P1) "작은 TSFM은 CPU도 충분" 가정의 CPU 최적화 경로 마련 (AVX/NEON 등 장기)
- [ ] (P2) 동일 모델의 quantization 버전 선택 로직(가능한 경우)
- [~] (P2) 런너별 메모리/VRAM 회수 정책 강화(`keep_alive`, idle timeout, 강제 unload)
  - 현재: `keep_alive` + `unload` + 만료 시 runner stop 지원
  - TODO: 기본 idle 정책/상세 회수 정책/백오프 포함 고도화

### 5) 표준 예측 인터페이스: API Protocol

- [~] (P0) 표준 출력 규격 확정
  - 현재 Output: `mean`, `quantiles`, `warnings`, `usage`, optional
    `metrics`(`mape`/`mase`/`mae`/`rmse`/`smape`/`wape`/`rmsse`/`pinball`),
    optional `timing`(`model_load_ms`/`inference_ms`/`total_ms`),
    optional `explanation`, optional `narrative`(요청 `response_options.narrative=true`) 구현
  - TODO Meta 확장: `model_version/digest`, `median` 명시 규격 보강
- [~] (P1) 신뢰도/품질 메타 지표 설계
  - 현재: 요청 `series.actuals` + `parameters.metrics` 기반
    `mape`/`mase`/`mae`/`rmse`/`smape`/`wape`/`rmsse`/`pinball` 계산 및 응답 `metrics` 제공
  - TODO: interval coverage 확장
- [~] (P1) 포맷 지원 확대: JSON + Parquet (대용량/배치)
  - 현재: `data_url` + upload endpoint로 CSV/Parquet 입력 forecast 지원
  - TODO: 대용량 배치 출력(Parquet writer) 및 스트리밍 최적화

---

## 2) 모델 온보딩/지원 범위 Todo

### A) 이미 진행/우선 순위 높은 라인업

- [x] (P0/P1) Chronos2: covariates 포함 표준 forecasting 구현
- [x] (P1) TimesFM 2.5: XReg(best effort/strict) + quantile 매핑 구현
- [~] (P1) Moirai(Uni2TS): 공변량/멀티시리즈 지원 + 성능/메모리 튜닝 고도화 필요
- [x] (P1) IBM Granite(TTM/PatchTST 계열): known-future vs conditional covariates 매핑 구현

### B) 추가 확장 모델

- [x] (R->P2) Toto 통합 1차 완료
  - 현재: 추론 API/입력 포맷/라이선스/샘플 기반 quantile 산출 구현
  - TODO: 추가 최적화 및 기능 확장
- [x] (R->P2) Sundial 통합 1차 완료
  - 현재: flow 계열 샘플 생성 결과를 통합 출력(mean/quantile)로 변환
  - TODO: 추가 최적화 및 기능 확장
- [~] (R) 각 모델별 지원 기능 체크리스트 유지
  - 현재: covariates 중심 capability matrix는 문서화됨
  - TODO: 멀티변량/샘플링/최대 context/horizon까지 체크리스트 고도화

---

## 3) 개발 단계(Stage) 기반 Todo 로드맵

### Stage 1 (P0~P1): "사용자 경험이 Ollama처럼 느껴지는 제품"

- [x] (P0) `/api/*` 기반 `pull/list/show/ps/forecast`, 스트리밍 진행률, `keep_alive` 완성
- [x] (P1) `tollama config`, `tollama info`, `/api/info`, `tollama doctor`까지 UX 정리
  - 현재: `doctor` 구현(`pass/warn/fail`, `--json`, exit code `0/1/2`)
  - 현재: `tollama run --dry-run` + `/api/validate` 구현(무추론 요청 검증)
  - 현재: `tollama list`/`tollama ps` 테이블 기본 출력 + `--json` 호환 모드 지원
  - 현재: `tollama config keys`/`tollama config init` 및 unknown key 추천(`Did you mean`) 지원
  - 현재: daemon/client/CLI/MCP/LangChain 공통 에러 `hint` 전달(기존 `detail` 호환 유지)
  - 현재: long-running CLI에 `--progress auto|on|off` 도입(`pull`/`run`/`quickstart`/`runtime install`), stdout 계약 유지(stder-only progress)
  - 현재: `tollama run`에서 MODEL 생략 시 설치된 모델 선택 프롬프트 + `--interactive` 예제 payload 선택 지원
- [ ] (P1) "pull이 런타임까지 설치"되는 installer(런타임 venv 자동 구성) 확립

### Stage 2 (P1): "공통 전처리/공변량 표준화"

- [x] (P1) covariates contract 확정 + strict/best_effort 동작 통일
- [x] (P1) 모델별 공변량 매핑 구현 + 호환성 매트릭스 제공(문서 + `/api/info`)
- [~] (P1) 데이터 어댑터(패치/바이닝/연속값) 공통 모듈화

### Stage 3 (P2): "고성능/의존성 최소화(선택 트랙)"

- [ ] (P2) Chronos/TimesFM 등 일부를 ONNX/TensorRT/LibTorch 기반으로 PoC
- [ ] (P2) 코어를 Go/Rust로 옮기고 추론 엔진만 동적 로딩하는 구조 검토
- [ ] (P2) 하드웨어별 최적화(quantization 선택, CPU SIMD 가속, GPU 커널 튜닝)

---

## 4) 단기 액션 아이템 (이번 주~다음 주)

- [x] (P1) Toto/Sundial 리서치 카드 생성
  - 현재: 통합 runner + docs/examples/tests까지 1차 완료
- [x] (P1) OpenClaw `tollama-forecast` 스킬 v3 운영 보강
  - 현재: `skills/tollama-forecast/` 추가(`SKILL.md`, `bin/*.sh`, `examples/*.json`)
  - 현재: 공통 helper `bin/_tollama_lib.sh`로 에러 분류/HTTP 유틸 통합
  - 현재: `TOLLAMA_JSON_STDERR=1` 구조화 stderr 지원(`code/subcode/exit_code/message/hint`)
  - 현재: lifecycle wrapper 스크립트 추가(`tollama-pull.sh`, `tollama-rm.sh`, `tollama-info.sh`)
  - 현재: OpenAI function 정의 파일 `openai-tools.json` 추가
  - 현재: 경로/exec host/PATH/auto-pull/endpoint/timeout 이슈를 스킬 계층에서 고정
  - 현재: metadata 완화(`bins=["bash"]`, `anyBins=["tollama","curl"]`) + daemon-only `available` 정책 확정
  - 현재: 스크립트 exit code contract v2(`0/2/3/4/5/6/10`)로 통일
  - 현재: 정적 검증 스크립트 `scripts/validate_openclaw_skill_tollama_forecast.sh` 추가
  - 현재: CI(`.github/workflows/ci.yml`)에서 skill validator 실행
  - 현재: 스크립트 동작 회귀 테스트 `tests/test_openclaw_skill_tollama_forecast_scripts.py` 추가
  - 현재: `README.md`/`roadmap.md`/`SKILL.md`에 스킬 구현 상세(스크립트별 계약/HTTP 경로/에러 출력 포맷) 반영
  - 현재: OpenClaw 운영 런북 추가(`docs/openclaw-sandbox-runbook.md`,
    `docs/openclaw-gateway-runbook.md`)
- [x] (P1) Python SDK / LangChain Tool 래퍼 추가
  - 현재: `src/tollama/skill/langchain.py`에
    `TollamaForecastTool`/`TollamaAutoForecastTool`/`TollamaAnalyzeTool`/
    `TollamaWhatIfTool`/`TollamaPipelineTool`/`TollamaCompareTool`/`TollamaRecommendTool`/
    `TollamaHealthTool`/`TollamaModelsTool` 구현
  - 현재: LangChain 주요 툴 description에 schema/model/example 가이드 추가
  - 현재: 팩토리 `get_tollama_tools(base_url="http://127.0.0.1:11435", timeout=10.0)` 제공
  - 현재: optional extra `.[langchain]`(`langchain-core`) 추가
  - 현재: `tests/test_langchain_skill.py` 검증 추가
  - 현재: 고수준 SDK `src/tollama/sdk.py` 추가(`from tollama import Tollama`)
    - dict/list/pandas Series/DataFrame 입력 정규화
    - 결과 helper(`mean`, `quantiles`, `to_df()`) 제공
    - `tests/test_sdk.py` 검증 추가
- [~] (P1) MCP 서버(Claude Code 네이티브 연동) 1차 도입
  - 현재: 공통 HTTP client `src/tollama/client/` 신설(CLI/MCP 공용)
  - 현재: `src/tollama/mcp/` 서버/툴 핸들러/엔트리포인트 추가
  - 현재: `tollama-mcp` 스크립트 및 optional extra `.[mcp]` 추가
  - 현재: `scripts/install_mcp.sh`, `CLAUDE.md` 추가
  - 현재: 실 SDK 환경 E2E smoke 완료(`tollama-mcp` stdio + live daemon tool call)
  - 현재: `README.md`/`roadmap.md`/`CLAUDE.md`에 MCP 구현 상세(툴 계약/에러 매핑/기본값) 반영
  - 현재: MCP 15개 툴(`health/models/forecast/auto_forecast/analyze/generate/counterfactual/scenario_tree/report/what_if/pipeline/compare/recommend/pull/show`) description에
    입력 스키마/모델 예시/호출 예시 추가
  - TODO: Claude Desktop 운영 가이드(권한/세션/배포 정책) 보강
- [~] (P1) A2A(Agent2Agent) 프로토콜 1차 도입
  - 현재: discovery `GET /.well-known/agent-card.json` + legacy alias `GET /.well-known/agent.json` 추가
  - 현재: JSON-RPC endpoint `POST /a2a` 추가
  - 현재: 메서드 `message/send`, `message/stream(SSE)`, `tasks/get`, `tasks/query`, `tasks/cancel` 구현
  - 현재: API key 설정 시 discovery + `/a2a` 모두 bearer 인증 적용(기본 authenticated discovery)
  - 현재: in-memory task lifecycle + 모델 family 식별 시 cancel 시도(`runner_manager.stop`) 구현
  - 현재: outbound helper `src/tollama/a2a/client.py` 추가(discover/send/get/query/cancel/poll)
  - 현재: 회귀 테스트 `tests/test_a2a_agent_card.py`, `tests/test_a2a_server.py`,
    `tests/test_a2a_tasks.py`, `tests/test_a2a_client.py` 추가
  - TODO: `tasks/subscribe`, push notification config, extended agent card 확장
- [x] (P1) Real-time SSE 스트리밍 1차
  - 현재: `GET /api/events` 추가(키 범위 SSE 구독, event filter/heartbeat/max_events 지원)
  - 현재: `/api/forecast`, `/api/analyze` 경로에서
    `model.loaded`, `forecast.progress`, `forecast.complete`, `analysis.complete`,
    `anomaly.detected` 이벤트 발행
  - 현재: `POST /api/forecast/progressive` 추가(단계별 `model.selected` -> `forecast.progress` -> `forecast.complete`)
  - 현재: `message/stream` A2A SSE 이벤트(`TaskStatusUpdateEvent`, `TaskArtifactUpdateEvent`) 구현
  - 현재: 회귀 테스트 `tests/test_sse.py`, `tests/test_progressive.py`, `tests/test_a2a_server.py` 확장
- [x] (P1) Agentic 비교/추천 기능 1차
  - 현재: `/api/compare` 추가(다중 모델 동일 요청 비교, per-model success/error 반환)
  - 현재: `tollama_compare` MCP/LangChain 툴 추가
  - 현재: `tollama_recommend` MCP/LangChain 툴 추가(registry+capability 기반 랭킹)
  - 현재: `tests/test_compare.py`, `tests/test_recommend.py` 추가
- [x] (P1) Agentic 시계열 분석 기능 1차
  - 현재: `/api/analyze` 추가(주기/계절성/추세/이상치/정상성/품질 점수)
  - 현재: `TollamaClient`/`AsyncTollamaClient`/SDK `Tollama.analyze()` 추가
  - 현재: `tollama_analyze` MCP/LangChain/CrewAI/AutoGen/smolagents 툴 추가
  - 현재: `tests/test_series_analysis.py`, `tests/test_daemon_api.py`,
    `tests/test_client_http.py`, `tests/test_mcp_tools.py`, `tests/test_langchain_skill.py`,
    `tests/test_agent_wrappers.py`, `tests/test_sdk.py`, `tests/test_schemas.py` 검증 추가
- [x] (P1) Synthetic Time Series Generation 1차(통계 기반, deterministic)
  - 현재: `/api/generate` 추가(`GenerateRequest`/`GenerateResponse`)
  - 현재: `src/tollama/core/synthetic.py` 추가(시계열 프로파일 기반 synthetic 생성)
  - 현재: `TollamaClient`/`AsyncTollamaClient`/SDK `Tollama.generate()` 추가
  - 현재: `tollama_generate` MCP/LangChain/CrewAI/AutoGen/smolagents/A2A 경로 추가
  - 현재: `tests/test_synthetic.py` + 연관 client/MCP/LangChain/A2A/schema 회귀 테스트 추가
- [x] (P1) Composite Report / Counterfactual / Scenario-Tree 1차
  - 현재: `/api/report`, `/api/counterfactual`, `/api/scenario-tree` 추가
  - 현재: core 모듈 `src/tollama/core/report.py`, `counterfactual.py`, `scenario_tree.py` 추가
  - 현재: `Counterfactual*`, `ScenarioTree*`, `Report*`, `AnomalyRecord` 스키마 추가
  - 현재: `TollamaClient`/`AsyncTollamaClient`/SDK 메서드(`report`, `counterfactual`, `scenario_tree`) 추가
  - 현재: MCP/LangChain/CrewAI/AutoGen/smolagents 툴(`tollama_report`, `tollama_counterfactual`, `tollama_scenario_tree`) 추가
  - 현재: 회귀 테스트 `tests/test_report.py`, `tests/test_counterfactual.py`, `tests/test_scenario_tree.py` 및 연동 계층 테스트 확장
- [x] (P1) Zero-config Auto-Forecast 1차
  - 현재: `/api/auto-forecast` 추가(설치 모델 기준 auto/fastest/best_accuracy/ensemble 선택)
  - 현재: `AutoForecastRequest/AutoSelectionInfo/AutoForecastResponse` 스키마 추가
  - 현재: explicit `model`은 기본 hard override, `allow_fallback=true`일 때만 fallback 수행
  - 현재: ensemble은 weighted `mean`/`median` 집계 + bounded parallel 실행 (quantiles 생략 + warning)
  - 현재: `TollamaClient`/`AsyncTollamaClient`/SDK `Tollama.auto_forecast()` 추가
  - 현재: `tollama_auto_forecast` MCP/LangChain/CrewAI/AutoGen/smolagents 툴 추가
  - 현재: `tests/test_auto_forecast.py` + 연관 스키마/클라이언트/에이전트 래퍼 회귀 테스트 추가
- [x] (P1) What-If Scenarios 1차
  - 현재: `/api/what-if` 추가(기준 예측 + 시나리오별 변환 결과 side-by-side 반환)
  - 현재: 시나리오 변환 모듈 `src/tollama/core/scenarios.py` 추가
    (multiply/add/replace, target/past_covariates/future_covariates)
  - 현재: `WhatIfRequest/WhatIfResponse` 스키마 및 `continue_on_error` 부분 성공 모드 추가
  - 현재: `TollamaClient`/`AsyncTollamaClient`/SDK `Tollama.what_if()` 추가
  - 현재: `tollama_what_if` MCP/LangChain 툴 추가
  - 현재: `tests/test_what_if.py` + 연관 스키마/클라이언트/에이전트 래퍼 회귀 테스트 추가
- [x] (P1) Forecast Timing/Explainability 1차
  - 현재: forecast 응답에 `timing`(`model_load_ms`,`inference_ms`,`total_ms`) + enriched
    `usage`(`runner`,`device`,`peak_memory_mb`) 포함
  - 현재: `src/tollama/core/explainability.py` 기반 `explanation` 필드 자동 생성
  - 현재: `tests/test_explainability.py` + daemon/runner/client 회귀 테스트 추가
- [x] (P1) Prometheus Metrics Endpoint 1차
  - 현재: `GET /metrics` 추가(옵션 의존성: `prometheus-client`)
  - 현재: `tollama_forecast_requests_total`, `tollama_forecast_latency_seconds`,
    `tollama_models_loaded`, `tollama_runner_restarts_total` 노출
  - 현재: `ForecastMetricsMiddleware`로 forecast/auto-forecast/compare 경로 계측
  - 현재: `tests/test_daemon_metrics.py` 추가(성공/실패 카운터, gauge, restart counter)
- [x] (P1) API Key Auth 1차
  - 현재: `config.json` `auth.api_keys` 기반 bearer 토큰 인증 추가
  - 현재: 키 미설정 시 인증 비활성(local-first 기본), 설정 시 전 엔드포인트 인증 강제
  - 현재: `TollamaClient`/`AsyncTollamaClient`/SDK/CLI에서 `api_key` 전달 지원
  - 현재: `tests/test_auth.py`, `tests/test_config.py`, `tests/test_cli_info.py` 검증 추가
- [x] (P1) Usage Metering + Rate Limiting 1차
  - 현재: SQLite usage 집계(`~/.tollama/usage.db`) 추가
    (`request_count`, `total_inference_ms`, `series_processed`)
  - 현재: `GET /api/usage` 추가(키별 사용량 조회)
  - 현재: optional token-bucket rate limiting 추가
    (`TOLLAMA_RATE_LIMIT_PER_MINUTE`, `TOLLAMA_RATE_LIMIT_BURST`)
  - 현재: `tests/test_metering.py` 검증 추가
- [x] (P1) Agentic Pipeline Endpoint 1차
  - 현재: `/api/pipeline` 추가(analyze -> recommend -> optional pull -> auto-forecast)
  - 현재: `PipelineRequest`/`PipelineResponse` + `src/tollama/core/pipeline.py` 추가
  - 현재: `TollamaClient`/`AsyncTollamaClient`/SDK `Tollama.pipeline()` 추가
  - 현재: `tollama_pipeline` MCP/LangChain 툴 추가
  - 현재: `tests/test_pipeline.py` + 연관 client/MCP/LangChain/SDK/schema 회귀 테스트 추가
- [x] (P0) 온보딩 마찰 완화 1차
  - 현재: `tollama quickstart` 명령 추가(daemon 확인 -> pull -> demo forecast -> next steps 출력)
  - 현재: `README.md` 상단을 설치/quickstart/SDK/agent 중심으로 재구성
  - 현재: 모델별 설치/실행 가이드를 `docs/models.md`로 분리
- [x] (P1) 탐색성/배포 편의 1차
  - 현재: 노트북 추가(`examples/quickstart.ipynb`, `examples/agent_demo.ipynb`)
  - 현재: 컨테이너 파일 추가(`Dockerfile` multi-stage, `.dockerignore`, `docker-compose.yml` GPU profile)
  - 현재: `pyproject.toml`에 PyPI 메타데이터(`keywords`, `classifiers`, `project.urls`) 보강
- [x] (P2) LangChain async 툴 실행 경로 완성
  - 현재: `AsyncTollamaClient` 추가(`src/tollama/client/http.py`)
  - 현재: `TollamaHealthTool`/`TollamaModelsTool`/`TollamaForecastTool`/`TollamaAnalyzeTool`/
    `TollamaCompareTool`/`TollamaRecommendTool`의 `_arun`이 실제 async 호출 사용
  - 현재: `tests/test_langchain_skill.py`, `tests/test_client_http.py` async 경로 검증 추가
- [x] (P2) 추가 Agent 프레임워크 래퍼 제공
  - 현재: `src/tollama/skill/crewai.py` (`get_crewai_tools`)
  - 현재: `src/tollama/skill/autogen.py`
    (`get_autogen_tool_specs`, `get_autogen_function_map`, `register_autogen_tools`)
  - 현재: `src/tollama/skill/smolagents.py` (`get_smolagents_tools`)
  - 현재: 공통 스펙 어댑터 `src/tollama/skill/framework_common.py` 재사용
  - 현재: `tests/test_agent_wrappers.py` 추가
- [x] (P2) SDK vs raw 벤치마크 스크립트 추가
  - 현재: `benchmarks/tollama_vs_raw.py` (LOC + time-to-first-forecast 비교)
  - 현재: `tests/test_benchmark_tollama_vs_raw.py` 추가
- [x] (P1) TSModelfile 스펙 초안 작성 + parser 구현
  - 현재: YAML parser(`src/tollama/core/modelfile.py`) + request 우선순위 규칙(`request > modelfile > defaults`) 반영
- [~] (P1) Unified Data Adapter 설계 문서
  - 현재: `docs/covariates.md`에 모델별 매핑은 정리됨
  - TODO: patching/binning/flow 변환 단계 표준 설계 문서로 분리
- [x] (P1) covariates strict/best_effort 정책 모델별 표준화 (warning 템플릿 포함)
- [x] (P1) `tollama doctor` 구현
  - 현재: python constraint/runtime/disk/token/daemon/config 체크 + 요약/종료코드 제공
- [~] (P0) 공개 릴리스 라이선스/컴플라이언스 체크리스트 운영
  - 현재: `docs/public-release-checklist.md` 추가(업스트림 라이선스 검증/서드파티 라이선스 인벤토리/제한 라이선스 정책)
  - TODO: 릴리스 태그마다 체크 결과(`docs/releases/`)를 남기고, `cc-by-nc-4.0` 모델 공개 정책을 최종 확정

- [x] (P1) 6개 모델 e2e 통합 테스트 매트릭스 재검증 + 문서 동기화
  - 기준 일자: `2026-02-17`
  - pass: `chronos2`, `granite-ttm-r2`, `timesfm-2.5-200m`,
    `moirai-2.0-R-small`, `sundial-base-128m`
  - skipped: `toto-open-base-1.0` (`toto` 패키지 미설치 환경)
  - 추가 재검증(per-family runtime isolation smoke, TimesFM pin 수정 후):
    `chronos2`, `granite-ttm-r2`, `timesfm-2.5-200m`, `moirai-2.0-R-small`,
    `sundial-base-128m`, `toto-open-base-1.0` 모두 pass
- [x] (P1) 최종 E2E 회귀(per-family/skill/MCP) 확인
  - 기준 일자: `2026-02-20`
  - pass: `bash scripts/e2e_all_families.sh`
  - pass: `bash scripts/e2e_skills_test.sh`
  - pass: MCP stdio SDK smoke(`tollama_health`, `tollama_models`, `tollama_show`, `tollama_forecast`)
- [x] (P1) Phase 4 신규 기능 E2E 재검증
  - 기준 일자: `2026-02-20`
  - pass: `ruff check .`
  - pass: `pytest -q`
  - pass: ingest/modelfile/ensemble 신규 테스트
    (`tests/test_ingest.py`, `tests/test_modelfile.py`, `tests/test_ensemble.py`)

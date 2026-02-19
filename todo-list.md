# tollama Todo List (Ollama for TSFM)

기준 시점: 2026-02-17 (repo 현재 구현 반영)

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

- [ ] (P0) TSModelfile 스펙 정의 (YAML/텍스트 중 택1)
  - 필수 필드(예시):
    - `FROM <base_model>` (예: `timesfm-2.5-200m`, `chronos2`, `moirai-1.1`)
    - `PARAM freq`, `PARAM horizon`, `PARAM context_length`
    - `PARAM scaling` (`standard`/`minmax`/`none`)
    - `PARAM patch_size` / `binning` 관련 옵션(모델별)
    - `PARAM quantiles` 기본값
    - `PARAM covariates_mode` (`best_effort`/`strict`)
- [~] (P1) `tollama create`/`tollama show`에서 TSModelfile 기반 모델 등록/조회 지원
  - 현재: `tollama show`, `/api/show`는 구현되어 있으나 Modelfile은 빈 문자열로 반환되고 `create`는 없음
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
  - TODO: 공통 계층으로 완전 통합, Parquet 입력까지 일원화
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
  - 현재 Output: `mean`, `quantiles`, `warnings`, `usage`, optional `metrics`(`mape`/`mase`) 구현
  - TODO Meta 확장: `execution_time`, `model_version/digest`, `device` 표준 필드화, `median` 명시 규격 보강
- [~] (P1) 신뢰도/품질 메타 지표 설계
  - 현재: 요청 `series.actuals` + `parameters.metrics` 기반 `mape`/`mase` 계산 및 응답 `metrics` 제공
  - TODO: `mae/rmse/smape/wape/rmsse`, quantile 기반 pinball loss, interval coverage 확장
- [ ] (P1) 포맷 지원 확대: JSON + Parquet (대용량/배치)

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
- [~] (P1) MCP 서버(Claude Code 네이티브 연동) 1차 도입
  - 현재: 공통 HTTP client `src/tollama/client/` 신설(CLI/MCP 공용)
  - 현재: `src/tollama/mcp/` 서버/툴 핸들러/엔트리포인트 추가
  - 현재: `tollama-mcp` 스크립트 및 optional extra `.[mcp]` 추가
  - 현재: `scripts/install_mcp.sh`, `CLAUDE.md` 추가
  - TODO: 실 SDK 환경 E2E smoke + 운영 가이드 보강
- [ ] (P1) TSModelfile 스펙 초안 작성 + parser 구현 계획
  - 파일 포맷/키 목록/우선순위 규칙 정의
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

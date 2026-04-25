# Dashboard User Guide (Web GUI + TUI + macOS App)

This guide explains how to use Tollama's dashboard interfaces:

- Web GUI dashboard (`/dashboard` in browser)
- Terminal TUI dashboard (`tollama dashboard`)
- macOS app shell (`Tollama.app`)

## 1) Prerequisites

- Start daemon first:

```bash
tollama serve
```

- Default daemon URL is `http://127.0.0.1:11435`.
- If API key auth is enabled, prepare a valid key.

## 2) Web GUI Dashboard

### 2.1 Launch

Use either method:

```bash
# Open in default browser
tollama open
```

or open directly:

- `http://127.0.0.1:11435/dashboard`

### 2.2 Main Tabs

- `Overview`
  - Installed models table (`/api/tags`)
  - Loaded models table (`/api/ps`)
  - Live events (`/api/events` SSE)
  - Usage summary (`/api/usage`)
  - Connection status dot in header
- `Forecast`
  - Model select (installed-model list) + optional custom model input
  - Horizon + Series JSON input
  - Forecast response JSON view
  - Forecast chart rendering
  - Export actions: CSV / JSON / PNG / copy JSON
- `Models`
  - Model detail lookup (`/api/show`)
  - Pull model (`/api/pull`, streaming output)
  - License acceptance checkbox for restricted-license pulls
  - Delete model (`/api/delete`)
- `Compare`
  - Multi-model compare form (`/api/compare`)
  - Overlay chart + response output
  - Partial-success summary (`N succeeded, M failed`)
- `Help`
  - Keyboard shortcut summary
  - API docs link

### 2.3 Keyboard Shortcuts (Web)

- `f`: Forecast tab
- `m`: Models tab
- `?`: Help tab

Shortcuts are ignored while typing in input/textarea/select fields.

### 2.4 Auth Behavior

- If dashboard API returns `401`, login dialog appears.
- API key is saved to `sessionStorage` only (tab session scope).

### 2.5 PWA Install

- In supported browsers, an install banner appears.
- You can install dashboard as a desktop app.

### 2.6 macOS App Shell

The GitHub Releases DMG bundles `Tollama.app`, which starts or attaches to a
local daemon and presents a native SwiftUI forecast workspace.

The app shell adds:

- first-launch runtime bootstrap
- child-process daemon lifecycle management
- built-in actions for demo forecast, starter-model install, log access, repair, and reset
- native model browsing/install, local CSV folder preview, forecast charts, and
  inline daemon log-tail visibility

Native app tabs:

- `Models`: curated registry-backed model list with install/remove actions and
  streamed pull progress.
- `Data`: local folder picker, visible `.csv` scan, selected-file preview, and
  inferred schema badges.
- `Forecast`: installed-model picker, horizon/quantile controls, native chart,
  and forecast table.
- `Logs`: daemon log tail.

See `docs/macos-app.md` for the packaging and first-launch details.

## 3) TUI Dashboard

### 3.1 Install TUI Dependency

```bash
python -m pip install -e ".[tui]"
```

### 3.2 Launch

```bash
tollama dashboard
```

Custom daemon URL or timeout:

```bash
tollama dashboard --base-url http://127.0.0.1:11435 --timeout 10
```

If API key is enabled, set:

```bash
export TOLLAMA_API_KEY="your-key"
```

### 3.3 TUI Screens and Actions

- `Dashboard` (default screen)
  - Installed models DataTable (snapshot polling)
  - Loaded models DataTable (snapshot polling)
  - Quick stats (`info` + usage summary + warnings)
  - Live event log (SSE stream with retry)
- `Forecast` screen
  - Model / horizon / series(JSON) input
  - Run forecast
  - Render ASCII chart
  - Export JSON (`tollama_forecast.json`) and CSV (`tollama_forecast.csv`) to current working directory
- `Model detail` screen
  - Show model metadata
  - Pull model events
  - Pull + accept-license action for restricted models
  - Delete model

### 3.4 Keyboard Shortcuts (TUI)

- `q`: Quit app
- `f`: Open Forecast screen
- `Enter`: Open Model detail screen (from dashboard overview)
- `Esc`: Back to previous screen
- `Tab`: Focus next control

## 4) Forecast Input Tips

### 4.1 Minimal JSON example

```json
[{"id":"s1","freq":"D","timestamps":["2025-01-01","2025-01-02"],"target":[10,11]}]
```

### 4.2 Short histories

Some model families use a fixed context window. Granite TTM pads shorter
histories with the earliest observed values and returns a warning; for best
quality, provide as much real history as the selected model's context window
allows.

## 5) Troubleshooting

- Web dashboard not opening:
  - verify daemon is running on the expected `--base-url`
  - open `http://127.0.0.1:11435/dashboard` directly
- TUI `ReadTimeout`:
  - upgrade to latest repo version (`python -m pip install -e ".[tui]"`)
  - confirm daemon URL is reachable
  - increase `--timeout` if network is slow
- Auth failures (`401`):
  - verify `TOLLAMA_API_KEY` or entered key is valid
- Model actions failing:
  - verify model name
  - check daemon logs for detailed backend errors

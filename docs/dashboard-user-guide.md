# Dashboard User Guide (Web GUI + TUI)

This guide explains how to use Tollama's two dashboard interfaces:

- Web GUI dashboard (`/dashboard` in browser)
- Terminal TUI dashboard (`tollama dashboard`)

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
  - Delete model (`/api/delete`)
- `Compare`
  - Multi-model compare form (`/api/compare`)
  - Overlay chart + response output
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
  - Installed models (snapshot polling)
  - Loaded models (snapshot polling)
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

### 4.2 Common error: context length

If you see an error like:

- `input series length is shorter than model context_length (...)`

then:

- provide a longer `target` history
- or switch to a demo-friendly model such as `mock`

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


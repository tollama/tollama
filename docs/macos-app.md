# macOS App Guide

If you want a more user-friendly Tollama experience on Apple Silicon Macs, the
macOS app gives you a native SwiftUI workspace that starts and manages a local
daemon for you — no terminal commands required.

This guide is split into two parts:

- **Part 1 — Using the app**: install, first launch, the four tabs, built-in
  actions, troubleshooting.
- **Part 2 — Building releases**: build inputs, signing, and artifact layout
  for release engineers.

---

## Part 1 — Using the app

## 1) Prerequisites

- Apple Silicon Mac (arm64). Intel Macs are not supported by the release
  artifacts.
- ~1 GB free disk space under your home directory for the bundled runtime and
  starter model.
- No prior Tollama install required. If you already run a daemon on
  `127.0.0.1:11435`, see [§6 Attached-daemon mode](#6-attached-daemon-mode).

## 2) Install

Two artifacts are published in each GitHub Release. Both are signed with a
Developer ID and notarized by Apple.

- `Tollama-<version>-arm64.dmg` — drag-to-install disk image. Open it and drag
  `Tollama.app` to `Applications`. Recommended for most users.
- `Tollama-<version>-arm64.pkg` — admin-style installer that copies
  `Tollama.app` into `/Applications`. Useful for install-flow testing or
  managed deployments.

## 3) First launch — what to expect

The first time you open `Tollama.app`:

1. The app prepares a private Python runtime under
   `~/Library/Application Support/Tollama/`. This takes about 30 seconds and
   does not download anything from the internet — everything is bundled inside
   the app.
2. The app starts a local daemon listening on `127.0.0.1:11435`.
3. The native forecast workspace opens with four tabs.

Subsequent launches reuse the prepared runtime and existing app-local state,
so they start in a couple of seconds.

The app bundles the Tollama core plus the `runner-sundial` extra, so the
built-in starter flow can forecast immediately after pulling the starter
model. Other model families and checkpoints are downloaded on demand from the
`Models` tab.

### 3.1 Under the hood

For diagnostic and packaging context, the bootstrap actually does this:

- Unpacks a private Python 3.11 runtime into
  `~/Library/Application Support/Tollama/runtime`.
- Creates a local venv inside that runtime and installs the bundled Tollama
  package spec from a wheelhouse shipped inside the app.
- Starts `tollamad` as a child process with:
  - `TOLLAMA_HOME=~/Library/Application Support/Tollama/state`
  - `TOLLAMA_HOST=127.0.0.1:11435`
  - `TOLLAMA_RUNTIME_WHEELHOUSE=<bundled wheelhouse path>` for
    wheelhouse-backed family-runtime bootstrap.

Most users can skip this section.

## 4) The four tabs

### 4.1 Models

- Lists the curated registry-backed TSFM inventory, grouped by family.
- Shows Hugging Face source metadata and license notices for each model.
- Install and remove actions stream `/api/pull` progress inline.

### 4.2 Data

The `Data` tab picks a CSV from a local folder and previews it before you
forecast.

#### File picker

- Choose a local folder; the app scans up to 500 visible `.csv` files without
  counting every row up front, so listings stay snappy on large folders.
- Selecting a file shows a bounded fixed-cell preview grid plus inferred
  `timestamp`, `target`, `series`, and `frequency` columns.
- Wide datasets such as `electricity.csv` show a bounded subset of columns in
  the preview grid.

#### Recognized column names

The CSV sniffer auto-detects common column aliases:

- Timestamp: `date`, `Date`, `datetime`, `observation_date`
- Series ID: `series`, `series_id`
- Target: `OT`, `demand`, `users`, `pm2.5`
- OPSD-style: `*_load_actual_entsoe_transparency`

#### Frequency override

When the timestamp cadence cannot be inferred, the Data tab can send an
explicit pandas frequency alias such as `D` (daily), `h` (hourly), or `min`
(minute).

When raw timestamp cadence is regular — or a large file is mostly regular
despite some gaps in the target column — the preview pre-fills that frequency
override for you.

#### Encoding and delimiter support

- Preview text decodes UTF-8, Windows-1252, and Latin-1 — covers most
  weather-dataset CSV headers.
- Upload ingest accepts semicolon- and tab-delimited CSVs.
- Common metadata-preamble files such as World Bank yearly indicator exports
  are accepted.

#### Missing values

The `Missing Values` control is **off by default**. When you turn it on, the
app sends an opt-in interpolation setting for null target or cadence-gap
repair:

- `auto`
- `B-spline`
- `linear`
- `seasonal`

Selecting a different CSV resets this control back to `Off`.

#### Duplicate timestamps

If duplicate timestamps appear within a resolved series, ingest collapses them
by averaging non-null target values and reports a warning, instead of failing
the forecast.

### 4.3 Forecast

- Runs one selected CSV against one installed forecast-ready model via
  `/v1/forecast` using a local `data_url` — this avoids multipart request-size
  limits, so large local files like `electricity.csv` work.
- Clamps the horizon to registry metadata when available.
- For single-series models, down-selects CSVs that have multiple inferred
  series and surfaces a response warning.
- Renders native Swift Charts output plus a forecast table.
- Manifest-only registry entries such as `TimeMixer` and `ForecastPFN` are
  visible in the `Models` tab but omitted from this picker until
  runner-consumable weights or installable upstream packages are wired. The
  daemon also rejects direct forecast calls for those entries before runner
  startup.

### 4.4 Logs

- Shows the live daemon log tail in the main pane.

## 5) Built-in actions

The app shell exposes:

- **Try Demo Forecast** — installs the `mock` model locally and runs a
  network-free demo forecast. Good for verifying the install without pulling a
  real model.
- **Install Starter Model** — pulls `sundial-base-128m`, the default starter.
- **Open Logs** — reveals `~/Library/Logs/Tollama/daemon.log` in Finder.
- **Repair Runtime** — rebuilds the bundled runtime environment and refreshes
  stale family runner environments on next use.
- **Reset Local Data** — clears app-local runtime, state, and logs, then
  bootstraps from scratch on next launch. Destructive: your installed models
  will be re-downloaded.

## 6) Attached-daemon mode

If another Tollama daemon is already listening on `127.0.0.1:11435` when you
launch the app, the app attaches to that daemon instead of starting its own
child process.

In attached mode, destructive maintenance actions (`Repair Runtime`,
`Reset Local Data`) are disabled — the app will not modify state owned by an
external daemon.

## 7) Troubleshooting

### First launch hangs or fails to bootstrap

**Symptoms:** the app is stuck on the bootstrap screen for more than a couple
of minutes, or shows a runtime error before the workspace opens.

**Fix:**

- Use the `Repair Runtime` built-in action to rebuild the bundled environment.
- If it still fails, check `~/Library/Logs/Tollama/daemon.log` and see
  [Troubleshooting §6 Runtime install fails](troubleshooting.md).

### Forecast picker is empty

**Symptoms:** the `Forecast` tab does not list any models.

**Fix:** open the `Models` tab and install at least one forecast-ready model
— e.g. via `Install Starter Model`.

### A model shows in `Models` but not in the forecast picker

**Symptoms:** `TimeMixer` or `ForecastPFN` is visible in `Models` but cannot
be selected in `Forecast`.

**Fix:** these are manifest-only registry entries today. Use
`sundial-base-128m` or `timer-base` for downloadable zero-shot forecasts. See
the model table in the [README](../README.md#supported-models) for the full
list.

### Maintenance actions are disabled

**Symptoms:** `Repair Runtime` or `Reset Local Data` are greyed out.

**Fix:** the app is in attached-daemon mode (see
[§6](#6-attached-daemon-mode)). Stop the external daemon if you want the app
to manage runtime state itself.

### Where to look when something goes wrong

- App-managed daemon logs: `~/Library/Logs/Tollama/daemon.log`
- App-local runtime and state:
  `~/Library/Application Support/Tollama/runtime` and
  `~/Library/Application Support/Tollama/state`

For broader runtime, daemon, and CLI issues see
[Troubleshooting](troubleshooting.md).

---

## Part 2 — Building releases

This section is for release engineers building the macOS artifacts from
source.

## 8) Build inputs

The macOS builders live under `packaging/macos/` and expect a relocatable
Python archive URL:

```bash
export TOLLAMA_PYTHON_STANDALONE_URL="https://..."
```

Optional integrity and release-signing inputs:

```bash
export TOLLAMA_PYTHON_STANDALONE_SHA256="..."
export TOLLAMA_MACOS_BUNDLED_EXTRAS="preprocess,runner-sundial"
export MACOS_SIGNING_IDENTITY="Developer ID Application: Example, Inc. (TEAMID)"
export MACOS_INSTALLER_SIGNING_IDENTITY="Developer ID Installer: Example, Inc. (TEAMID)"
export APPLE_ID="release-bot@example.com"
export APPLE_APP_SPECIFIC_PASSWORD="xxxx-xxxx-xxxx-xxxx"
export APPLE_TEAM_ID="TEAMID"
```

Notes:

- When overriding `TOLLAMA_STARTER_MODEL`, keep `TOLLAMA_MACOS_BUNDLED_EXTRAS`
  in sync with that model's runner family if you want the starter flow to stay
  forecast-ready in the bundled app.
- The asset builder accepts underscore extra names but writes the runtime
  manifest with pip's canonical hyphen spelling, and verifies the bundled
  wheelhouse using the same Python runtime that ships in the app.
- `eval` is no longer bundled by default. Add it explicitly only if your
  release artifact needs `tollama-eval` inside the app-local runtime.

## 9) Build commands

From the repository root:

```bash
packaging/macos/build_dmg.sh
packaging/macos/build_pkg.sh
```

To generate both artifacts from one app-bundle build:

```bash
packaging/macos/build_release_artifacts.sh
```

## 10) Artifacts

Output is written to `dist/macos/`:

- `Tollama-<version>-arm64.dmg`
- `Tollama-<version>-arm64.pkg`
- `SHA256SUMS.txt`

---

## Related documentation

- [Dashboard User Guide](dashboard-user-guide.md) — the dashboard surfaces
  embedded in the app
- [Troubleshooting](troubleshooting.md) — runtime, daemon, CLI, and MCP issues
- [README](../README.md) — Tollama overview, CLI Quickstart, model table

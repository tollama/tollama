# macOS App Guide

This guide covers the user-friendly macOS app distribution for Tollama.

## What ships in the macOS release artifacts

The main consumer-facing release artifact is a signed and notarized Apple
Silicon DMG that contains a single `Tollama.app` bundle.

For install-flow testing and admin-style deployment, Tollama also supports an
installable PKG that copies `Tollama.app` into `/Applications`.

The app includes:

- a native SwiftUI forecast workspace with `Models`, `Data`, `Forecast`, and
  `Logs` tabs
- a first-launch bootstrap flow for a private Python runtime and bundled wheelhouse

The app bundles the Tollama core plus the default starter-model runner extra
(`runner-sundial`) so the built-in starter flow can forecast immediately after
pull. Other heavy runner extras and model checkpoints remain on-demand.

## First launch behavior

On the first launch, the app:

1. unpacks a private Python 3.11 runtime into `~/Library/Application Support/Tollama/runtime`
2. creates a local venv and installs the bundled Tollama package spec from the wheelhouse
3. starts `tollamad` as a child process with:
   - `TOLLAMA_HOME=~/Library/Application Support/Tollama/state`
   - `TOLLAMA_HOST=127.0.0.1:11435`
   - `TOLLAMA_RUNTIME_WHEELHOUSE=<bundled wheelhouse path>` for wheelhouse-backed family-runtime bootstrap
4. opens the native forecast workspace

Subsequent launches reuse the prepared runtime and existing app-local state.

## Native forecast workspace

The detail pane exposes four native tabs:

- `Models`: lists the curated registry-backed TSFM inventory, grouped by
  family, with Hugging Face source metadata, license notices, install/remove
  actions, and streamed `/api/pull` progress.
- `Data`: chooses a local folder, scans up to 500 visible `.csv` files, previews
  the selected file, and shows inferred timestamp/target/series/frequency
  columns before forecasting. The CSV sniffer recognizes common aliases such as
  `date`, `Date`, `datetime`, `observation_date`, `series`, `series_id`, `OT`,
  `demand`, `users`, `pm2.5`, and OPSD-style
  `*_load_actual_entsoe_transparency` columns. When timestamp cadence cannot be
  inferred, the Data tab can send an explicit pandas frequency alias such as
  `D`, `h`, or `min`.
- `Forecast`: runs one selected CSV against one installed forecast-ready model via
  `/api/forecast/upload`, clamps the horizon to registry metadata when
  available, down-selects CSVs with multiple inferred series for single-series
  models with a response warning, and renders native Swift Charts output plus a
  forecast table. Manifest-only registry entries such as TimeMixer and
  ForecastPFN are shown in `Models`, but are omitted from this forecast picker
  until runner-consumable weights or installable upstream packages are wired.
- `Logs`: shows the daemon log tail in the main pane.

## Built-in app actions

The app shell exposes:

- `Try Demo Forecast`: installs `mock` locally and runs a network-free demo forecast
- `Install Starter Model`: pulls the default starter model (`sundial-base-128m`)
- `Open Logs`: reveals `~/Library/Logs/Tollama/daemon.log`
- `Repair Runtime`: rebuilds the bundled runtime environment
- `Reset Local Data`: clears app-local runtime/state/logs and bootstraps again

If another Tollama daemon is already listening on `127.0.0.1:11435`, the app
attaches to that daemon instead of starting its own child process. In that
attached mode, destructive maintenance actions are disabled.

## Build inputs

The macOS builders live under `packaging/macos/` and expect a relocatable
Python archive URL via:

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

When overriding `TOLLAMA_STARTER_MODEL`, keep `TOLLAMA_MACOS_BUNDLED_EXTRAS`
in sync with that model's runner family if you want the starter flow to stay
forecast-ready in the bundled app.
The asset builder accepts underscore extra names but writes the runtime manifest
with pip's canonical hyphen spelling and verifies the bundled wheelhouse using
the same Python runtime that ships in the app.

`eval` is no longer bundled by default; add it explicitly only if your release
artifact needs `tollama-eval` inside the app-local runtime.

Build locally from the repository root:

```bash
packaging/macos/build_dmg.sh
packaging/macos/build_pkg.sh
```

To generate both artifacts from one app-bundle build:

```bash
packaging/macos/build_release_artifacts.sh
```

Artifacts are written to `dist/macos/`:

- `Tollama-<version>-arm64.dmg`
- `Tollama-<version>-arm64.pkg`
- `SHA256SUMS.txt`

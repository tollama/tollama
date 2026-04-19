# macOS App Guide

This guide covers the user-friendly macOS app distribution for Tollama.

## What ships in the macOS release artifacts

The main consumer-facing release artifact is a signed and notarized Apple
Silicon DMG that contains a single `Tollama.app` bundle.

For install-flow testing and admin-style deployment, Tollama also supports an
installable PKG that copies `Tollama.app` into `/Applications`.

The app includes:

- a thin native shell built with SwiftUI + `WKWebView`
- the bundled dashboard UI loaded from `http://127.0.0.1:11435/dashboard`
- a first-launch bootstrap flow for a private Python runtime and bundled wheelhouse

The app bundles the Tollama core plus the default starter-model runner extra
(`runner_sundial`) so the built-in starter flow can forecast immediately after
pull. Other heavy runner extras and model checkpoints remain on-demand.

## First launch behavior

On the first launch, the app:

1. unpacks a private Python 3.11 runtime into `~/Library/Application Support/Tollama/runtime`
2. creates a local venv and installs the bundled Tollama package spec from the wheelhouse
3. starts `tollamad` as a child process with:
   - `TOLLAMA_HOME=~/Library/Application Support/Tollama/state`
   - `TOLLAMA_HOST=127.0.0.1:11435`
   - `TOLLAMA_RUNTIME_WHEELHOUSE=<bundled wheelhouse path>` for wheelhouse-backed family-runtime bootstrap
4. opens the embedded dashboard

Subsequent launches reuse the prepared runtime and existing app-local state.

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
export TOLLAMA_MACOS_BUNDLED_EXTRAS="preprocess,runner_sundial"
export MACOS_SIGNING_IDENTITY="Developer ID Application: Example, Inc. (TEAMID)"
export MACOS_INSTALLER_SIGNING_IDENTITY="Developer ID Installer: Example, Inc. (TEAMID)"
export APPLE_ID="release-bot@example.com"
export APPLE_APP_SPECIFIC_PASSWORD="xxxx-xxxx-xxxx-xxxx"
export APPLE_TEAM_ID="TEAMID"
```

When overriding `TOLLAMA_STARTER_MODEL`, keep `TOLLAMA_MACOS_BUNDLED_EXTRAS`
in sync with that model's runner family if you want the starter flow to stay
forecast-ready in the bundled app.

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

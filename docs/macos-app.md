# macOS App Guide

This guide covers the user-friendly macOS app distribution for Tollama.

## What ships in the DMG

The macOS release artifact is a signed and notarized Apple Silicon DMG that
contains a single `Tollama.app` bundle.

The app includes:

- a thin native shell built with SwiftUI + `WKWebView`
- the bundled dashboard UI loaded from `http://127.0.0.1:11435/dashboard`
- a first-launch bootstrap flow for a private Python runtime and bundled wheelhouse

The app does **not** bundle heavy runner extras or model checkpoints. Those
remain on-demand and install when the user pulls a model.

## First launch behavior

On the first launch, the app:

1. unpacks a private Python 3.11 runtime into `~/Library/Application Support/Tollama/runtime`
2. creates a local venv and installs `tollama[preprocess,eval]` from the bundled wheelhouse
3. starts `tollamad` as a child process with:
   - `TOLLAMA_HOME=~/Library/Application Support/Tollama/state`
   - `TOLLAMA_HOST=127.0.0.1:11435`
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

The DMG builder lives under `packaging/macos/` and expects a relocatable Python
archive URL via:

```bash
export TOLLAMA_PYTHON_STANDALONE_URL="https://..."
```

Optional integrity and release-signing inputs:

```bash
export TOLLAMA_PYTHON_STANDALONE_SHA256="..."
export MACOS_SIGNING_IDENTITY="Developer ID Application: Example, Inc. (TEAMID)"
export APPLE_ID="release-bot@example.com"
export APPLE_APP_SPECIFIC_PASSWORD="xxxx-xxxx-xxxx-xxxx"
export APPLE_TEAM_ID="TEAMID"
```

Build locally from the repository root:

```bash
packaging/macos/build_dmg.sh
```

Artifacts are written to `dist/macos/`.

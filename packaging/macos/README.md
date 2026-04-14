# macOS App Packaging

This directory contains the user-friendly macOS app distribution scaffold for
Tollama.

The primary release artifact is a signed and notarized Apple Silicon DMG that
contains a single `Tollama.app` bundle. The app bootstraps a private Python
runtime on first launch, starts `tollamad` as a child process, and renders the
bundled `/dashboard` UI inside a native `WKWebView` shell.

## Layout

- `build_dmg.sh`: end-to-end macOS build script
- `prepare_runtime_assets.sh`: prepares the bundled Python runtime archive and wheelhouse
- `TollamaApp/`: SwiftUI + `WKWebView` thin native shell sources

## Required Build Inputs

The build expects a relocatable Python 3.11 archive URL. Set:

```bash
export TOLLAMA_PYTHON_STANDALONE_URL="https://..."
```

Optional integrity input:

```bash
export TOLLAMA_PYTHON_STANDALONE_SHA256="..."
```

## Signing / Notarization Inputs

For signed + notarized release builds, set:

```bash
export MACOS_SIGNING_IDENTITY="Developer ID Application: Example, Inc. (TEAMID)"
export APPLE_ID="release-bot@example.com"
export APPLE_APP_SPECIFIC_PASSWORD="xxxx-xxxx-xxxx-xxxx"
export APPLE_TEAM_ID="TEAMID"
```

If these are omitted, the script still builds an unsigned local DMG for smoke
testing.

## Local Build

From the repository root:

```bash
packaging/macos/build_dmg.sh
```

Artifacts are written to `dist/macos/`.

## Runtime Behavior

At runtime, `Tollama.app`:

1. unpacks bundled runtime assets into `~/Library/Application Support/Tollama/runtime`
2. creates a local venv and installs `tollama[preprocess,eval]` from the bundled wheelhouse
3. starts `tollamad` with `TOLLAMA_HOME=~/Library/Application Support/Tollama/state`
4. loads `http://127.0.0.1:11435/dashboard` in an embedded `WKWebView`

Heavy runner extras and model checkpoints remain on-demand and are not bundled
in the DMG.

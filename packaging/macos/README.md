# macOS App Packaging

This directory contains the user-friendly macOS app distribution scaffold for
Tollama.

The primary user-facing release artifact is a signed and notarized Apple Silicon
DMG that contains a single `Tollama.app` bundle. For install-flow testing and
admin-style deployment, the same app bundle can also be wrapped as an installable
PKG.

The app bootstraps a private Python runtime on first launch, starts `tollamad`
as a child process, and exposes native SwiftUI tabs for model management, CSV
preview, forecasting, and logs.

## Layout

- `build_dmg.sh`: end-to-end macOS build script
- `build_pkg.sh`: builds an installable macOS PKG that installs `Tollama.app` into `/Applications`
- `build_release_artifacts.sh`: builds both DMG and PKG from one app-bundle build
- `prepare_runtime_assets.sh`: prepares the bundled Python runtime archive and wheelhouse
- `TollamaApp/`: SwiftUI native app shell sources

## Required Build Inputs

The build expects a relocatable Python 3.11 archive URL. Set:

```bash
export TOLLAMA_PYTHON_STANDALONE_URL="https://..."
```

Optional integrity input:

```bash
export TOLLAMA_PYTHON_STANDALONE_SHA256="..."
```

Optional bundle contents override for local/test builds:

```bash
export TOLLAMA_MACOS_BUNDLED_EXTRAS="preprocess,runner-sundial"
```

By default the build also bundles the current starter-model runner extra so the
starter flow is forecast-ready inside `Tollama.app`.
Add `eval` explicitly only when you also need `tollama-eval` available inside
the bundled app runtime.
The asset builder normalizes underscore extras to pip's canonical hyphen form
and verifies the wheelhouse with the same bundled Python runtime that the app
uses at first launch.

## Signing / Notarization Inputs

For signed + notarized release builds, set:

```bash
export MACOS_SIGNING_IDENTITY="Developer ID Application: Example, Inc. (TEAMID)"
export MACOS_INSTALLER_SIGNING_IDENTITY="Developer ID Installer: Example, Inc. (TEAMID)"
export APPLE_ID="release-bot@example.com"
export APPLE_APP_SPECIFIC_PASSWORD="xxxx-xxxx-xxxx-xxxx"
export APPLE_TEAM_ID="TEAMID"
```

If these are omitted, the scripts still build unsigned local artifacts for smoke
testing.

## Local Build

From the repository root:

```bash
packaging/macos/build_dmg.sh
packaging/macos/build_pkg.sh
```

To build both in one pass:

```bash
packaging/macos/build_release_artifacts.sh
```

Artifacts are written to `dist/macos/`.

## Runtime Behavior

At runtime, `Tollama.app`:

1. unpacks bundled runtime assets into `~/Library/Application Support/Tollama/runtime`
2. creates a local venv and installs the bundled Tollama package spec from the wheelhouse
3. starts `tollamad` with `TOLLAMA_HOME=~/Library/Application Support/Tollama/state`
4. loads `http://127.0.0.1:11435/dashboard` in an embedded `WKWebView`

Most heavy runner extras and model checkpoints remain on-demand and are not
bundled in the DMG. The default starter-model runner extra is bundled so the
starter flow can forecast immediately after pull.

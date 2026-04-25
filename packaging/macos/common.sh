#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${TOLLAMA_MACOS_BUILD_DIR:-$ROOT_DIR/.build/macos}"
DIST_DIR="${TOLLAMA_MACOS_DIST_DIR:-$ROOT_DIR/dist/macos}"
APP_NAME="Tollama"
APP_BUNDLE="$BUILD_DIR/$APP_NAME.app"
DMG_STAGING_DIR="$BUILD_DIR/dmg-staging"
SWIFT_SOURCES_DIR="$ROOT_DIR/packaging/macos/TollamaApp/Sources"
RESOURCE_TEMPLATE_DIR="$ROOT_DIR/packaging/macos/TollamaApp/Resources"
ASSET_DIR="$BUILD_DIR/runtime-assets"
MODULE_CACHE_DIR="$BUILD_DIR/module-cache"
TARGET_ARCH="${TOLLAMA_MACOS_ARCH:-arm64}"
MACOS_SIGNING_IDENTITY="${MACOS_SIGNING_IDENTITY:-}"
MACOS_INSTALLER_SIGNING_IDENTITY="${MACOS_INSTALLER_SIGNING_IDENTITY:-}"
APPLE_ID="${APPLE_ID:-}"
APPLE_APP_SPECIFIC_PASSWORD="${APPLE_APP_SPECIFIC_PASSWORD:-}"
APPLE_TEAM_ID="${APPLE_TEAM_ID:-}"
TOLLAMA_VERSION="${TOLLAMA_VERSION:-}"


resolve_tollama_version() {
  if [[ -n "$TOLLAMA_VERSION" ]]; then
    return
  fi

  TOLLAMA_VERSION="$(
    cd "$ROOT_DIR"
    python - <<'PY'
from pathlib import Path
import re

text = Path("pyproject.toml").read_text(encoding="utf-8")
match = re.search(r'^version = "([^"]+)"', text, re.MULTILINE)
if match is None:
    raise SystemExit("Unable to determine tollama version from pyproject.toml")
print(match.group(1))
PY
  )"
}


prepare_common_dirs() {
  resolve_tollama_version
  mkdir -p "$BUILD_DIR" "$DIST_DIR" "$MODULE_CACHE_DIR"
}


has_notary_credentials() {
  [[ -n "$APPLE_ID" && -n "$APPLE_APP_SPECIFIC_PASSWORD" && -n "$APPLE_TEAM_ID" ]]
}


build_app_bundle() {
  prepare_common_dirs

  "$ROOT_DIR/packaging/macos/prepare_runtime_assets.sh"

  rm -rf "$APP_BUNDLE"
  mkdir -p \
    "$APP_BUNDLE/Contents/MacOS" \
    "$APP_BUNDLE/Contents/Resources/RuntimeAssets"

  sed \
    -e "s/__APP_NAME__/$APP_NAME/g" \
    -e "s/__VERSION__/$TOLLAMA_VERSION/g" \
    "$RESOURCE_TEMPLATE_DIR/Info.plist.template" \
    > "$APP_BUNDLE/Contents/Info.plist"

  ditto "$ASSET_DIR" "$APP_BUNDLE/Contents/Resources/RuntimeAssets"

  local swift_sources=()
  while IFS= read -r -d '' source_path; do
    swift_sources+=("$source_path")
  done < <(find "$SWIFT_SOURCES_DIR" -name '*.swift' -print0 | sort -z)

  echo "Compiling SwiftUI app bundle..."
  xcrun swiftc \
    -target "$TARGET_ARCH-apple-macos13" \
    -module-cache-path "$MODULE_CACHE_DIR" \
    -parse-as-library \
    "${swift_sources[@]}" \
    -framework AppKit \
    -framework Charts \
    -framework SwiftUI \
    -framework WebKit \
    -o "$APP_BUNDLE/Contents/MacOS/$APP_NAME"

  chmod +x "$APP_BUNDLE/Contents/MacOS/$APP_NAME"

  if [[ -n "$MACOS_SIGNING_IDENTITY" ]]; then
    echo "Signing app bundle..."
    codesign \
      --force \
      --deep \
      --timestamp \
      --options runtime \
      --sign "$MACOS_SIGNING_IDENTITY" \
      "$APP_BUNDLE"
  fi

  verify_app_bundle_signature
}


verify_app_bundle_signature() {
  if [[ -n "$MACOS_SIGNING_IDENTITY" ]]; then
    echo "Verifying app signature..."
    codesign --verify --deep --strict --verbose=2 "$APP_BUNDLE"
  else
    echo "Skipping app codesign verification for unsigned local build."
  fi
}


create_dmg_artifact() {
  prepare_common_dirs
  local dmg_path="$DIST_DIR/${APP_NAME}-${TOLLAMA_VERSION}-${TARGET_ARCH}.dmg"

  rm -rf "$DMG_STAGING_DIR"
  mkdir -p "$DMG_STAGING_DIR"
  cp -R "$APP_BUNDLE" "$DMG_STAGING_DIR/$APP_NAME.app"
  ln -sfn /Applications "$DMG_STAGING_DIR/Applications"

  rm -f "$dmg_path"
  echo "Creating DMG..."
  hdiutil create \
    -volname "$APP_NAME" \
    -srcfolder "$DMG_STAGING_DIR" \
    -ov \
    -format UDZO \
    "$dmg_path"

  if [[ -n "$MACOS_SIGNING_IDENTITY" ]]; then
    echo "Signing DMG..."
    codesign \
      --force \
      --timestamp \
      --sign "$MACOS_SIGNING_IDENTITY" \
      "$dmg_path"
  fi

  if [[ -n "$MACOS_SIGNING_IDENTITY" ]] && has_notary_credentials; then
    maybe_notarize_and_staple "$dmg_path"
    echo "Stapling notarization ticket to $APP_NAME.app..."
    xcrun stapler staple "$APP_BUNDLE"
  fi

  echo "DMG mount smoke test..."
  local mount_point
  mount_point="$(mktemp -d "/tmp/tollama-dmg.XXXXXX")"
  hdiutil attach "$dmg_path" -mountpoint "$mount_point" -nobrowse -quiet
  test -d "$mount_point/$APP_NAME.app"
  hdiutil detach "$mount_point" -quiet
  rmdir "$mount_point"
}


create_pkg_artifact() {
  prepare_common_dirs
  local pkg_path="$DIST_DIR/${APP_NAME}-${TOLLAMA_VERSION}-${TARGET_ARCH}.pkg"
  local pkgbuild_args=(
    --component "$APP_BUNDLE"
    --install-location "/Applications"
    --identifier "ai.tollama.desktop"
    --version "$TOLLAMA_VERSION"
  )

  if [[ -n "$MACOS_INSTALLER_SIGNING_IDENTITY" ]]; then
    pkgbuild_args+=(--sign "$MACOS_INSTALLER_SIGNING_IDENTITY")
  fi

  rm -f "$pkg_path"
  echo "Creating PKG..."
  pkgbuild "${pkgbuild_args[@]}" "$pkg_path"

  if [[ -n "$MACOS_INSTALLER_SIGNING_IDENTITY" ]]; then
    echo "Verifying installer package signature..."
    pkgutil --check-signature "$pkg_path"
  else
    echo "Skipping pkg signature verification for unsigned local build."
  fi

  if [[ -n "$MACOS_INSTALLER_SIGNING_IDENTITY" ]] && has_notary_credentials; then
    maybe_notarize_and_staple "$pkg_path"
  fi
}


maybe_notarize_and_staple() {
  local artifact_path="$1"

  echo "Submitting $(basename "$artifact_path") for notarization..."
  xcrun notarytool submit \
    "$artifact_path" \
    --apple-id "$APPLE_ID" \
    --password "$APPLE_APP_SPECIFIC_PASSWORD" \
    --team-id "$APPLE_TEAM_ID" \
    --wait

  echo "Stapling notarization ticket to $(basename "$artifact_path")..."
  xcrun stapler staple "$artifact_path"
}


refresh_checksums() {
  (
    cd "$DIST_DIR"
    rm -f SHA256SUMS.txt
    set --
    while IFS= read -r file_path; do
      set -- "$@" "$file_path"
    done < <(find . -maxdepth 1 -type f ! -name 'SHA256SUMS.txt' -print | LC_ALL=C sort)

    if [[ $# -gt 0 ]]; then
      shasum -a 256 "$@" > SHA256SUMS.txt
    fi
  )
}

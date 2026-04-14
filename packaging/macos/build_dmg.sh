#!/usr/bin/env bash
set -euo pipefail

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
APPLE_ID="${APPLE_ID:-}"
APPLE_APP_SPECIFIC_PASSWORD="${APPLE_APP_SPECIFIC_PASSWORD:-}"
APPLE_TEAM_ID="${APPLE_TEAM_ID:-}"

mkdir -p "$BUILD_DIR" "$DIST_DIR"
mkdir -p "$MODULE_CACHE_DIR"

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

"$ROOT_DIR/packaging/macos/prepare_runtime_assets.sh"

rm -rf "$APP_BUNDLE" "$DMG_STAGING_DIR"
mkdir -p \
  "$APP_BUNDLE/Contents/MacOS" \
  "$APP_BUNDLE/Contents/Resources/RuntimeAssets" \
  "$DMG_STAGING_DIR"

sed \
  -e "s/__APP_NAME__/$APP_NAME/g" \
  -e "s/__VERSION__/$TOLLAMA_VERSION/g" \
  "$RESOURCE_TEMPLATE_DIR/Info.plist.template" \
  > "$APP_BUNDLE/Contents/Info.plist"

ditto "$ASSET_DIR" "$APP_BUNDLE/Contents/Resources/RuntimeAssets"

echo "Compiling SwiftUI app bundle..."
xcrun swiftc \
  -target "$TARGET_ARCH-apple-macos13" \
  -module-cache-path "$MODULE_CACHE_DIR" \
  -parse-as-library \
  "$SWIFT_SOURCES_DIR"/*.swift \
  -framework AppKit \
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

cp -R "$APP_BUNDLE" "$DMG_STAGING_DIR/$APP_NAME.app"
ln -s /Applications "$DMG_STAGING_DIR/Applications"

DMG_PATH="$DIST_DIR/${APP_NAME}-${TOLLAMA_VERSION}-${TARGET_ARCH}.dmg"
rm -f "$DMG_PATH"

echo "Creating DMG..."
hdiutil create \
  -volname "$APP_NAME" \
  -srcfolder "$DMG_STAGING_DIR" \
  -ov \
  -format UDZO \
  "$DMG_PATH"

if [[ -n "$MACOS_SIGNING_IDENTITY" ]]; then
  echo "Signing DMG..."
  codesign \
    --force \
    --timestamp \
    --sign "$MACOS_SIGNING_IDENTITY" \
    "$DMG_PATH"
fi

if [[ -n "$MACOS_SIGNING_IDENTITY" ]]; then
  echo "Verifying app signature..."
  codesign --verify --deep --strict --verbose=2 "$APP_BUNDLE"
else
  echo "Skipping codesign verification for unsigned local build."
fi

if [[ -n "$MACOS_SIGNING_IDENTITY" && -n "$APPLE_ID" && -n "$APPLE_APP_SPECIFIC_PASSWORD" && -n "$APPLE_TEAM_ID" ]]; then
  echo "Submitting DMG for notarization..."
  xcrun notarytool submit \
    "$DMG_PATH" \
    --apple-id "$APPLE_ID" \
    --password "$APPLE_APP_SPECIFIC_PASSWORD" \
    --team-id "$APPLE_TEAM_ID" \
    --wait

  echo "Stapling notarization tickets..."
  xcrun stapler staple "$APP_BUNDLE"
  xcrun stapler staple "$DMG_PATH"
fi

echo "DMG mount smoke test..."
MOUNT_POINT="$(mktemp -d "/tmp/tollama-dmg.XXXXXX")"
hdiutil attach "$DMG_PATH" -mountpoint "$MOUNT_POINT" -nobrowse -quiet
test -d "$MOUNT_POINT/$APP_NAME.app"
hdiutil detach "$MOUNT_POINT" -quiet
rmdir "$MOUNT_POINT"

echo "Writing checksums..."
(
  cd "$DIST_DIR"
  shasum -a 256 "$(basename "$DMG_PATH")" > SHA256SUMS.txt
)

echo "Built $DMG_PATH"

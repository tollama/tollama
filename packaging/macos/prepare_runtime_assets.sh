#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${TOLLAMA_MACOS_BUILD_DIR:-$ROOT_DIR/.build/macos}"
ASSET_DIR="$BUILD_DIR/runtime-assets"
DOWNLOAD_DIR="$BUILD_DIR/downloads"
PYTHON_ARCHIVE_URL="${TOLLAMA_PYTHON_STANDALONE_URL:?Set TOLLAMA_PYTHON_STANDALONE_URL to a relocatable Python 3.11 archive URL.}"
PYTHON_ARCHIVE_SHA256="${TOLLAMA_PYTHON_STANDALONE_SHA256:-}"
STARTER_MODEL="${TOLLAMA_STARTER_MODEL:-sundial-base-128m}"
# Bundle the starter-model runner so a first forecast works inside the app
# without forcing extras that are not required for the dashboard starter flow.
# Override TOLLAMA_MACOS_BUNDLED_EXTRAS explicitly for custom build footprints
# or when changing the starter model family. Normalize to PEP 685/PIP's
# canonical extra spelling so the runtime manifest matches wheel metadata.
BUNDLED_EXTRAS="${TOLLAMA_MACOS_BUNDLED_EXTRAS:-preprocess,runner-sundial}"
BUNDLED_EXTRAS="${BUNDLED_EXTRAS//_/-}"

mkdir -p "$ASSET_DIR" "$DOWNLOAD_DIR"
rm -rf "$ASSET_DIR"
mkdir -p "$ASSET_DIR/wheelhouse" "$ASSET_DIR/tmp/extracted" "$ASSET_DIR/tmp/normalized/python"

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

ARCHIVE_PATH="$DOWNLOAD_DIR/python-runtime.tar.gz"
echo "Downloading Python runtime archive..."
curl -L --fail "$PYTHON_ARCHIVE_URL" -o "$ARCHIVE_PATH"

if [[ -n "$PYTHON_ARCHIVE_SHA256" ]]; then
  echo "$PYTHON_ARCHIVE_SHA256  $ARCHIVE_PATH" | shasum -a 256 -c -
fi

echo "Extracting Python runtime archive..."
tar -xzf "$ARCHIVE_PATH" -C "$ASSET_DIR/tmp/extracted"

detect_python_root() {
  local search_root="$1"
  local candidate=""

  while IFS= read -r -d '' candidate; do
    if [[ -x "$candidate/bin/python3" || -x "$candidate/bin/python3.11" || -x "$candidate/bin/python" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
    if [[ -x "$candidate/install/bin/python3" || -x "$candidate/install/bin/python3.11" || -x "$candidate/install/bin/python" ]]; then
      printf '%s\n' "$candidate/install"
      return 0
    fi
  done < <(find "$search_root" -type d -print0)

  return 1
}

detect_python_executable() {
  local root="$1"
  local candidate=""
  local candidates=(
    "$root/bin/python3"
    "$root/bin/python3.11"
    "$root/bin/python"
  )

  for candidate in "${candidates[@]}"; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  return 1
}

PYTHON_ROOT="$(detect_python_root "$ASSET_DIR/tmp/extracted")"
if [[ -z "$PYTHON_ROOT" ]]; then
  echo "Unable to locate extracted Python runtime root" >&2
  exit 1
fi

echo "Normalizing Python runtime layout..."
ditto "$PYTHON_ROOT" "$ASSET_DIR/tmp/normalized/python"
tar -czf "$ASSET_DIR/python-runtime.tar.gz" -C "$ASSET_DIR/tmp/normalized/python" .

BUNDLED_PYTHON="$(detect_python_executable "$ASSET_DIR/tmp/normalized/python")"
WHEELHOUSE_VENV="$ASSET_DIR/tmp/wheelhouse-venv"
VERIFY_VENV="$ASSET_DIR/tmp/verify-venv"

echo "Building wheelhouse..."
cd "$ROOT_DIR"
"$BUNDLED_PYTHON" -m venv "$WHEELHOUSE_VENV"
WHEELHOUSE_PYTHON="$(detect_python_executable "$WHEELHOUSE_VENV")"
"$WHEELHOUSE_PYTHON" -m pip install --upgrade pip setuptools build wheel
"$WHEELHOUSE_PYTHON" -m build --wheel --no-isolation --outdir "$ASSET_DIR/wheelhouse"

if [[ -n "$BUNDLED_EXTRAS" ]]; then
  LOCAL_WHEEL_TARGET="./[$BUNDLED_EXTRAS]"
  BUNDLED_INSTALL_SPEC="tollama[$BUNDLED_EXTRAS]==$TOLLAMA_VERSION"
else
  LOCAL_WHEEL_TARGET="./"
  BUNDLED_INSTALL_SPEC="tollama==$TOLLAMA_VERSION"
fi

"$WHEELHOUSE_PYTHON" -m pip wheel \
  --wheel-dir "$ASSET_DIR/wheelhouse" \
  --no-build-isolation \
  "$LOCAL_WHEEL_TARGET"

echo "Verifying wheelhouse with bundled Python..."
"$BUNDLED_PYTHON" -m venv "$VERIFY_VENV"
VERIFY_PYTHON="$(detect_python_executable "$VERIFY_VENV")"
"$VERIFY_PYTHON" -m pip install \
  --no-index \
  --find-links "$ASSET_DIR/wheelhouse" \
  "$BUNDLED_INSTALL_SPEC"
"$VERIFY_PYTHON" -c 'import tollama.daemon.main'

"$WHEELHOUSE_PYTHON" - <<PY
from pathlib import Path
import json

payload = {
    "install_spec": "$BUNDLED_INSTALL_SPEC",
    "tollama_version": "$TOLLAMA_VERSION",
    "starter_model": "$STARTER_MODEL",
    "python_archive": "python-runtime.tar.gz",
    "wheelhouse_dir": "wheelhouse",
}
Path("$ASSET_DIR/runtime-manifest.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

rm -rf "$ASSET_DIR/tmp"
echo "Runtime assets prepared under $ASSET_DIR"

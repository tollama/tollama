#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

build_app_bundle
create_dmg_artifact
create_pkg_artifact
refresh_checksums

resolve_tollama_version
dmg_path="$DIST_DIR/${APP_NAME}-${TOLLAMA_VERSION}-${TARGET_ARCH}.dmg"
pkg_path="$DIST_DIR/${APP_NAME}-${TOLLAMA_VERSION}-${TARGET_ARCH}.pkg"
echo "Built $dmg_path"
echo "Built $pkg_path"

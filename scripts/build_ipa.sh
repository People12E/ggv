#!/usr/bin/env bash
# Builds B.P into an .ipa. Needs macOS + Xcode 15+ (iOS 17 SDK) and XcodeGen (installed via Homebrew if missing).
#
#   Signed build (device testing / TestFlight / App Store):
#     DEVELOPMENT_TEAM=ABCDE12345 BUNDLE_ID=com.yourname.bp METHOD=development ./scripts/build_ipa.sh
#   Unsigned build (re-sign later with AltStore / Sideloadly / your own tooling):
#     UNSIGNED=1 ./scripts/build_ipa.sh
#
# METHOD: development | ad-hoc | app-store-connect | enterprise      (default: development)
# Output: build/ipa/BP.ipa   (signed)   or   build/BP-unsigned.ipa   (unsigned)
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p build
LOG=build/xcodebuild.log
: > "$LOG"
on_fail() {
  echo "---- build failed; first errors (full log: $LOG) ----"
  # "::error::" makes GitHub show each line as an annotation; it is just text when run locally.
  grep -E "error:|fatal error:|Command .* failed|BUILD FAILED|ARCHIVE FAILED" "$LOG" | sort -u | head -40 | while IFS= read -r l; do echo "::error::${l}"; done
}
trap on_fail ERR

BUNDLE_ID="${BUNDLE_ID:-com.example.bp}"
METHOD="${METHOD:-development}"
UNSIGNED="${UNSIGNED:-0}"

if [[ "$(uname)" != "Darwin" ]]; then echo "error: building an IPA needs macOS with Xcode." >&2; exit 1; fi
command -v xcodebuild >/dev/null || { echo "error: xcodebuild not found; install Xcode." >&2; exit 1; }
if ! command -v xcodegen >/dev/null; then
  command -v brew >/dev/null || { echo "error: install XcodeGen (brew install xcodegen)." >&2; exit 1; }
  brew install xcodegen
fi

# Shader and catalog consistency checks (cheap; catches slot mismatches before a long build).
command -v python3 >/dev/null && python3 tools/check_catalog.py

xcodegen generate 2>&1 | tee -a "$LOG"

if [[ "$UNSIGNED" == "1" ]]; then
  xcodebuild -project BP.xcodeproj -scheme BP -configuration Release -sdk iphoneos \
    -derivedDataPath build/DerivedData \
    PRODUCT_BUNDLE_IDENTIFIER="$BUNDLE_ID" \
    CODE_SIGNING_ALLOWED=NO CODE_SIGNING_REQUIRED=NO CODE_SIGN_IDENTITY="" \
    build 2>&1 | tee -a "$LOG"
  APP="build/DerivedData/Build/Products/Release-iphoneos/BP.app"
  [[ -d "$APP" ]] || { echo "error: $APP not found" >&2; exit 1; }
  rm -rf build/Payload build/BP-unsigned.ipa
  mkdir -p build/Payload
  cp -R "$APP" build/Payload/
  (cd build && zip -qr BP-unsigned.ipa Payload)
  rm -rf build/Payload
  echo "Built build/BP-unsigned.ipa (unsigned: it will not install until it is signed)."
  exit 0
fi

: "${DEVELOPMENT_TEAM:?set DEVELOPMENT_TEAM to your 10-character Apple Developer Team ID (or use UNSIGNED=1)}"

xcodebuild -project BP.xcodeproj -scheme BP -configuration Release \
  -destination 'generic/platform=iOS' -archivePath build/BP.xcarchive \
  DEVELOPMENT_TEAM="$DEVELOPMENT_TEAM" PRODUCT_BUNDLE_IDENTIFIER="$BUNDLE_ID" \
  CODE_SIGN_STYLE=Automatic -allowProvisioningUpdates \
  archive 2>&1 | tee -a "$LOG"

cat > build/ExportOptions.plist <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>method</key><string>${METHOD}</string>
  <key>teamID</key><string>${DEVELOPMENT_TEAM}</string>
  <key>signingStyle</key><string>automatic</string>
  <key>compileBitcode</key><false/>
  <key>stripSwiftSymbols</key><true/>
</dict>
</plist>
PLIST

rm -rf build/ipa
xcodebuild -exportArchive -archivePath build/BP.xcarchive \
  -exportPath build/ipa -exportOptionsPlist build/ExportOptions.plist -allowProvisioningUpdates 2>&1 | tee -a "$LOG"

echo "Built $(ls build/ipa/*.ipa)"

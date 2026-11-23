#!/bin/sh
# Build a release binary and wrap it in a macOS .app bundle.
set -eu
cd "$(dirname "$0")/.."

cargo build --release

APP="target/Talk with Agent.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS"
cp macos/Info.plist "$APP/Contents/Info.plist"
cp target/release/talk-with-agent "$APP/Contents/MacOS/"
# Ad-hoc signature so macOS can remember the microphone/speech permissions.
codesign --force --sign - "$APP"

echo "Built: $APP"
echo "Run:   open \"$APP\""

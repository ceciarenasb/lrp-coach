#!/usr/bin/env bash
# Create an "LRP Coach.app" in /Applications that launches the server and opens the browser.
# Run once from the project root: ./scripts/setup_launcher.sh

set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
APP_NAME="LRP Coach"
APP_BUNDLE="/Applications/$APP_NAME.app"
ICON_PNG="$APP_DIR/scripts/lrp_icon.png"
ICNS_DIR="/tmp/lrp_coach.iconset"
LAUNCH_SCRIPT="$APP_DIR/scripts/launch.sh"

echo "Building $APP_BUNDLE …"

if [ ! -f "$ICON_PNG" ]; then
  echo "Error: icon not found at $ICON_PNG"
  exit 1
fi

# ── 1. Convert PNG → .icns ────────────────────────────────────────────────
make_icns() {
  local src="$1" dest="$2"
  rm -rf "$ICNS_DIR" && mkdir "$ICNS_DIR"
  for size in 16 32 64 128 256 512 1024; do
    sips -z $size $size "$src" --out "$ICNS_DIR/icon_${size}x${size}.png" &>/dev/null
  done
  cp "$ICNS_DIR/icon_32x32.png"   "$ICNS_DIR/icon_16x16@2x.png"
  cp "$ICNS_DIR/icon_64x64.png"   "$ICNS_DIR/icon_32x32@2x.png"
  cp "$ICNS_DIR/icon_256x256.png" "$ICNS_DIR/icon_128x128@2x.png"
  cp "$ICNS_DIR/icon_512x512.png" "$ICNS_DIR/icon_256x256@2x.png"
  cp "$ICNS_DIR/icon_1024x1024.png" "$ICNS_DIR/icon_512x512@2x.png"
  iconutil -c icns "$ICNS_DIR" -o "$dest"
  rm -rf "$ICNS_DIR"
}

ICNS_PATH="/tmp/lrp_coach.icns"
if [ -f "$ICON_PNG" ]; then
  make_icns "$ICON_PNG" "$ICNS_PATH"
  echo "Icon converted to $ICNS_PATH"
fi

# ── 3. Compile AppleScript → .app ─────────────────────────────────────────
APPLE_SCRIPT=$(cat <<ASEOF
on run
  set appDir to "$APP_DIR"
  set launchScript to "$LAUNCH_SCRIPT"
  do shell script "bash " & quoted form of launchScript
end run
ASEOF
)

osacompile -o "$APP_BUNDLE" - <<<"$APPLE_SCRIPT"

# ── 4. Inject custom icon ─────────────────────────────────────────────────
if [ -f "$ICNS_PATH" ]; then
  RESOURCES="$APP_BUNDLE/Contents/Resources"
  cp "$ICNS_PATH" "$RESOURCES/applet.icns"
  # Touch the app so Finder picks up the new icon
  touch "$APP_BUNDLE"
  /System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister \
    -f "$APP_BUNDLE" 2>/dev/null || true
  echo "Custom icon applied."
fi

# ── 5. Make scripts executable ───────────────────────────────────────────
chmod +x "$LAUNCH_SCRIPT" "$APP_DIR/scripts/stop.sh"

echo ""
echo "Done! '$APP_NAME' is now in /Applications."
echo "Double-click to start the server and open your browser."
echo ""
echo "From the terminal you can also use:"
echo "  ./scripts/launch.sh   — start"
echo "  ./scripts/stop.sh     — stop"

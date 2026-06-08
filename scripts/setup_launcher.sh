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

# ── 1. Generate icon PNG via Python ──────────────────────────────────────
"$APP_DIR/.venv/bin/python" - <<'PYEOF'
import os, sys
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    fig, ax = plt.subplots(figsize=(4, 4), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.set_xlim(0, 4); ax.set_ylim(0, 4)
    ax.axis("off")

    # Circular track
    circle = plt.Circle((2, 2), 1.5, fill=False, color="#e94560", linewidth=8, zorder=1)
    ax.add_patch(circle)

    # Running figure (simplified SVG-like silhouette as ellipses)
    body = mpatches.Ellipse((2, 2.2), 0.4, 0.65, angle=10, color="#f0f0f0", zorder=2)
    head = plt.Circle((2.1, 2.7), 0.22, color="#f0f0f0", zorder=2)
    ax.add_patch(body); ax.add_patch(head)

    # Legs
    ax.plot([2, 1.6], [1.9, 1.4], color="#f0f0f0", linewidth=5, solid_capstyle="round", zorder=2)
    ax.plot([2, 2.3], [1.9, 1.35], color="#f0f0f0", linewidth=5, solid_capstyle="round", zorder=2)
    # Arms
    ax.plot([2, 1.55], [2.3, 2.0], color="#f0f0f0", linewidth=4, solid_capstyle="round", zorder=2)
    ax.plot([2, 2.45], [2.3, 2.0], color="#f0f0f0", linewidth=4, solid_capstyle="round", zorder=2)

    # Label
    ax.text(2, 0.35, "LRP Coach", ha="center", va="center",
            fontsize=18, fontweight="bold", color="#e94560", fontfamily="monospace")

    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/lrp_icon.png"
    fig.savefig(out, dpi=128, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Icon written to {out}")
except Exception as e:
    print(f"Warning: could not generate icon: {e}", file=sys.stderr)
PYEOF

# move if saved to /tmp default
if [ ! -f "$ICON_PNG" ] && [ -f "/tmp/lrp_icon.png" ]; then
  mv /tmp/lrp_icon.png "$ICON_PNG"
fi

# ── 2. Convert PNG → .icns ────────────────────────────────────────────────
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

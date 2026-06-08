#!/usr/bin/env bash
# Start LRP Coach and open it in the browser.
# Usage: ./scripts/launch.sh

set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$APP_DIR/.venv"
PORT=7860
PID_FILE="$APP_DIR/.lrp_coach.pid"
LOG_FILE="$APP_DIR/.lrp_coach.log"

# ── Check if already running ──────────────────────────────────────────────
if [ -f "$PID_FILE" ]; then
  PID=$(cat "$PID_FILE")
  if kill -0 "$PID" 2>/dev/null; then
    echo "LRP Coach is already running (PID $PID)"
    open "http://localhost:$PORT"
    exit 0
  else
    rm -f "$PID_FILE"
  fi
fi

# ── Activate venv ─────────────────────────────────────────────────────────
if [ ! -d "$VENV" ]; then
  echo "Error: virtual environment not found at $VENV"
  echo "Run: python -m venv $VENV && $VENV/bin/pip install -r $APP_DIR/requirements.txt"
  exit 1
fi

# ── Start server ──────────────────────────────────────────────────────────
echo "Starting LRP Coach on http://localhost:$PORT …"
cd "$APP_DIR"
"$VENV/bin/python" app.py > "$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo $SERVER_PID > "$PID_FILE"

# ── Wait for port ─────────────────────────────────────────────────────────
for i in $(seq 1 30); do
  if lsof -i ":$PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! lsof -i ":$PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
  echo "Server did not start in time. Check $LOG_FILE for errors."
  rm -f "$PID_FILE"
  exit 1
fi

echo "LRP Coach is up — opening browser."
open "http://localhost:$PORT"

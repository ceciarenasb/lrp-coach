#!/usr/bin/env bash
# Stop a running LRP Coach server.
# Usage: ./scripts/stop.sh

set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PID_FILE="$APP_DIR/.lrp_coach.pid"
PORT=7860

if [ -f "$PID_FILE" ]; then
  PID=$(cat "$PID_FILE")
  if kill -0 "$PID" 2>/dev/null; then
    echo "Stopping LRP Coach (PID $PID) …"
    kill "$PID"
    rm -f "$PID_FILE"
    echo "Done."
  else
    echo "PID $PID not found — already stopped?"
    rm -f "$PID_FILE"
  fi
else
  # Fallback: find anything on the port
  PID=$(lsof -i ":$PORT" -sTCP:LISTEN -t 2>/dev/null | head -1 || true)
  if [ -n "$PID" ]; then
    echo "Stopping process $PID on port $PORT …"
    kill "$PID"
    echo "Done."
  else
    echo "LRP Coach does not appear to be running."
  fi
fi

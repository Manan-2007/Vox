#!/usr/bin/env bash
# Vox — one-command start (macOS / Linux).
# Starts the FastAPI backend and the Vite frontend, then opens the browser.
set -euo pipefail
cd "$(dirname "$0")"

PY=python3.12
command -v "$PY" >/dev/null 2>&1 || PY=python3

if [ ! -d venv ]; then
  echo "[vox] creating venv + installing Python deps (first run only)…"
  "$PY" -m venv venv
  ./venv/bin/pip install -r requirements.txt
fi

if [ ! -d frontend/node_modules ]; then
  echo "[vox] installing frontend deps (first run only)…"
  (cd frontend && npm install)
fi

if [ ! -f ml/models/vox_lstm.keras ]; then
  echo "[vox] NOTE: no trained model at ml/models/vox_lstm.keras."
  echo "[vox]       The app will run, but recognition needs:"
  echo "[vox]       python ml/collect.py <sign> && python ml/preprocess.py && python ml/train.py"
fi

echo "[vox] starting backend on :8000"
./venv/bin/uvicorn backend.main:app --port 8000 &
BACKEND_PID=$!

echo "[vox] starting frontend on :5173"
(cd frontend && npm run dev -- --port 5173 --strictPort) &
FRONTEND_PID=$!

cleanup() {
  echo; echo "[vox] shutting down…"
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

printf "[vox] waiting for the dev server"
until curl -s -o /dev/null http://localhost:5173/; do printf "."; sleep 0.5; done
echo " ready."

URL="http://localhost:5173/"
if command -v open >/dev/null 2>&1; then open "$URL"
elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL"
else echo "[vox] open $URL in your browser"
fi

echo "[vox] running — Ctrl-C stops both servers."
wait

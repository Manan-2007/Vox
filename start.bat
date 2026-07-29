@echo off
rem Vox - one-command start (Windows).
rem Starts backend + frontend in their own windows, then opens the browser.
cd /d "%~dp0"

if not exist venv (
  echo [vox] creating venv + installing Python deps ^(first run only^)...
  py -3.12 -m venv venv || python -m venv venv
  venv\Scripts\pip install -r requirements.txt
)

if not exist frontend\node_modules (
  echo [vox] installing frontend deps ^(first run only^)...
  pushd frontend
  call npm install
  popd
)

if not exist ml\models\vox_lstm.keras (
  echo [vox] NOTE: no trained model at ml\models\vox_lstm.keras.
  echo [vox]       The app runs, but recognition needs collect / preprocess / train first.
)

echo [vox] starting backend on :8000
start "vox-backend" cmd /k venv\Scripts\uvicorn backend.main:app --port 8000

echo [vox] starting frontend on :5173
start "vox-frontend" cmd /k "cd frontend && npm run dev -- --port 5173 --strictPort"

echo [vox] waiting for the dev server...
timeout /t 6 /nobreak >nul
start http://localhost:5173/

echo [vox] running - close the two server windows to stop.

# Vox

Real-time Indian Sign Language (ISL) recognizer. See [ARCHITECTURE.md](ARCHITECTURE.md)
for the full design.

**Status: Milestone 0 (spike) — scaffolding only. No feature code yet.**

## Layout

```
vox/
  ml/         # collect.py, preprocess.py, train.py, normalize.py, data/, models/
  backend/    # main.py (FastAPI + WebSocket)
  frontend/   # React + Vite + TypeScript
```

## Setup

Python 3.12 (`mediapipe` and `tensorflow` do not both have wheels on 3.13+).

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

```bash
cd frontend && npm install
```

## Running

Activate the Python env:

```bash
source venv/bin/activate
```

Backend (once `backend/main.py` exists):

```bash
source venv/bin/activate && uvicorn backend.main:app --reload --port 8000
```

Frontend dev server (http://localhost:5173):

```bash
cd frontend && npm run dev
```

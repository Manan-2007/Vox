# Vox

Real-time, two-way Indian Sign Language (ISL) interpreter.

- **Sign → speech:** hands are tracked in the browser (MediaPipe, in a Web
  Worker), an LSTM on the backend recognizes isolated signs, sentences build
  word by word and are read aloud (Web Speech API).
- **Speech → sign:** spoken English is transcribed, glossed into ISL tokens,
  and played back as a sequence of sign video clips.

No video ever leaves the machine — only 126 hand-landmark floats per frame go
over a local WebSocket. See [ARCHITECTURE.md](ARCHITECTURE.md) for the design.

## Quick start

```bash
./start.sh        # macOS / Linux — creates venv + node_modules on first run
start.bat         # Windows
```

Both start the backend (`:8000`) and frontend (`:5173`) and open the browser.

## Manual setup

Python **3.12** required (`mediapipe` and `tensorflow` have no common wheels on
3.13+). Node 20+.

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd frontend && npm install
```

Run the backend and frontend in two shells:

```bash
source venv/bin/activate && uvicorn backend.main:app --port 8000
```

```bash
cd frontend && npm run dev
```

## Training the vocabulary

The repo ships **no trained model** — recognition needs data collected on your
own webcam (30 frames x 126 landmarks per sample, raw; normalization is shared
between training and inference via `ml/normalize.py`):

```bash
source venv/bin/activate
python ml/collect.py hello        # SPACE records one sample; 30-50 per sign
python ml/collect.py thanks       # ...repeat per sign, 2-3 different people
python ml/preprocess.py           # normalize + stratified 80/20 split
python ml/train.py                # LSTM + augmentation -> ml/models/vox_lstm.keras
```

`train.py` prints validation accuracy, a per-class report, and writes
`ml/models/confusion_matrix.png`. Drop or re-record any sign that shows up
off-diagonal. The backend picks up the new model + label map on restart; the
frontend needs **no changes** — it displays whatever words the backend emits.

### Current sign list

Defined by `ml/models/label_map.json` after training. The clip manifest and
demo script assume this starter set:

`bye · eat · hello · help · no · please · thanks · yes`

## Speech → sign clips

Supply one short video (1–3 s) per word in
`frontend/public/clips/` and list it in
[manifest.json](frontend/public/clips/manifest.json).
`hello` and `thanks` currently point at generated placeholder rectangles so the
player is testable — replace them. Words without a clip are skipped and named
in the UI.

## Known limitations

- **Vocabulary is small and fixed** — isolated signs only; continuous signing
  and fingerspelling are out of scope.
- **Hands only.** ISL uses facial expression and body pose that the model
  never sees; regional ISL variation means signs may differ from your dialect.
- **Transition artifacts:** when one sign flows into another, the sliding
  window briefly contains a mixture and can emit a confident wrong word
  (documented in `backend/main.py`; mitigated by the stability gate and the
  Undo control, not eliminated).
- Validation accuracy is optimistic (it drives early stopping); judge the
  model live.
- SpeechRecognition (mic input) is Chrome/Edge-only; the typed-phrase box
  drives the same pipeline everywhere else.

## Layout

```
vox/
  ml/          collect.py · preprocess.py · train.py · normalize.py · data/ · models/
  backend/     main.py (FastAPI + WebSocket inference) · test_client.py
  frontend/    React + Vite + TS (landing + session; MediaPipe in a Web Worker)
  docs/        DEMO.md (demo script + shot list)
```

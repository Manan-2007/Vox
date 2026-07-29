# Vox

Real-time, two-way Indian Sign Language (ISL) interpreter — **works out of the
box**: the trained model and the sign clip library ship with the repo.

- **Sign → speech:** hands are tracked in the browser (MediaPipe, in a Web
  Worker), an LSTM on the backend recognizes isolated signs, sentences build
  word by word and are read aloud (Web Speech API). Voice output is a
  first-class toggle — Deaf, mute, or hearing users each choose voice or
  text-only.
- **Speech → sign:** spoken English is transcribed, glossed into ISL tokens,
  and played back as a sequence of real ISL dictionary clips.

No video ever leaves the machine — only 126 hand-landmark floats per frame go
over a local WebSocket. See [ARCHITECTURE.md](ARCHITECTURE.md) for the design.

## Quick start

```bash
./start.sh        # macOS / Linux — creates venv + node_modules on first run
start.bat         # Windows
```

Both start the backend (`:8000`) and frontend (`:5173`) and open the browser.

## Vocabulary

`bye · come · eat · hello · help · please · sorry · thanks`

The model was trained on public ISL dictionary videos (ISLRTC, Goa Board of
Education, DEAF TV — via YouTube), and **the same clips play in the app's
Speech → ISL panel**. That loop matters: type a word, watch its clip, copy the
sign — you are then signing exactly the form the model was trained on.

Sign close to the dictionary form, hold each sign ~2 seconds, and pause
between sentences.

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

## Retraining / extending the vocabulary

The shipped model was built like this — rerun it to reproduce or extend:

```bash
source venv/bin/activate
pip install yt-dlp                                  # one-time
python ml/fetch_videos.py --out videos/             # sources in ml/videos.txt
python ml/extract.py --videos-dir videos/           # video -> (30,126) samples
python ml/preprocess.py                             # normalize + split
python ml/train.py                                  # -> ml/models/vox_lstm.keras
```

To add a word: add `<word> <youtube-id>` lines to `ml/videos.txt` (short
dictionary-style clips work best), then rerun the four commands. You can also
record your own samples on webcam with `python ml/collect.py <word>` — mixing
your own recordings with the dictionary data is the best way to make
recognition robust for *your* signing.

`train.py` prints validation accuracy and writes
`ml/models/confusion_matrix.png`. The backend picks the new model up on
restart; the frontend needs no changes. Add a clip for the new word in
`frontend/public/clips/` + `manifest.json`.

## Known limitations

- **Small, fixed vocabulary of isolated signs** — continuous signing and
  fingerspelling are out of scope.
- **Trained on a handful of dictionary signers.** Regional or personal
  variants of a sign may not be recognized — mimic the in-app clip for best
  results, or add your own recordings (above). In our held-out test, a
  different signer's variant of one sign was missed (though the confidence
  gate kept it from emitting a *wrong* word).
- **Hands only.** ISL uses facial expression and body pose that the model
  never sees.
- Window-level validation accuracy is optimistic (overlapping windows);
  judge the model live.
- SpeechRecognition (mic input) is Chrome/Edge-only; the typed-phrase box
  drives the same pipeline everywhere else.

## Credits

Sign video sources: [ISLRTC](https://www.youtube.com/@islrtc) (Govt. of
India), Goa Board of Education, and DEAF TV — ISL dictionary clips on
YouTube, used here for local training and demo playback. The orb assistant
started from the `orb-ai-assistant` AI-Studio template, reworked for Vox.

## Layout

```
vox/
  ml/          fetch_videos.py · extract.py · collect.py · preprocess.py
               train.py · normalize.py · videos.txt · data/ · models/
  backend/     main.py (FastAPI + WebSocket inference) · test_client.py
  frontend/    React + Vite + TS (landing + session; MediaPipe in a Web Worker)
  docs/        DEMO.md (demo script + shot list)
```

# Vox

Real-time, two-way Indian Sign Language (ISL) interpreter — **works out of the
box**: the trained model and the sign clip library ship with the repo.

- **Sign → speech:** hands are tracked in the browser (MediaPipe, in a Web
  Worker), an LSTM on the backend recognizes isolated signs, sentences build
  word by word and are read aloud (Web Speech API). Voice output is a
  first-class toggle — Deaf, mute, or hearing users each choose voice or
  text-only.
- **Speech → sign:** spoken English is transcribed, glossed into ISL tokens,
  and played back as real ISL clips, alongside the sign's 3D motion.
- **3D holistic motion:** hands *and* upper body are tracked and rendered as a
  live 3D skeleton — both your own signing and the reference clip.

No video ever leaves the machine — only 141 landmark floats per frame go over a
local WebSocket. See [ARCHITECTURE.md](ARCHITECTURE.md) for the design and
[docs/3D-AVATAR.md](docs/3D-AVATAR.md) for the 3D work.

## Quick start

```bash
./start.sh        # macOS / Linux — creates venv + node_modules on first run
start.bat         # Windows
```

Both start the backend (`:8000`) and frontend (`:5173`) and open the browser.

## Vocabulary

`hello · how are you · thank you · pleased · alright · good morning`

Trained on the [INCLUDE](https://zenodo.org/records/4010759) ISL dataset —
**21 different signers per word** — and scored **100% on held-out signers the
model never trained on** (validation splits by source video, so no signer
appears on both sides).

The same recordings play in the app's Speech → ISL panel. That loop matters:
type the phrase, watch its clip *and its 3D motion*, copy it — you are then
signing exactly the form the model was trained on. Hold each sign ~2 seconds
and pause between sentences.

Four more greetings (good afternoon / evening / night) were trained and then
**deliberately dropped**: they share a handshape with "good morning" and the
model could not separate them reliably (they scored 0.17–0.40 F1 while every
shipped sign scored 1.00). Shipping them would have meant shipping wrong
answers.

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

The shipped model was built from INCLUDE. To reproduce:

1. Download a category from [Zenodo](https://zenodo.org/records/4010759)
   (`Greetings_1of2.zip`, `Greetings_2of2.zip`) and unzip it.
2. Then:

```bash
source venv/bin/activate
python ml/stage_include.py --include-dir <unzipped>/Greetings --out videos/
python ml/extract.py --videos-dir videos/     # video -> (30,141) samples
python ml/preprocess.py                       # normalize + split BY SOURCE VIDEO
python ml/train.py --augment 5                # -> ml/models/vox_lstm.keras
```

**More signers beats more samples.** Measured on the same code, same split
method — only the data differed:

| training data | accuracy on unseen signers |
|---|---|
| 2–8 YouTube dictionary videos per word | 27.9% |
| 21 INCLUDE signers per word | **100%** |

To add a word, download the INCLUDE category containing it. You can also
record yourself with `python ml/collect.py <word>` — mixing your own
recordings in is the best way to make recognition robust for *your* signing.
`ml/videos.txt` + `ml/fetch_videos.py` still fetch single-signer YouTube
dictionary clips, but a class built only from those will memorise the signer.

`train.py` prints validation accuracy and writes
`ml/models/confusion_matrix.png`. The backend picks the new model up on
restart; the frontend needs no changes. Add a clip for the new word in
`frontend/public/clips/` + `manifest.json`.

## Why a sign might not be recognized

The session view now shows the model's **live top-3** under the camera, plus a
hint over the video when a frame is unusable. Read them in this order:

1. **"Move back a little"** — your shoulders must be in frame. Normalization
   anchors on them; without a body the frame is discarded and the model sees
   literally nothing.
2. **"No hands detected"** — raise your hands into view.
3. **Top-3 shows your sign but below the marker** — it is being recognized but
   not confidently enough. Lower the threshold in Settings, or match the
   reference clip more closely.
4. **Top-3 never lists your sign** — the model does not know that form. Watch
   the clip in the Speech → ISL panel and copy it, or record your own samples
   (below) and retrain.

## Roadmap

[docs/ROADMAP.md](docs/ROADMAP.md) assesses the four long-term goals —
two-way conversation, generalization, continuous signing, and facial grammar
— with what each actually costs, plus what is and is not reusable from the
related ISL projects surveyed.

## Known limitations

- **Small, fixed vocabulary of isolated signs** — continuous signing and
  fingerspelling are out of scope.
- **Six signs only**, and they were chosen for separability. Signs that share
  a handshape and differ only in a small detail are not reliably
  distinguishable at this data scale — see the dropped time-of-day greetings.
- **Regional variation.** INCLUDE signers use one set of forms; mimic the
  in-app clip, or add your own recordings.
- **Hands + upper body, no face.** ISL uses facial expression the model never
  sees. Depth is MediaPipe's rough estimate, not true 3D.
- Validation splits by source video, so the reported number is honest — but 21
  signers is still a small population. Judge it live.
- SpeechRecognition (mic input) is Chrome/Edge-only; the typed-phrase box
  drives the same pipeline everywhere else.

## Credits

Training data and sign clips: **INCLUDE — A Large Scale Dataset for Indian
Sign Language Recognition** (CC-BY-4.0),
<https://zenodo.org/records/4010759>. Additional dictionary clips referenced
in `ml/videos.txt` come from [ISLRTC](https://www.youtube.com/@islrtc) (Govt.
of India), Goa Board of Education, NCERT, and others on YouTube. The orb
assistant started from the `orb-ai-assistant` AI-Studio template, reworked for
Vox. 3D avatar notes and a SignAvatars assessment: [docs/3D-AVATAR.md](docs/3D-AVATAR.md).

## Layout

```
vox/
  ml/          stage_include.py · extract.py · collect.py · preprocess.py
               train.py · normalize.py · clip_motion.py · fetch_videos.py
               videos.txt · data/ · models/
  backend/     main.py (FastAPI + WebSocket inference) · test_client.py
  frontend/    React + Vite + TS (landing + session; MediaPipe in a Web Worker)
  docs/        DEMO.md (demo script) · 3D-AVATAR.md (3D + SignAvatars)
```

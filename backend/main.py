"""Vox inference backend — FastAPI + WebSocket.

The browser does the computer vision and streams 126-float landmark frames here;
this process does the ML. Per connection we keep a 30-frame sliding window,
predict on every frame once it is full, and gate emissions on confidence plus
stability.

Normalization is imported from ml/normalize.py — the SAME function training
used. It is never reimplemented here; that divergence is the single most
likely way to make this whole system silently output garbage.

Run:
    source venv/bin/activate
    uvicorn backend.main:app --reload --port 8000

Env:
    VOX_MODEL_DIR   directory holding vox_lstm.keras + label_map.json
                    (default: ml/models)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

REPO_ROOT = Path(__file__).resolve().parent.parent
ML_DIR = REPO_ROOT / "ml"

# The one shared normalization implementation — see module docstring.
sys.path.insert(0, str(ML_DIR))
from normalize import FEATURE_DIM, SEQUENCE_LENGTH, normalize_frame  # noqa: E402

CONFIDENCE_THRESHOLD = 0.85
STABILITY_FRAMES = 3  # identical top class this many predictions in a row
# A label the model may predict but which is never emitted as a word: it means
# "no sign is being made". Training includes it so the classifier has an honest
# way to reject idle hands instead of forcing them into a real word.
REJECT_LABEL = "rest"

# ---------------------------------------------------------------------------
# VOCABULARY GATING — why the model knows 242 words and offers 37
# ---------------------------------------------------------------------------
# Measured on the held-out split (ml/models/metrics.json):
#
#     242 classes, 1065 training samples, MEDIAN 2 SAMPLES PER CLASS
#     top-1 accuracy 0.382
#     37 classes at 80%+ ; 118 classes with no test sample at all
#
# 4.4 samples per class is not a tuning problem, it is the information-theoretic
# ceiling of this dataset. No threshold recovers a class the model never learned.
#
# Offering all 242 and being right 38% of the time is worse than offering 37 and
# being right 80% of the time, and for an accessibility product it is much worse:
# this is used in medical conversations, where a confidently wrong word is a
# safety problem rather than an annoyance. So a prediction outside the verified
# set is never emitted — it is reported as unrecognised, which is true.
#
# The gate is DATA, not a hard-coded list: ml/evaluate.py writes the verified set
# after measuring it, so retraining on more data widens the vocabulary
# automatically. Set VOX_VERIFIED_ONLY=0 to disable for debugging.
VERIFIED_ONLY = os.environ.get("VOX_VERIFIED_ONLY", "1") != "0"
# Words at or above this held-out accuracy are offered to users.
VERIFIED_ACCURACY = 0.8

MODEL_DIR = Path(os.environ.get("VOX_MODEL_DIR", ML_DIR / "models"))
MODEL_PATH = MODEL_DIR / "vox_lstm.keras"
LABEL_MAP_PATH = MODEL_DIR / "label_map.json"
METRICS_PATH = MODEL_DIR / "metrics.json"

logging.basicConfig(
    level=os.environ.get("VOX_LOG_LEVEL", "INFO"),
    format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
)
log = logging.getLogger("vox")

# Keras predict() is not safe to call concurrently, and it blocks for ~16ms.
# Serialize it with a lock and run it off the event loop so one client's
# inference never stalls another's socket.
_predict_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and label map once, at startup."""
    app.state.model = None
    app.state.labels = []

    if not MODEL_PATH.exists() or not LABEL_MAP_PATH.exists():
        # Deliberately not fatal: /health should be able to tell you what's
        # missing rather than the process refusing to boot.
        log.error("Model not found in %s", MODEL_DIR)
        log.error("  expected %s and %s", MODEL_PATH.name, LABEL_MAP_PATH.name)
        log.error("  run: python ml/collect.py <label> && python ml/preprocess.py && python ml/train.py")
        log.error("Serving /health only; /ws will refuse connections.")
        yield
        return

    import keras  # slow import, and only needed once we know there's a model

    # Importing this registers the custom layers the saved model is built from.
    # Without it load_model raises "Unknown layer: AttentionPooling", and the
    # only clue is a stack trace deep inside Keras deserialization.
    import layers  # noqa: F401

    log.info("Loading model from %s", MODEL_PATH)
    model = keras.models.load_model(MODEL_PATH)
    labels = json.loads(LABEL_MAP_PATH.read_text())["labels"]

    _, timesteps, features = model.input_shape
    if (timesteps, features) != (SEQUENCE_LENGTH, FEATURE_DIM):
        raise RuntimeError(
            f"Model expects {(timesteps, features)}, backend streams "
            f"{(SEQUENCE_LENGTH, FEATURE_DIM)}. Retrain or fix the contract."
        )
    if model.output_shape[-1] != len(labels):
        raise RuntimeError(
            f"Model has {model.output_shape[-1]} outputs but label_map.json has "
            f"{len(labels)} labels — they are out of sync. Retrain."
        )

    # Warm up: the first predict() pays for tracing/compiling, which would
    # otherwise land on the first real frame of the first connection.
    model.predict(np.zeros((1, SEQUENCE_LENGTH, FEATURE_DIM), np.float32), verbose=0)

    app.state.model = model
    app.state.labels = labels
    app.state.verified = load_verified(labels)

    log.info("Ready — %d labels", len(labels))
    if VERIFIED_ONLY:
        log.info(
            "Emitting only the %d VERIFIED word(s) (>=%.0f%% on held-out "
            "recordings); the other %d are recognised internally but reported "
            "as unrecognised. Set VOX_VERIFIED_ONLY=0 to disable.",
            len(app.state.verified),
            VERIFIED_ACCURACY * 100,
            len(labels) - len(app.state.verified),
        )
        log.info("  %s", ", ".join(sorted(app.state.verified)))
    else:
        log.warning(
            "VOX_VERIFIED_ONLY=0 — all %d classes may be emitted. Measured "
            "top-1 on this model is 38%%; do not use this setting in front of "
            "a real user.",
            len(labels),
        )
    yield
    log.info("Shutting down")


def load_verified(labels: list[str]) -> set[str]:
    """The words measured accurate enough to offer, from ml/evaluate.py's output.

    Read from disk rather than hard-coded so that retraining widens the
    vocabulary without a code change. A missing or unreadable metrics file is
    NOT treated as "everything is fine" — with no evidence, nothing is verified,
    and the app says so. Failing open here would silently restore exactly the
    behaviour this gate exists to prevent.
    """
    if not METRICS_PATH.exists():
        log.warning(
            "%s not found — no word has measured accuracy, so none will be "
            "emitted. Run: python ml/evaluate.py",
            METRICS_PATH,
        )
        return set()
    try:
        metrics = json.loads(METRICS_PATH.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        log.error("Could not read %s (%s) — treating no word as verified", METRICS_PATH, exc)
        return set()

    # Prefer the explicit list; fall back to recomputing from per-class scores so
    # an older metrics file still works.
    reliable = metrics.get("reliable")
    if isinstance(reliable, list):
        verified = {word for word in reliable if word in labels}
    else:
        verified = {
            word
            for word, stats in (metrics.get("per_class") or {}).items()
            if word in labels
            and stats.get("verified")
            and (stats.get("top1") or 0) >= VERIFIED_ACCURACY
        }
    return verified


app = FastAPI(title="Vox", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    loaded = app.state.model is not None
    return {
        "status": "ok" if loaded else "degraded",
        "reject_label": REJECT_LABEL,
        "model_loaded": loaded,
        "model_dir": str(MODEL_DIR),
        "labels": app.state.labels,
        "verified_only": VERIFIED_ONLY,
        "verified": sorted(getattr(app.state, "verified", set())),
        "sequence_length": SEQUENCE_LENGTH,
        "feature_dim": FEATURE_DIM,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "stability_frames": STABILITY_FRAMES,
    }


def parse_landmarks(payload: dict) -> np.ndarray:
    """Validate one incoming message and return its raw 126-float vector.

    Raises ValueError with a client-facing message.
    """
    landmarks = payload.get("landmarks")
    if landmarks is None:
        raise ValueError("missing 'landmarks'")
    if not isinstance(landmarks, list):
        raise ValueError("'landmarks' must be a list")
    if len(landmarks) != FEATURE_DIM:
        raise ValueError(f"expected {FEATURE_DIM} floats, got {len(landmarks)}")
    try:
        vec = np.asarray(landmarks, dtype=np.float32)
    except (TypeError, ValueError):
        raise ValueError("'landmarks' must contain only numbers")
    if not np.isfinite(vec).all():
        raise ValueError("'landmarks' contains NaN or Inf")
    return vec


async def predict(model, window: deque) -> np.ndarray:
    """Run one prediction on the (1, SEQUENCE_LENGTH, FEATURE_DIM) window."""
    batch = np.stack(window)[None].astype(np.float32)
    async with _predict_lock:
        probs = await asyncio.to_thread(model.predict, batch, verbose=0)
    return probs[0]


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    model = app.state.model
    labels = app.state.labels
    verified: set[str] = getattr(app.state, "verified", set())
    if model is None:
        # Stay connected: the speech->ISL half of the app works without a
        # model, and closing here put the frontend into a reconnect loop that
        # made the whole product look dead. Tell the client once, then answer
        # every frame with the same status instead of predictions.
        await websocket.send_json({"status": "no-model"})
        try:
            while True:
                await websocket.receive_text()
                await websocket.send_json({"status": "no-model"})
        except WebSocketDisconnect:
            return

    client = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "?"
    log.info("client connected: %s", client)

    window: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)
    stable_class: int | None = None   # top class of the previous prediction
    stable_count = 0                  # how many predictions in a row it has won
    emitted_class: int | None = None  # latch, see below
    frames = 0
    # Per-connection settings; the UI's settings panel can adjust these.
    conf_threshold = CONFIDENCE_THRESHOLD

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise ValueError("message must be a JSON object")

                # Settings message: {"config": {"confidence_threshold": 0.9}}
                if "config" in payload:
                    cfg = payload["config"]
                    if not isinstance(cfg, dict):
                        raise ValueError("'config' must be an object")
                    if "confidence_threshold" in cfg:
                        value = float(cfg["confidence_threshold"])
                        if not 0.5 <= value <= 0.99:
                            raise ValueError(
                                "confidence_threshold must be in [0.5, 0.99]"
                            )
                        conf_threshold = value
                        log.info("config: threshold=%.2f (%s)", value, client)
                    await websocket.send_json(
                        {"status": "config", "confidence_threshold": conf_threshold}
                    )
                    continue

                vec = parse_landmarks(payload)
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                await websocket.send_json({"error": str(exc)})
                continue

            normalized = normalize_frame(vec)
            window.append(normalized)
            frames += 1

            # Frame quality, so the UI can say WHY nothing is happening rather
            # than sitting silent: no body -> normalize_frame zeroes the frame
            # and the model genuinely sees nothing.
            quality = {
                "hands": int(bool(vec[:126][:63].any())) + int(bool(vec[63:126].any())),
                "body": bool(vec[126:].any()),
                "usable": bool(normalized.any()),
            }

            if len(window) < SEQUENCE_LENGTH:
                await websocket.send_json(
                    {"status": "listening", "buffered": len(window),
                     "needed": SEQUENCE_LENGTH, "quality": quality}
                )
                continue

            probs = await predict(model, window)
            top = int(np.argmax(probs))
            confidence = float(probs[top])
            # Top-3 with scores: the single most useful thing for understanding
            # why a sign is not being accepted.
            ranked = np.argsort(probs)[::-1][:3]
            top3 = [
                {"word": labels[int(i)], "confidence": float(probs[int(i)])}
                for i in ranked
            ]

            if top == stable_class:
                stable_count += 1
            else:
                stable_class = top
                stable_count = 1
                emitted_class = None  # a new class may be emitted again

            log.debug(
                "frame %d  top=%s conf=%.3f run=%d", frames, labels[top],
                confidence, stable_count,
            )

            # Why each condition is here:
            #   reject      — the model's own "no sign is being made" class
            #   verified    — see the VOCABULARY GATING note at the top
            #   confidence  — the calibration gate
            #   stability   — the same class must win several predictions running
            #   latch       — the window slides one frame at a time, so without
            #                 this a held sign re-emits on every single frame
            word = labels[top]
            is_reject = word == REJECT_LABEL
            is_verified = (not VERIFIED_ONLY) or word in verified
            accept = (
                not is_reject
                and is_verified
                and confidence > conf_threshold
                and stable_count >= STABILITY_FRAMES
                and emitted_class != top
            )

            if accept:
                emitted_class = top
                log.info("WORD  %-16s conf=%.3f  (%s)", word, confidence, client)
                await websocket.send_json(
                    {"word": word, "confidence": confidence,
                     "top3": top3, "quality": quality}
                )
            else:
                # Say WHY nothing was emitted.
                #
                # The frontend used to render `top3[0].word` in the headline slot
                # regardless of confidence, so the model's argmax over 242 classes
                # — frequently a word with two training samples and 0% held-out
                # accuracy — was displayed exactly where a user reads "this is
                # what you signed". The backend was already refusing to emit it;
                # the interface was showing it anyway. `reason` exists so the UI
                # can never make that mistake again by omission.
                if is_reject:
                    reason = "resting"
                elif not is_verified:
                    reason = "unverified"
                elif confidence <= conf_threshold:
                    reason = "low-confidence"
                elif stable_count < STABILITY_FRAMES:
                    reason = "unstable"
                else:
                    reason = "already-emitted"

                await websocket.send_json(
                    {
                        "status": "listening",
                        # Deliberately NOT called "top": the old key invited the
                        # client to treat it as a prediction. This is a candidate.
                        "candidate": word,
                        "confidence": confidence,
                        "accepted": False,
                        "reason": reason,
                        "stable_for": stable_count,
                        "top3": top3,
                        "quality": quality,
                    }
                )

    except WebSocketDisconnect:
        log.info("client disconnected: %s (%d frames)", client, frames)
    except Exception:
        log.exception("connection error: %s", client)
        try:
            await websocket.close(code=1011)
        except RuntimeError:
            pass

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
from normalize import FEATURE_DIM, normalize_frame  # noqa: E402

SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.85
STABILITY_FRAMES = 3  # identical top class this many predictions in a row

MODEL_DIR = Path(os.environ.get("VOX_MODEL_DIR", ML_DIR / "models"))
MODEL_PATH = MODEL_DIR / "vox_lstm.keras"
LABEL_MAP_PATH = MODEL_DIR / "label_map.json"

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
    log.info("Ready — %d labels: %s", len(labels), ", ".join(labels))
    yield
    log.info("Shutting down")


app = FastAPI(title="Vox", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    loaded = app.state.model is not None
    return {
        "status": "ok" if loaded else "degraded",
        "model_loaded": loaded,
        "model_dir": str(MODEL_DIR),
        "labels": app.state.labels,
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
    """Run one prediction on the (1, 30, 126) window, off the event loop."""
    batch = np.stack(window)[None].astype(np.float32)
    async with _predict_lock:
        probs = await asyncio.to_thread(model.predict, batch, verbose=0)
    return probs[0]


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    model = app.state.model
    labels = app.state.labels
    if model is None:
        await websocket.send_json(
            {"error": "no model loaded — train one first (see /health)"}
        )
        await websocket.close(code=1011)
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

            window.append(normalize_frame(vec))
            frames += 1

            if len(window) < SEQUENCE_LENGTH:
                await websocket.send_json(
                    {"status": "listening", "buffered": len(window),
                     "needed": SEQUENCE_LENGTH}
                )
                continue

            probs = await predict(model, window)
            top = int(np.argmax(probs))
            confidence = float(probs[top])

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

            accept = (
                confidence > conf_threshold
                and stable_count >= STABILITY_FRAMES
                # Latch: the window slides one frame at a time, so without this
                # a held sign would re-emit its word on every single frame.
                # Cleared above when the winning class changes.
                and emitted_class != top
            )

            if accept:
                emitted_class = top
                log.info("WORD  %-16s conf=%.3f  (%s)", labels[top], confidence, client)
                await websocket.send_json(
                    {"word": labels[top], "confidence": confidence}
                )
            else:
                await websocket.send_json(
                    {
                        "status": "listening",
                        "top": labels[top],
                        "confidence": confidence,
                        "stable_for": stable_count,
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

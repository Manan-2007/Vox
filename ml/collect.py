"""Webcam data collection for ISL landmark sequences.

Records one sample = SEQUENCE_LENGTH consecutive frames of RAW hand landmarks,
saved as ml/data/<label>/<uuid>.npy with shape (SEQUENCE_LENGTH, 126).

Raw landmarks only — normalization lives in ml/normalize.py and is applied at
preprocess/train/inference time, never here.

================================================================================
THE 126-FLOAT FRAME CONTRACT  --  the frontend must reproduce this EXACTLY
================================================================================

Layout, per frame:

    [  0 : 63 ]  Left  hand: 21 landmarks x (x, y, z)
    [ 63 :126 ]  Right hand: 21 landmarks x (x, y, z)

  - Landmarks stay in MediaPipe's own order (0 = wrist ... 20 = pinky tip) and in
    its normalized coordinate space: x, y in [0, 1] relative to image width and
    height, z in the same scale as x with the wrist as the depth origin.
  - "Left" / "Right" is the `category_name` string MediaPipe reports for that
    hand, NOT a guess from screen position.
  - An absent hand's 63 slots are zeros. No hands -> 126 zeros. A frame is always
    emitted, so a sequence is always exactly SEQUENCE_LENGTH frames long.
  - If MediaPipe labels both detected hands the same way, the higher-confidence
    one wins that block and the other is dropped. Never two hands in one block.

Two conventions that are easy to get wrong and silently poison the model:

  1. FEED THE RAW, UNMIRRORED CAMERA FRAME. This script mirrors the image for
     display ONLY, after landmarks have been computed. In the browser,
     `detectForVideo(videoEl, ts)` on an un-flipped <video> is the equivalent —
     so a CSS `transform: scaleX(-1)` on the preview is fine, but do not flip
     the pixels that reach the detector.
  2. Because of (1), MediaPipe's handedness label is anatomically inverted: the
     model assumes a mirrored selfie image, so the hand it calls "Left" is the
     signer's right. That is fine and deliberate — both sides make the same
     assumption, so the blocks line up. What matters is that collection and
     inference agree, not that the label is anatomically true. Do not "fix" this
     on one side only.

Usage:
    python ml/collect.py hello
    python ml/collect.py            # prompts for the label
    python ml/collect.py hello --camera 1

Keys:
    SPACE   record one sample
    Q / ESC quit
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import uuid
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarksConnections

# --- frame contract ----------------------------------------------------------
HAND_ORDER = ("Left", "Right")  # block order; index 0 -> floats 0:63, 1 -> 63:126
LANDMARKS_PER_HAND = 21
COORDS_PER_LANDMARK = 3  # x, y, z
FEATURES_PER_HAND = LANDMARKS_PER_HAND * COORDS_PER_LANDMARK  # 63
FEATURE_DIM = FEATURES_PER_HAND * len(HAND_ORDER)  # 126
SEQUENCE_LENGTH = 30

# --- paths -------------------------------------------------------------------
ML_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ML_DIR / "data"
DEFAULT_MODEL_PATH = ML_DIR / "models" / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# --- drawing -----------------------------------------------------------------
# BGR. Distinct per block so you can see which 63 floats a hand is filling.
HAND_COLORS = {"Left": (255, 176, 0), "Right": (0, 200, 255)}
IDLE_COLOR = (200, 200, 200)
REC_COLOR = (0, 0, 255)


def build_frame_vector(result) -> np.ndarray:
    """Flatten one HandLandmarkerResult into the 126-float frame vector.

    See the module docstring for the layout. This is the single place the
    ordering and zero-padding rules are implemented; the frontend mirrors it.
    """
    vec = np.zeros(FEATURE_DIM, dtype=np.float32)
    if not result.hand_landmarks:
        return vec

    # Resolve one detection per block first, so a duplicated handedness label
    # can't let a second hand overwrite the first non-deterministically.
    best: dict[str, tuple[float, list]] = {}
    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        if not handedness:
            continue
        label = handedness[0].category_name
        score = handedness[0].score
        if label not in HAND_ORDER:
            continue
        if label not in best or score > best[label][0]:
            best[label] = (score, landmarks)

    for block, label in enumerate(HAND_ORDER):
        if label not in best:
            continue  # absent hand keeps its 63 zeros
        offset = block * FEATURES_PER_HAND
        for i, lm in enumerate(best[label][1][:LANDMARKS_PER_HAND]):
            base = offset + i * COORDS_PER_LANDMARK
            vec[base] = lm.x
            vec[base + 1] = lm.y
            vec[base + 2] = lm.z
    return vec


def filled_blocks(vec: np.ndarray) -> dict[str, bool]:
    """Which hand blocks carry data in this vector (for the on-screen readout)."""
    return {
        label: bool(
            vec[i * FEATURES_PER_HAND : (i + 1) * FEATURES_PER_HAND].any()
        )
        for i, label in enumerate(HAND_ORDER)
    }


def draw_landmarks(frame: np.ndarray, result) -> None:
    """Draw the hand skeleton onto the raw (un-mirrored) frame, in place."""
    h, w = frame.shape[:2]
    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        label = handedness[0].category_name if handedness else "?"
        color = HAND_COLORS.get(label, IDLE_COLOR)
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for c in HandLandmarksConnections.HAND_CONNECTIONS:
            if c.start < len(pts) and c.end < len(pts):
                cv2.line(frame, pts[c.start], pts[c.end], color, 2)
        for p in pts:
            cv2.circle(frame, p, 3, (255, 255, 255), -1)
            cv2.circle(frame, p, 3, color, 1)


def draw_hud(
    display: np.ndarray,
    label: str,
    saved: int,
    blocks: dict[str, bool],
    recording: bool,
    frame_no: int,
) -> None:
    """Draw text overlays. Called AFTER the mirror flip so text reads normally."""
    h, w = display.shape[:2]
    cv2.rectangle(display, (0, 0), (w, 64), (0, 0, 0), -1)

    cv2.putText(
        display, f"label: {label}    saved: {saved}",
        (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
    )

    x = 12
    for name in HAND_ORDER:
        on = blocks[name]
        cv2.putText(
            display, f"[{'x' if on else ' '}] {name}",
            (x, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            HAND_COLORS[name] if on else (110, 110, 110), 2,
        )
        x += 130

    if recording:
        cv2.putText(
            display, f"RECORDING  frame {frame_no}/{SEQUENCE_LENGTH}",
            (x + 20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, REC_COLOR, 2,
        )
        cv2.rectangle(display, (0, 0), (w - 1, h - 1), REC_COLOR, 6)
        bar = int(w * frame_no / SEQUENCE_LENGTH)
        cv2.rectangle(display, (0, h - 10), (bar, h), REC_COLOR, -1)
    else:
        cv2.putText(
            display, "SPACE = record    Q = quit",
            (x + 20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, IDLE_COLOR, 2,
        )


def ensure_model(path: Path) -> Path:
    """Download the HandLandmarker bundle on first run."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"HandLandmarker model not found at {path}")
    print(f"Downloading from {MODEL_URL} ...")
    urllib.request.urlretrieve(MODEL_URL, path)
    print(f"Saved {path.stat().st_size / 1e6:.1f} MB\n")
    return path


def open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        sys.exit(
            f"Could not open camera {index}.\n"
            "  - try a different index: --camera 1\n"
            "  - on macOS, grant camera access to your terminal in\n"
            "    System Settings > Privacy & Security > Camera"
        )
    return cap


def print_instructions(label: str, out_dir: Path, saved: int) -> None:
    print("=" * 70)
    print("  Vox - ISL landmark collection")
    print("=" * 70)
    print(f"  label            : {label}")
    print(f"  writing to       : {out_dir}")
    print(f"  already recorded : {saved} sample(s)")
    print(f"  sample shape     : ({SEQUENCE_LENGTH}, {FEATURE_DIM})  raw, un-normalized")
    print()
    print("  SPACE   record one sample (30 frames, no countdown - be in position)")
    print("  Q/ESC   quit")
    print()
    print("  The preview is mirrored for comfort; the detector sees the raw frame.")
    print("  Vary distance, lighting and position between samples.")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect ISL landmark sequences.")
    parser.add_argument("label", nargs="?", help="sign label, e.g. hello")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    label = args.label or input("Sign label: ").strip()
    if not label:
        sys.exit("A label is required.")

    out_dir = args.data_dir / label
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = len(list(out_dir.glob("*.npy")))

    model_path = ensure_model(args.model)
    print_instructions(label, out_dir, saved)

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )

    cap = open_camera(args.camera)
    window = f"Vox collect - {label}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    recording = False
    buffer: list[np.ndarray] = []
    timestamp_ms = 0  # must increase strictly for RunningMode.VIDEO

    try:
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Dropped frame from camera, retrying...")
                    continue

                # Detect on the RAW frame — see contract note (1).
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms
                )
                timestamp_ms += 33  # ~30 fps; only monotonicity matters

                vec = build_frame_vector(result)
                if recording:
                    buffer.append(vec)

                draw_landmarks(frame, result)
                display = cv2.flip(frame, 1)  # mirror for display only
                draw_hud(
                    display, label, saved, filled_blocks(vec), recording, len(buffer)
                )
                cv2.imshow(window, display)

                if recording and len(buffer) == SEQUENCE_LENGTH:
                    sample = np.stack(buffer)
                    assert sample.shape == (SEQUENCE_LENGTH, FEATURE_DIM), sample.shape
                    path = out_dir / f"{uuid.uuid4().hex}.npy"
                    np.save(path, sample)
                    saved += 1
                    empty = int((~sample.any(axis=1)).sum())
                    note = f"  ({empty} empty frames - re-record?)" if empty else ""
                    print(f"[{label}] saved {saved:>3}  {path.name}  {sample.shape}{note}")
                    recording = False
                    buffer = []

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" ") and not recording:
                    recording = True
                    buffer = []
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\nDone. {saved} sample(s) for '{label}' in {out_dir}")


if __name__ == "__main__":
    main()

"""Extract reference 3D motion from the ISL clips for the in-app avatar.

Writes frontend/public/clips/motion.json: {word: [[141 floats] x T]} at 15 fps,
the same landmarks the recognizer consumes. The Speech -> ISL panel plays this
next to the video so a learner can see the sign as motion, not just footage.

    python ml/clip_motion.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

ML_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ML_DIR))
from collect import (  # noqa: E402
    DEFAULT_MODEL_PATH, DEFAULT_POSE_MODEL_PATH,
    build_frame_vector, ensure_model, ensure_pose_model,
)

CLIPS_DIR = ML_DIR.parent / "frontend" / "public" / "clips"
MAX_FRAMES = 90  # ~6 s at 15 fps; keeps motion.json small


def main() -> None:
    manifest = json.loads((CLIPS_DIR / "manifest.json").read_text())["clips"]
    hand_model = ensure_model(DEFAULT_MODEL_PATH)
    pose_model = ensure_pose_model(DEFAULT_POSE_MODEL_PATH)

    hand_opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(hand_model)),
        running_mode=vision.RunningMode.VIDEO, num_hands=2)
    pose_opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(pose_model)),
        running_mode=vision.RunningMode.VIDEO, num_poses=1)

    motion: dict[str, list[list[float]]] = {}
    for word, filename in sorted(manifest.items()):
        path = CLIPS_DIR / filename
        if not path.exists():
            print(f"  {word:<8} missing {filename}")
            continue
        cap = cv2.VideoCapture(str(path))
        frames: list[list[float]] = []
        with vision.HandLandmarker.create_from_options(hand_opts) as hands, \
             vision.PoseLandmarker.create_from_options(pose_opts) as pose:
            ts = 0
            while len(frames) < MAX_FRAMES:
                ok, frame = cap.read()
                if not ok:
                    break
                image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                 data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                vec = build_frame_vector(hands.detect_for_video(image, ts),
                                         pose.detect_for_video(image, ts))
                ts += 67
                frames.append([round(float(v), 4) for v in vec])
        cap.release()
        motion[word] = frames
        print(f"  {word:<8} {len(frames):>3} frames")

    out = CLIPS_DIR / "motion.json"
    out.write_text(json.dumps(motion, separators=(",", ":")))
    print(f"\nwrote {out} ({out.stat().st_size/1e3:.0f} KB)")


if __name__ == "__main__":
    main()

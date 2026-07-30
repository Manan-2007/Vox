"""Extract training samples from sign-language video files.

Turns dictionary-style word videos (one sign per clip, e.g. ISLRTC / Goa Board
ISL dictionaries) into the same raw .npy sequences ml/collect.py records, so
preprocess.py and train.py work unchanged:

    ml/data/<label>/<video>_<start>.npy   shape (30, 126), raw landmarks

Videos are named <label>__<anything>.mp4 and read from --videos-dir.

How a video becomes samples:
  1. Landmarks are computed on frames resampled to ~15 FPS — the same rate the
     browser streams at, so temporal density matches inference.
  2. The signing segment is found (frames where a hand is visible), trimming
     title cards and rest position at either end.
  3. A 30-frame window slides over the segment (stride 2). Windows where hands
     are visible in fewer than 70% of frames are dropped.
  4. A segment shorter than 30 frames is time-stretched to 30 by linear
     interpolation instead of being thrown away — dictionary clips are short.

Usage:
    python ml/extract.py --videos-dir path/to/videos
    python ml/extract.py --videos-dir vids --data-dir ml/data --stride 2
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    DEFAULT_POSE_MODEL_PATH,
    FEATURE_DIM,
    SEQUENCE_LENGTH,
    aspect_scale,
    build_frame_vector,
    ensure_model,
    ensure_pose_model,
)

ML_DIR = Path(__file__).resolve().parent
TARGET_FPS = 15.0
MIN_HAND_RATIO = 0.7  # a window must have hands in at least this share of frames


def video_to_frames(path: Path, landmarker, pose_landmarker) -> np.ndarray:
    """Run hand + pose landmarks over a video at ~15 FPS. Returns (T, 141) raw."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps / TARGET_FPS))
    # Non-16:9 sources are mapped onto the contract's aspect — see collect.py.
    x_scale = aspect_scale(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    vectors: list[np.ndarray] = []
    frame_index = 0
    timestamp_ms = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % step == 0:
            # 1080p source clips: downscale before detection. MediaPipe works
            # on normalized coordinates, so this changes nothing downstream
            # and is several times faster.
            if frame.shape[1] > 960:
                scale = 960 / frame.shape[1]
                frame = cv2.resize(frame, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(image, timestamp_ms)
            pose_result = pose_landmarker.detect_for_video(image, timestamp_ms)
            timestamp_ms += 67  # ~15 FPS spacing; only monotonicity matters
            vectors.append(build_frame_vector(result, pose_result, x_scale))
        frame_index += 1
    cap.release()

    if not vectors:
        return np.zeros((0, FEATURE_DIM), np.float32)
    return np.stack(vectors)


def signing_segment(frames: np.ndarray) -> tuple[int, int]:
    """[start, end) span from first to last frame with any HAND visible.

    Hands only: the pose block is filled in almost every frame (the signer is
    always on camera), so including it would never trim anything.
    """
    present = frames[:, :126].any(axis=1)
    if not present.any():
        return 0, 0
    indices = np.flatnonzero(present)
    return int(indices[0]), int(indices[-1]) + 1


def stretch_to(frames: np.ndarray, length: int) -> np.ndarray:
    """Linearly time-stretch a (T, 126) segment to `length` frames."""
    src = np.linspace(0, len(frames) - 1, length)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, len(frames) - 1)
    frac = (src - lo)[:, None]
    out = frames[lo] * (1 - frac) + frames[hi] * frac
    # interpolation between an empty and a hand frame is neither — keep hard
    # emptiness from the nearer source frame
    nearest = np.where(frac[:, 0] < 0.5, lo, hi)
    out[~frames[:, :126].any(axis=1)[nearest], :126] = 0.0
    return out.astype(np.float32)


def evenly_sample(items: list, cap: int) -> list:
    """At most `cap` items, evenly spaced — keeps clip coverage without letting
    a long video contribute 20x more (near-identical) windows than a short one."""
    if cap <= 0 or len(items) <= cap:
        return items
    idx = np.linspace(0, len(items) - 1, cap).round().astype(int)
    return [items[i] for i in sorted(set(idx.tolist()))]


def rest_windows(frames: np.ndarray, stride: int) -> list[np.ndarray]:
    """Windows from OUTSIDE the signing segment — hands down, walking in,
    adjusting, transitions.

    Without a class for "not a sign", a 6-way classifier must force every
    gesture into one of its 6 words: idle hands become a confident wrong
    answer. Training on the footage either side of the sign gives the model an
    honest "nothing is happening" option, which is what makes the live system
    stay quiet instead of guessing.
    """
    start, end = signing_segment(frames)
    out = []
    for lo, hi in ((0, start), (end, len(frames))):
        region = frames[lo:hi]
        if len(region) < SEQUENCE_LENGTH:
            continue
        for offset in range(0, len(region) - SEQUENCE_LENGTH + 1, stride):
            window = region[offset : offset + SEQUENCE_LENGTH]
            # keep it only if a BODY is visible — a black frame teaches nothing
            if window[:, 126:].any(axis=1).mean() >= 0.7:
                out.append(window.astype(np.float32))
    return out


def windows_from(frames: np.ndarray, stride: int) -> list[np.ndarray]:
    """All acceptable 30-frame windows over the signing segment."""
    start, end = signing_segment(frames)
    segment = frames[start:end]
    if len(segment) == 0:
        return []

    if len(segment) < SEQUENCE_LENGTH:
        if len(segment) < SEQUENCE_LENGTH // 3:
            return []  # too little signing to stretch honestly
        return [stretch_to(segment, SEQUENCE_LENGTH)]

    out = []
    for offset in range(0, len(segment) - SEQUENCE_LENGTH + 1, stride):
        window = segment[offset : offset + SEQUENCE_LENGTH]
        if window[:, :126].any(axis=1).mean() >= MIN_HAND_RATIO:
            out.append(window.astype(np.float32))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract samples from sign videos.")
    parser.add_argument("--videos-dir", type=Path, required=True,
                        help="directory of <label>__<id>.mp4 files")
    parser.add_argument("--data-dir", type=Path, default=ML_DIR / "data")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-per-video", type=int, default=10,
                        help="cap windows kept per video (0 = no cap)")
    parser.add_argument("--rest-label", default=None, metavar="NAME",
                        help="also emit a rejection class (e.g. 'rest') from the "
                             "non-signing parts of each video")
    parser.add_argument("--max-rest-per-video", type=int, default=2)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    videos = sorted(
        v for pattern in ("*.mp4", "*.webm", "*.mov", "*.MOV", "*.avi")
        for v in args.videos_dir.glob(pattern)
    )
    if not videos:
        sys.exit(f"no videos in {args.videos_dir} (expected <label>__<id>.mp4)")

    model_path = ensure_model(args.model)
    pose_model_path = ensure_pose_model(DEFAULT_POSE_MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    pose_options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(pose_model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )

    counts: dict[str, int] = defaultdict(int)
    for video in videos:
        label = video.stem.split("__")[0]
        if not label:
            print(f"  skip {video.name}: no label prefix")
            continue

        # One landmarker per video: VIDEO mode carries tracking state between
        # frames, which must not leak from one clip into the next.
        with vision.HandLandmarker.create_from_options(options) as landmarker, \
             vision.PoseLandmarker.create_from_options(pose_options) as pose_landmarker:
            frames = video_to_frames(video, landmarker, pose_landmarker)

        start, end = signing_segment(frames)
        samples = evenly_sample(windows_from(frames, args.stride), args.max_per_video)

        if args.rest_label:
            rest = evenly_sample(
                rest_windows(frames, args.stride), args.max_rest_per_video
            )
            if rest:
                rest_dir = args.data_dir / args.rest_label
                rest_dir.mkdir(parents=True, exist_ok=True)
                stem_r = video.stem.split("__")[-1]
                for i, sample in enumerate(rest):
                    np.save(rest_dir / f"{label}-{stem_r}_{i:03d}.npy", sample)
                counts[args.rest_label] += len(rest)

        out_dir = args.data_dir / label
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = video.stem.split("__")[-1]
        for i, sample in enumerate(samples):
            assert sample.shape == (SEQUENCE_LENGTH, FEATURE_DIM)
            np.save(out_dir / f"{stem}_{i:03d}.npy", sample)
        counts[label] += len(samples)

        print(
            f"  {video.name:<28} {len(frames):>4} frames @15fps, "
            f"signing {end - start:>3}, -> {len(samples):>3} sample(s)"
        )

    print("\nPer-label totals (this run):")
    for label in sorted(counts):
        print(f"  {label:<10} {counts[label]:>4}")
    print(f"\nWrote into {args.data_dir}. Next: python ml/preprocess.py && python ml/train.py")


if __name__ == "__main__":
    main()

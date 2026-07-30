"""Build the avatar's sign motion library from the dictionary clips.

For every gloss in ml/vocabulary.py that has a downloaded clip, this picks the
cleanest recording, trims it to the signing itself, and writes the motion the 3D
avatar plays back:

    frontend/public/signs/<gloss>.json     one file per sign, fetched on demand
    frontend/public/signs/manifest.json    what exists, and how long each is

    python ml/build_motion.py

--------------------------------------------------------------------------
WHY THIS STORES TWO DIFFERENT KINDS OF COORDINATE
--------------------------------------------------------------------------
The recogniser reads normalized *image* coordinates, because where a sign happens
relative to the body is most of its meaning (see ml/normalize.py). Those
coordinates are a bad basis for drawing a hand: their z is a weak per-landmark
guess, so a hand built from them is flat.

MediaPipe also returns `hand_world_landmarks` — 21 points in metres, with real
depth and real proportions, oriented roughly to the camera. Those are excellent
for *shape* and useless for *placement*, since the origin is the hand's own
centre and all body context is gone.

So each frame stores both, and the avatar composes them: the hand's shape and
orientation come from the world landmarks, and the whole hand is then placed in
signing space using the image-space wrist and the shoulder anchor. That is what
makes the avatar read as a hand in a body rather than a constellation of dots.

Frame layout, 140 floats:

    [  0 :  2 ]  left  wrist, image space (x, y) — 0,0 means the hand is absent
    [  2 : 65 ]  left  hand, 21 world landmarks (x, y, z) in metres
    [ 65 : 67 ]  right wrist, image space
    [ 67 :130 ]  right hand, 21 world landmarks
    [130 :140 ]  pose, image space (x, y) for nose, L/R shoulder, L/R elbow
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

ML_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ML_DIR))
from collect import (  # noqa: E402
    DEFAULT_MODEL_PATH, DEFAULT_POSE_MODEL_PATH, ensure_model, ensure_pose_model,
)
from vocabulary import GLOSSES, POS_BY_GLOSS, preferred_english  # noqa: E402

OUT_DIR = ML_DIR.parent / "frontend" / "public" / "signs"
TARGET_FPS = 15
#: Detection thresholds. Deliberately the same numbers the browser worker uses —
#: the avatar should be built from the same view of the world the live tracker has.
CONFIDENCE = 0.3
#: Pad the trimmed segment so a sign does not start mid-movement.
PAD_FRAMES = 2
#: A sign longer than this is almost certainly a phrase or an example sentence.
MAX_FRAMES = 75
POSE_INDICES = (0, 11, 12, 13, 14)

FRAME_FLOATS = 140
LEFT_WRIST, LEFT_WORLD = 0, 2
RIGHT_WRIST, RIGHT_WORLD = 65, 67
POSE_BLOCK = 130


def clip_motion(path: Path, hands, pose) -> np.ndarray:
    """Run both landmarkers over a clip at ~15 fps. Returns (T, 140).

    Image-space x is multiplied by the frame's aspect ratio before being stored,
    which turns MediaPipe's normalized coordinates into SQUARE ones.

    This matters more than it sounds. MediaPipe divides x by the frame width and
    y by the frame height, so in a 16:9 clip one unit of x is 1.78x as many
    pixels as one unit of y. Any distance computed across both axes — the
    shoulder width that everything here is measured in — is then wrong, and
    wrong in a direction that stretches the whole figure vertically: measured
    that way the nose sat 1.2 shoulder widths above the shoulders instead of the
    0.67 a real body has, and the avatar's head floated a head's height clear of
    its neck. Correcting x once, here, makes every downstream ratio true.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return np.zeros((0, FRAME_FLOATS), np.float32)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(fps / TARGET_FPS))
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 16.0
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 9.0
    aspect = float(frame_width) / float(frame_height) if frame_height else 1.0

    rows: list[np.ndarray] = []
    index = 0
    timestamp = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            hand_result = hands.detect_for_video(image, timestamp)
            pose_result = pose.detect_for_video(image, timestamp)
            timestamp += 67
            rows.append(_row(hand_result, pose_result, aspect))
        index += 1
    cap.release()

    if not rows:
        return np.zeros((0, FRAME_FLOATS), np.float32)
    return np.stack(rows)


def _row(hand_result, pose_result, aspect: float) -> np.ndarray:
    out = np.zeros(FRAME_FLOATS, np.float32)

    if pose_result and pose_result.pose_landmarks:
        landmarks = pose_result.pose_landmarks[0]
        for slot, index in enumerate(POSE_INDICES):
            if index < len(landmarks):
                out[POSE_BLOCK + slot * 2] = landmarks[index].x * aspect
                out[POSE_BLOCK + slot * 2 + 1] = landmarks[index].y

    if not hand_result or not hand_result.hand_landmarks:
        return out

    # One detection per side, best score wins — same rule as build_frame_vector.
    best: dict[str, tuple[float, int]] = {}
    for i, categories in enumerate(hand_result.handedness):
        if not categories:
            continue
        label, score = categories[0].category_name, categories[0].score
        if label in ("Left", "Right") and (label not in best or score > best[label][0]):
            best[label] = (score, i)

    for label, (wrist_slot, world_slot) in (
        ("Left", (LEFT_WRIST, LEFT_WORLD)),
        ("Right", (RIGHT_WRIST, RIGHT_WORLD)),
    ):
        if label not in best:
            continue
        i = best[label][1]
        image = hand_result.hand_landmarks[i]
        world = hand_result.hand_world_landmarks[i]
        # A wrist at exactly (0, 0) would read as "absent"; nudge it. This is a
        # sub-pixel move at the very edge of frame and never happens in practice.
        out[wrist_slot] = (image[0].x * aspect) or 1e-5
        out[wrist_slot + 1] = image[0].y or 1e-5
        for j, point in enumerate(world[:21]):
            base = world_slot + j * 3
            out[base] = point.x
            out[base + 1] = point.y
            out[base + 2] = point.z

    return out


def hands_present(frames: np.ndarray) -> np.ndarray:
    """Per frame: how many hands were resolved (0-2)."""
    left = (frames[:, LEFT_WRIST] != 0) | (frames[:, LEFT_WRIST + 1] != 0)
    right = (frames[:, RIGHT_WRIST] != 0) | (frames[:, RIGHT_WRIST + 1] != 0)
    return left.astype(int) + right.astype(int)


#: A wrist this far below the shoulder line, in shoulder widths, counts as
#: "hanging at the side" rather than signing. Measured from the clips once the
#: coordinates are square (see clip_motion): a resting wrist sits about 1.35
#: shoulder widths below the shoulder line, and the lowest real signs in this
#: vocabulary — the ones made at the waist — reach about 0.95.
RESTING_DEPTH = 1.05


def in_signing_space(frames: np.ndarray) -> np.ndarray:
    """Per frame: is either hand raised into the signing space?

    Trimming on hand *detection* alone is not enough. The signer stands in shot
    with their arms down for a second or more at each end of every dictionary
    clip, and their hands are detected perfectly well down there. Those frames
    are not the sign; keeping them makes the avatar spend most of its animation
    standing still with its arms by its sides, and — because the camera has to
    frame whatever the clip contains — shrinks the actual sign to the top third
    of the stage.

    So the test is positional: is a wrist above the resting line, measured in
    shoulder widths so it holds however far the signer is from the camera.
    """
    left_shoulder = frames[:, POSE_BLOCK + 2 : POSE_BLOCK + 4]
    right_shoulder = frames[:, POSE_BLOCK + 4 : POSE_BLOCK + 6]
    width = np.hypot(
        left_shoulder[:, 0] - right_shoulder[:, 0],
        left_shoulder[:, 1] - right_shoulder[:, 1],
    )
    mid_y = (left_shoulder[:, 1] + right_shoulder[:, 1]) / 2
    # Image y grows downward, so "raised" is a SMALLER y.
    limit = mid_y + RESTING_DEPTH * width

    raised = np.zeros(len(frames), bool)
    for wrist_slot in (LEFT_WRIST, RIGHT_WRIST):
        present = (frames[:, wrist_slot] != 0) | (frames[:, wrist_slot + 1] != 0)
        raised |= present & (width > 1e-6) & (frames[:, wrist_slot + 1] < limit)
    return raised


def trim(frames: np.ndarray) -> np.ndarray:
    """Cut down to the sign itself — hands up, doing something."""
    raised = in_signing_space(frames)
    if not raised.any():
        # No frame clears the resting line. Rather than drop the clip, fall back
        # to any frame with a hand: some signs really are made low.
        raised = hands_present(frames) > 0
        if not raised.any():
            return frames[:0]
    indices = np.flatnonzero(raised)
    start = max(0, int(indices[0]) - PAD_FRAMES)
    end = min(len(frames), int(indices[-1]) + 1 + PAD_FRAMES)
    return frames[start:end]


def fill_gaps(frames: np.ndarray) -> np.ndarray:
    """Interpolate across single-frame hand dropouts.

    A hand missed for one or two frames mid-sign leaves a hole that the avatar
    would render as the hand blinking out of existence. Interpolating between the
    surrounding frames is both truer to what happened and far less distracting.
    A longer gap is left alone: the hand really was out of view, and inventing
    half a second of motion would be a lie.
    """
    out = frames.copy()
    for wrist_slot, world_slot in ((LEFT_WRIST, LEFT_WORLD), (RIGHT_WRIST, RIGHT_WORLD)):
        present = (out[:, wrist_slot] != 0) | (out[:, wrist_slot + 1] != 0)
        columns = [wrist_slot, wrist_slot + 1] + list(range(world_slot, world_slot + 63))
        index = 0
        while index < len(out):
            if present[index]:
                index += 1
                continue
            gap_start = index
            while index < len(out) and not present[index]:
                index += 1
            gap_end = index  # exclusive
            if gap_start == 0 or gap_end == len(out):
                continue  # nothing to interpolate between
            if gap_end - gap_start > 2:
                continue  # a real absence, not a dropout
            before = out[gap_start - 1, columns]
            after = out[gap_end, columns]
            for k, position in enumerate(range(gap_start, gap_end), start=1):
                t = k / (gap_end - gap_start + 1)
                out[position, columns] = before * (1 - t) + after * t
    return out


def score(frames: np.ndarray) -> float:
    """How good a recording this is, for picking between clips of one word.

    Prefers clips where hands are tracked for most of the signing and which are
    not suspiciously short. Length is capped in the score so a clip that happens
    to include an example sentence does not beat a clean isolated sign.
    """
    if len(frames) < 6:
        return 0.0
    present = hands_present(frames)
    tracked = float((present > 0).mean())
    body = float((frames[:, POSE_BLOCK + 2] != 0).mean())

    # An isolated sign runs roughly 1-2.5 seconds. Score peaks in that band and
    # falls away on both sides: a very short clip is a fragment, and a very long
    # one is almost always the dictionary demonstrating the word inside an
    # example sentence, which is the wrong thing to teach as "the sign".
    seconds = len(frames) / TARGET_FPS
    if seconds < 0.7:
        length = seconds / 0.7
    elif seconds <= 2.5:
        length = 1.0
    else:
        length = max(0.0, 1.0 - (seconds - 2.5) / 3.0)

    return tracked * 0.5 + body * 0.15 + length * 0.35


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--videos-dir", type=Path, default=ML_DIR / "videos")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--only", nargs="*", help="rebuild only these glosses")
    args = parser.parse_args()

    by_gloss: dict[str, list[Path]] = defaultdict(list)
    for pattern in ("*.mp4", "*.webm", "*.mov", "*.MOV"):
        for video in sorted(args.videos_dir.glob(pattern)):
            by_gloss[video.stem.split("__")[0]].append(video)
    if not by_gloss:
        sys.exit(f"no clips in {args.videos_dir} — run ml/fetch_dictionary.py --download")

    hand_model = ensure_model(DEFAULT_MODEL_PATH)
    pose_model = ensure_pose_model(DEFAULT_POSE_MODEL_PATH)
    hand_opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(hand_model)),
        running_mode=vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=CONFIDENCE,
        min_hand_presence_confidence=CONFIDENCE,
        min_tracking_confidence=CONFIDENCE)
    pose_opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(pose_model)),
        running_mode=vision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=CONFIDENCE,
        min_pose_presence_confidence=CONFIDENCE,
        min_tracking_confidence=CONFIDENCE)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    skipped: list[str] = []

    # Normally only the curated vocabulary is built. An explicit --only may name
    # glosses that are not in it, which is how ml/add_words.py extends the
    # library without editing vocabulary.py.
    if args.only:
        targets = [g for g in args.only if g in by_gloss]
        unknown = [g for g in args.only if g not in by_gloss]
        if unknown:
            print(f"no clips downloaded for: {', '.join(unknown)}")
    else:
        targets = [g for g in GLOSSES if g in by_gloss]

    for n, gloss in enumerate(targets, 1):
        best: np.ndarray | None = None
        best_score = 0.0
        for video in by_gloss[gloss]:
            # One landmarker per clip: VIDEO mode carries tracking state between
            # frames and it must not leak from one recording into the next.
            with vision.HandLandmarker.create_from_options(hand_opts) as hands, \
                 vision.PoseLandmarker.create_from_options(pose_opts) as pose:
                frames = clip_motion(video, hands, pose)
            frames = fill_gaps(trim(frames))[:MAX_FRAMES]
            value = score(frames)
            if value > best_score:
                best, best_score = frames, value

        if best is None or len(best) < 6:
            skipped.append(gloss)
            print(f"  {n:>3}/{len(targets)} {gloss:<16} no usable clip")
            continue

        payload = {
            "gloss": gloss,
            "english": preferred_english(gloss),
            "pos": POS_BY_GLOSS.get(gloss, "noun"),
            "fps": TARGET_FPS,
            "frames": [[round(float(v), 4) for v in row] for row in best],
        }
        (args.out_dir / f"{gloss}.json").write_text(
            json.dumps(payload, separators=(",", ":")), encoding="utf-8"
        )
        manifest[gloss] = {
            "english": preferred_english(gloss),
            "pos": POS_BY_GLOSS.get(gloss, "noun"),
            "frames": len(best),
            "seconds": round(len(best) / TARGET_FPS, 2),
            "twoHanded": bool((hands_present(best) == 2).mean() > 0.5),
            "quality": round(best_score, 3),
        }
        print(f"  {n:>3}/{len(targets)} {gloss:<16} {len(best):>3} frames  "
              f"q={best_score:.2f}", flush=True)

    (args.out_dir / "manifest.json").write_text(json.dumps({
        "_source": "ISLRTC official Indian Sign Language dictionary "
                   "(Government of India). Landmarks only — no video is shipped.",
        "fps": TARGET_FPS,
        "signs": dict(sorted(manifest.items())),
    }, indent=1), encoding="utf-8")

    total = sum(
        (args.out_dir / f"{g}.json").stat().st_size for g in manifest
    )
    print(f"\n{len(manifest)} signs -> {args.out_dir} ({total/1e6:.1f} MB total, "
          f"loaded one at a time)")
    if skipped:
        print(f"no usable clip for {len(skipped)}: {', '.join(skipped)}")


if __name__ == "__main__":
    main()

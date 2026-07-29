"""The single shared normalization function for Vox.

THIS MODULE IS THE CONTRACT. `preprocess.py` (training) and the FastAPI backend
(inference) both import `normalize_frame` from here, and the browser mirrors the
frame layout in frontend/src/landmarks.ts. If normalization ever diverges,
predictions become garbage — so there is exactly one implementation.

--------------------------------------------------------------------------
WHY THIS IS BODY-ANCHORED (and what the first version got wrong)
--------------------------------------------------------------------------
v1 translated each hand so its own wrist sat at the origin, then scaled by hand
size. That made a hand position- and scale-invariant — which sounded right and
was in fact catastrophic: it pinned the wrist to (0,0) in EVERY frame, so both
WHERE the hand was and HOW IT MOVED were erased. Only finger shape survived.
For ISL that is exactly backwards: "eat" (hand to mouth), "please" (circle on
the chest) and "hello" (wave beside the head) share similar handshapes and
differ almost entirely by location and motion. The model was left guessing and
collapsed onto one class.

v2 anchors on the BODY instead of the hand:
    origin = midpoint between the shoulders
    unit   = distance between the shoulders
Every landmark is expressed in that frame. This keeps location and trajectory
(the discriminative signal) while still being invariant to where the signer
stands, how far they are from the camera, and their size — because the shoulder
span scales with all of those in the same way. This is the standard
"normalize into the signing space" treatment in the sign-recognition
literature.

--------------------------------------------------------------------------
FRAME LAYOUT (141 floats) — raw, as produced by collect.py / extract.py /
frontend/src/landmarks.ts
--------------------------------------------------------------------------
    [  0 : 63 ]  Left  hand: 21 landmarks x (x, y, z)
    [ 63 :126 ]  Right hand: 21 landmarks x (x, y, z)
    [126 :141 ]  Pose: nose, L shoulder, R shoulder, L elbow, R elbow (5 x xyz)

x, y are MediaPipe's image-normalized coordinates. An absent hand is 63 zeros;
absent pose is 15 zeros.

Output has the same shape. Per frame:
  - No pose, or degenerate shoulders -> the WHOLE frame is zeroed. There is no
    anchor, so nothing can be placed in signing space; the model's Masking
    layer then skips the frame, which is the honest thing to do.
  - x, y of every present point -> (p - shoulder_mid) / shoulder_width.
  - z is kept hand-local: (z - own wrist z) / hand size. MediaPipe's hand z and
    pose z use different references, so mixing them into one anchor would be
    meaningless; keeping z hand-relative preserves it as a pure shape cue.
    Pose z is dropped (set to 0) for the same reason.
"""

from __future__ import annotations

import numpy as np

# --- frame contract (mirrors ml/collect.py and frontend/src/landmarks.ts) ----
LANDMARKS_PER_HAND = 21
COORDS_PER_LANDMARK = 3
FEATURES_PER_HAND = LANDMARKS_PER_HAND * COORDS_PER_LANDMARK  # 63
HANDS_DIM = FEATURES_PER_HAND * 2  # 126

# Pose points kept, in this order, from MediaPipe's pose landmark indices.
POSE_LANDMARK_INDICES = (0, 11, 12, 13, 14)  # nose, L/R shoulder, L/R elbow
POSE_POINTS = len(POSE_LANDMARK_INDICES)
POSE_DIM = POSE_POINTS * COORDS_PER_LANDMARK  # 15

FEATURE_DIM = HANDS_DIM + POSE_DIM  # 141

# Frames per sample: 30 at 15 FPS = 2 s. Signs here run 1.9-3.1 s, so a window
# is often a *partial* sign — which is deliberate, because that is exactly what
# the backend sees as it slides a window over a live stream. Widening to 45 was
# tried and measured worse (70.8% vs 93.2% unseen-signer accuracy): it forced
# one sample per video, halving the training set, and did not resolve the
# time-of-day confusions it was meant to fix. Training, the backend deque and
# the browser's buffering all read this one value.
SEQUENCE_LENGTH = 30

WRIST = 0
MIDDLE_FINGER_MCP = 9
# offsets within the pose block
POSE_NOSE, POSE_L_SHOULDER, POSE_R_SHOULDER = 0, 1, 2

MIN_SHOULDER_WIDTH = 1e-6  # below this there is no usable anchor
MIN_HAND_SIZE = 1e-8


def _hand_z(points: np.ndarray) -> np.ndarray:
    """Hand-local depth: z relative to the wrist, scaled by hand size."""
    z = points[:, 2] - points[WRIST, 2]
    size = float(np.linalg.norm(points[MIDDLE_FINGER_MCP, :2] - points[WRIST, :2]))
    if size < MIN_HAND_SIZE:
        return np.zeros_like(z)
    return z / size


def normalize_frame(vec: np.ndarray) -> np.ndarray:
    """Normalize one 141-float frame into shoulder-anchored signing space.

    Returns a new array; the input is never modified. See the module docstring
    for the layout and the reasoning.
    """
    vec = np.asarray(vec)
    if vec.shape != (FEATURE_DIM,):
        raise ValueError(f"expected ({FEATURE_DIM},), got {vec.shape}")

    out = np.zeros(FEATURE_DIM, dtype=np.float32)

    pose = vec[HANDS_DIM:].reshape(POSE_POINTS, COORDS_PER_LANDMARK).astype(np.float32)
    if not pose.any():
        return out  # no body reference -> unusable frame, emit zeros

    left_shoulder = pose[POSE_L_SHOULDER, :2]
    right_shoulder = pose[POSE_R_SHOULDER, :2]
    width = float(np.linalg.norm(left_shoulder - right_shoulder))
    if width < MIN_SHOULDER_WIDTH:
        return out  # degenerate anchor (e.g. signer turned side-on)

    origin = (left_shoulder + right_shoulder) / 2.0

    for block in range(2):
        lo = block * FEATURES_PER_HAND
        hand = vec[lo : lo + FEATURES_PER_HAND]
        if not hand.any():
            continue  # absent hand keeps its 63 zeros
        points = hand.reshape(LANDMARKS_PER_HAND, COORDS_PER_LANDMARK).astype(np.float32)
        placed = np.empty_like(points)
        placed[:, :2] = (points[:, :2] - origin) / width
        placed[:, 2] = _hand_z(points)
        out[lo : lo + FEATURES_PER_HAND] = placed.reshape(FEATURES_PER_HAND)

    placed_pose = np.zeros_like(pose)
    present = pose.any(axis=1)
    placed_pose[present, :2] = (pose[present, :2] - origin) / width
    # pose z uses a different reference than hand z; dropped rather than mixed
    out[HANDS_DIM:] = placed_pose.reshape(POSE_DIM)

    return out


def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    """Apply `normalize_frame` to every frame of a (T, 141) sequence.

    Convenience only — the math lives in `normalize_frame` and is not duplicated.
    """
    seq = np.asarray(seq)
    if seq.ndim != 2 or seq.shape[1] != FEATURE_DIM:
        raise ValueError(f"expected (T, {FEATURE_DIM}), got {seq.shape}")
    return np.stack([normalize_frame(f) for f in seq]).astype(np.float32)

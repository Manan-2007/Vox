"""The single shared normalization function for Vox.

THIS MODULE IS THE CONTRACT. `preprocess.py` (training) and the FastAPI backend
(inference) both import `normalize_frame` from here. If normalization ever
diverges between the two, predictions become garbage — so there is exactly one
implementation of the math, and neither side is allowed a local copy.

Input and output are both the raw 126-float frame vector produced by
`ml/collect.py`:

    [  0 : 63 ]  Left  hand: 21 landmarks x (x, y, z)
    [ 63 :126 ]  Right hand: 21 landmarks x (x, y, z)

Per hand block, independently:
  - all zeros (hand absent)  -> left as zeros, so "absent" stays distinguishable
  - otherwise                -> translate so the wrist (landmark 0) is the
                                origin, then divide every coordinate by the
                                wrist -> middle-finger-MCP (landmark 9) distance

That makes a hand invariant to where it sits in the frame and how close it is to
the camera, while preserving its shape. The two hands are scaled independently,
so their relative size is not preserved — this is deliberate: it keeps a hand's
meaning stable regardless of which one happens to be nearer the lens.

Two caveats worth knowing:

  - The reference distance is measured in the x,y plane only (see
    `USE_Z_IN_REFERENCE`). MediaPipe's z is relative depth and is by far the
    noisiest channel; putting it in the denominator would let that noise
    modulate all 63 values of the hand every frame. z is still translated and
    scaled like x and y — it just doesn't get a vote in the scale factor.
  - MediaPipe normalizes x by image width and y by image height, so on a
    non-square frame the input space is already anisotropic. That cancels out as
    long as collection and inference use the same aspect ratio. Nothing here can
    correct for it, because the frame vector carries no image dimensions.
"""

from __future__ import annotations

import numpy as np

# --- frame contract (mirrors ml/collect.py) ----------------------------------
LANDMARKS_PER_HAND = 21
COORDS_PER_LANDMARK = 3
FEATURES_PER_HAND = LANDMARKS_PER_HAND * COORDS_PER_LANDMARK  # 63
FEATURE_DIM = FEATURES_PER_HAND * 2  # 126

WRIST = 0
MIDDLE_FINGER_MCP = 9

# Measure the reference distance in x,y only. Flip to True to include z — but
# flip it for training AND inference together, or the two stop agreeing.
USE_Z_IN_REFERENCE = False

# Below this, wrist and MCP are effectively the same point: the detection is
# degenerate and the hand is emitted as absent rather than blown up by a
# division that would swamp every real sample.
MIN_REFERENCE_DISTANCE = 1e-8


def normalize_hand(block: np.ndarray) -> np.ndarray:
    """Normalize one hand's 63 floats. Returns a new array; input is untouched."""
    if block.shape != (FEATURES_PER_HAND,):
        raise ValueError(f"expected ({FEATURES_PER_HAND},), got {block.shape}")

    if not block.any():
        return np.zeros(FEATURES_PER_HAND, dtype=np.float32)

    points = block.reshape(LANDMARKS_PER_HAND, COORDS_PER_LANDMARK).astype(np.float32)

    # 1. translate: wrist becomes the origin
    points = points - points[WRIST]

    # 2. scale: wrist -> middle-finger MCP becomes unit length
    ref = points[MIDDLE_FINGER_MCP]
    dist = float(np.linalg.norm(ref if USE_Z_IN_REFERENCE else ref[:2]))
    if dist < MIN_REFERENCE_DISTANCE:
        return np.zeros(FEATURES_PER_HAND, dtype=np.float32)
    points = points / dist

    return points.reshape(FEATURES_PER_HAND).astype(np.float32)


def normalize_frame(vec: np.ndarray) -> np.ndarray:
    """Normalize one 126-float frame, each hand block independently.

    Position- and scale-invariant per hand. Absent hands stay zeroed.
    """
    vec = np.asarray(vec)
    if vec.shape != (FEATURE_DIM,):
        raise ValueError(f"expected ({FEATURE_DIM},), got {vec.shape}")

    out = np.empty(FEATURE_DIM, dtype=np.float32)
    for block in range(2):
        lo = block * FEATURES_PER_HAND
        hi = lo + FEATURES_PER_HAND
        out[lo:hi] = normalize_hand(vec[lo:hi])
    return out


def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    """Apply `normalize_frame` to every frame of a (T, 126) sequence.

    Convenience only — the math lives in `normalize_frame` and is not duplicated.
    """
    seq = np.asarray(seq)
    if seq.ndim != 2 or seq.shape[1] != FEATURE_DIM:
        raise ValueError(f"expected (T, {FEATURE_DIM}), got {seq.shape}")
    return np.stack([normalize_frame(f) for f in seq]).astype(np.float32)

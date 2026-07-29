/**
 * The 141-float frame contract — the browser half.
 *
 * This is a deliberate, line-by-line mirror of `build_frame_vector` in
 * ml/collect.py. If the two ever disagree, the model is fed a differently
 * shaped world at inference than it saw in training, and every prediction is
 * garbage — with no error anywhere to tell you. Treat the two functions as one
 * unit: change one, change the other, and re-run the parity test.
 *
 * Layout, per frame:
 *
 *     [  0 : 63 ]  Left  hand: 21 landmarks x (x, y, z)
 *     [ 63 :126 ]  Right hand: 21 landmarks x (x, y, z)
 *     [126 :141 ]  Pose: nose, L shoulder, R shoulder, L elbow, R elbow
 *
 *   - The pose block is the BODY ANCHOR. ml/normalize.py places the hands in
 *     signing space relative to the shoulders, which is what preserves *where*
 *     a sign happens and *how* it moves. Sending hands alone (the original
 *     126-float contract) left the model only finger shape, and it collapsed
 *     onto a single class.
 *   - Landmarks stay in MediaPipe's own order (0 = wrist ... 20 = pinky tip)
 *     and in its normalized coordinate space. These are `result.landmarks`
 *     (normalized image coordinates), NOT `result.worldLandmarks`.
 *   - "Left" / "Right" is the `categoryName` MediaPipe reports for that hand,
 *     never a guess from screen position.
 *   - An absent hand's 63 slots are zeros; absent pose is 15 zeros.
 *   - If both detected hands carry the same label, the higher-score one wins
 *     that block and the other is dropped.
 *
 * Values are sent RAW. Normalization happens server-side in ml/normalize.py,
 * the single shared implementation — do not normalize here.
 *
 * Feed the detector the raw, un-mirrored video frame. The preview is mirrored
 * with CSS only. See the contract note in ml/collect.py.
 */

export const HAND_ORDER = ["Left", "Right"] as const;
export const LANDMARKS_PER_HAND = 21;
export const COORDS_PER_LANDMARK = 3;
export const FEATURES_PER_HAND = LANDMARKS_PER_HAND * COORDS_PER_LANDMARK; // 63
export const HANDS_DIM = FEATURES_PER_HAND * HAND_ORDER.length; // 126

/** MediaPipe pose landmark indices kept, in this order. */
export const POSE_LANDMARK_INDICES = [0, 11, 12, 13, 14] as const; // nose, L/R shoulder, L/R elbow
export const POSE_POINTS = POSE_LANDMARK_INDICES.length;
export const POSE_DIM = POSE_POINTS * COORDS_PER_LANDMARK; // 15

export const FEATURE_DIM = HANDS_DIM + POSE_DIM; // 141

/** Structural subset of MediaPipe's types, so this stays unit-testable. */
export interface LandmarkLike {
  x: number;
  y: number;
  z: number;
}
export interface CategoryLike {
  categoryName: string;
  score: number;
}
export interface HandResultLike {
  landmarks: LandmarkLike[][];
  handedness: CategoryLike[][];
}
export interface PoseResultLike {
  landmarks: LandmarkLike[][];
}

/**
 * Flatten one HandLandmarker (+ PoseLandmarker) result into the 141-float
 * frame vector. Float32Array so the values match ml/collect.py's float32
 * storage exactly.
 */
export function buildFrameVector(
  result: HandResultLike,
  pose?: PoseResultLike | null,
): Float32Array {
  const vec = new Float32Array(FEATURE_DIM);

  const poseLandmarks = pose?.landmarks?.[0];
  if (poseLandmarks) {
    POSE_LANDMARK_INDICES.forEach((index, slot) => {
      const lm = poseLandmarks[index];
      if (!lm) return;
      const base = HANDS_DIM + slot * COORDS_PER_LANDMARK;
      vec[base] = lm.x;
      vec[base + 1] = lm.y;
      vec[base + 2] = lm.z;
    });
  }

  if (!result?.landmarks?.length) return vec;

  // Resolve one detection per block first, so a duplicated handedness label
  // can't let a second hand overwrite the first non-deterministically.
  const best = new Map<string, { score: number; landmarks: LandmarkLike[] }>();
  for (let i = 0; i < result.landmarks.length; i += 1) {
    const categories = result.handedness[i];
    if (!categories?.length) continue;
    const { categoryName: label, score } = categories[0];
    if (!(HAND_ORDER as readonly string[]).includes(label)) continue;
    const current = best.get(label);
    // Strictly greater, so the first detection wins ties — same as Python.
    if (!current || score > current.score) {
      best.set(label, { score, landmarks: result.landmarks[i] });
    }
  }

  HAND_ORDER.forEach((label, block) => {
    const entry = best.get(label);
    if (!entry) return; // absent hand keeps its 63 zeros
    const offset = block * FEATURES_PER_HAND;
    const count = Math.min(LANDMARKS_PER_HAND, entry.landmarks.length);
    for (let i = 0; i < count; i += 1) {
      const base = offset + i * COORDS_PER_LANDMARK;
      const lm = entry.landmarks[i];
      vec[base] = lm.x;
      vec[base + 1] = lm.y;
      vec[base + 2] = lm.z;
    }
  });

  return vec;
}

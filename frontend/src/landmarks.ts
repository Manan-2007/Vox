/**
 * The 126-float frame contract — the browser half.
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
 *
 *   - Landmarks stay in MediaPipe's own order (0 = wrist ... 20 = pinky tip)
 *     and in its normalized coordinate space. These are `result.landmarks`
 *     (normalized image coordinates), NOT `result.worldLandmarks`.
 *   - "Left" / "Right" is the `categoryName` MediaPipe reports for that hand,
 *     never a guess from screen position.
 *   - An absent hand's 63 slots are zeros; no hands means 126 zeros.
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
export const FEATURE_DIM = FEATURES_PER_HAND * HAND_ORDER.length; // 126

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

/**
 * Flatten one HandLandmarker result into the 126-float frame vector.
 * Float32Array so the values match ml/collect.py's float32 storage exactly.
 */
export function buildFrameVector(result: HandResultLike): Float32Array {
  const vec = new Float32Array(FEATURE_DIM);
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

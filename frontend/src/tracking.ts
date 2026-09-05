/**
 * Landmark stabilisation — the difference between "it can't see my hands" and
 * a tracker you can trust.
 *
 * MediaPipe returns a plausible answer for every frame it manages to run, but
 * three things go wrong between that and a usable signal:
 *
 *   1. JITTER. Per-frame landmark noise of a few pixels is invisible in a
 *      skeleton overlay and very visible in a solid 3D mesh — and it moves the
 *      feature vector the recogniser sees on every frame, so a held sign never
 *      looks held.
 *
 *   2. HANDEDNESS FLAPPING. The label is re-decided from scratch each frame.
 *      When one hand passes in front of the other — which is most two-handed
 *      ISL signs — "Left" and "Right" can swap for a few frames. The two 63-float
 *      blocks then swap places mid-sign, which to the model is a discontinuity
 *      no training sample ever contained. This is the single worst failure
 *      because it looks like the tracker is working: hands are drawn, confidence
 *      is reported, and the prediction is nonsense.
 *
 *   3. DROPOUT. A frame where a hand is missed becomes 63 zeros. One dropped
 *      frame in a 30-frame window is a hole the sequence model has to absorb.
 *
 * This module fixes 1 and 2, and reports 3 honestly rather than papering over it.
 *
 * The filter is One Euro (Casiez et al., CHI 2012): a low-pass filter whose
 * cutoff rises with speed, so slow movement is smoothed hard and fast movement
 * is barely touched. A fixed-alpha exponential filter cannot do both — tuned to
 * kill jitter it also lags every fast sign, and a sign's speed profile is part
 * of its identity.
 */

import {
  COORDS_PER_LANDMARK,
  FEATURE_DIM,
  FEATURES_PER_HAND,
  HANDS_DIM,
  HAND_ORDER,
  LANDMARKS_PER_HAND,
  type CategoryLike,
  type LandmarkLike,
} from "./landmarks";

/* -------------------------------------------------------------- one euro -- */

/**
 * Tuning. `minCutoff` sets how hard a still landmark is smoothed (lower =
 * smoother); `beta` sets how quickly the filter gets out of the way when the
 * point accelerates (higher = more responsive).
 *
 * Hands and body get different settings on purpose. The shoulders are the
 * anchor every other coordinate is expressed relative to (see ml/normalize.py),
 * so shoulder noise is multiplied into every hand landmark — the body is
 * therefore smoothed much harder than the hands, and it can be, because
 * shoulders genuinely do not move fast while signing.
 */
const HAND_TUNING = { minCutoff: 1.7, beta: 0.35, derivateCutoff: 1.0 };
const BODY_TUNING = { minCutoff: 0.6, beta: 0.06, derivateCutoff: 1.0 };

const TWO_PI = Math.PI * 2;

function alpha(cutoff: number, dt: number): number {
  const tau = 1 / (TWO_PI * cutoff);
  return 1 / (1 + tau / dt);
}

/** One scalar channel of a One Euro filter. */
class OneEuroChannel {
  private value = 0;
  private derivative = 0;
  private started = false;
  private readonly tuning: typeof HAND_TUNING;

  constructor(tuning: typeof HAND_TUNING) {
    this.tuning = tuning;
  }

  reset(): void {
    this.started = false;
    this.derivative = 0;
  }

  filter(raw: number, dt: number): number {
    if (!this.started) {
      this.started = true;
      this.value = raw;
      return raw;
    }
    const rawDerivative = (raw - this.value) / dt;
    const dAlpha = alpha(this.tuning.derivateCutoff, dt);
    this.derivative += dAlpha * (rawDerivative - this.derivative);

    const cutoff = this.tuning.minCutoff + this.tuning.beta * Math.abs(this.derivative);
    const a = alpha(cutoff, dt);
    this.value += a * (raw - this.value);
    return this.value;
  }
}

/**
 * A One Euro filter over the whole 141-float frame.
 *
 * Presence-aware, which is the part that is easy to get wrong: a landmark block
 * that is absent this frame is 63 exact zeros, and filtering *toward* zero would
 * drag a hand smoothly across the scene to the origin before it disappeared —
 * and, worse, feed the model a hand that was never there. Absent stays absent,
 * and a block that reappears restarts its filters rather than easing in from
 * wherever it vanished.
 */
export class FrameSmoother {
  private readonly channels: OneEuroChannel[];
  private lastTime: number | null = null;

  constructor() {
    this.channels = Array.from({ length: FEATURE_DIM }, (_, i) =>
      new OneEuroChannel(i < HANDS_DIM ? HAND_TUNING : BODY_TUNING),
    );
  }

  /** `timeMs` is a monotonic clock; dt is derived from it. */
  smooth(frame: Float32Array, timeMs: number): Float32Array {
    const dt =
      this.lastTime === null
        ? 1 / 15
        : Math.min(0.5, Math.max(1 / 120, (timeMs - this.lastTime) / 1000));
    this.lastTime = timeMs;

    const out = new Float32Array(FEATURE_DIM);
    // Blocks: left hand, right hand, pose. Each is present or absent as a unit.
    const blocks: [number, number][] = [
      [0, FEATURES_PER_HAND],
      [FEATURES_PER_HAND, HANDS_DIM],
      [HANDS_DIM, FEATURE_DIM],
    ];

    for (const [lo, hi] of blocks) {
      let present = false;
      for (let i = lo; i < hi; i += 1) {
        if (frame[i] !== 0) {
          present = true;
          break;
        }
      }
      if (!present) {
        for (let i = lo; i < hi; i += 1) this.channels[i].reset();
        continue; // leave the block as zeros
      }
      for (let i = lo; i < hi; i += 1) {
        out[i] = this.channels[i].filter(frame[i], dt);
      }
    }
    return out;
  }

  reset(): void {
    this.channels.forEach((c) => c.reset());
    this.lastTime = null;
  }
}

/* ------------------------------------------------- handedness stabiliser -- */

const WRIST = 0;
/**
 * How far a wrist may move between frames (in normalized image units) before a
 * proximity match is rejected. At 30 FPS a hand moving fast covers roughly
 * 0.1 of the frame width per frame; 0.25 leaves headroom without matching
 * across the whole screen.
 */
const MAX_WRIST_TRAVEL = 0.25;

interface Detection {
  landmarks: LandmarkLike[];
  /** Metric world landmarks for the same hand, when MediaPipe supplied them. */
  world: LandmarkLike[] | null;
  label: string;
  score: number;
}

/** Both representations of the same two hands, indexed 0 = left, 1 = right. */
export interface AssignedHands {
  image: (LandmarkLike[] | null)[];
  world: (LandmarkLike[] | null)[];
}

/**
 * Assigns detections to the Left/Right blocks with memory.
 *
 * MediaPipe's own label is trusted while it is unambiguous. When both hands are
 * present and the labels disagree with where the hands actually *were* one frame
 * ago, continuity wins: a hand does not teleport across the body between frames,
 * but MediaPipe's classifier will happily relabel it. Continuity is checked
 * against the last accepted assignment, so a single bad frame does not poison
 * the track.
 */
export class HandAssigner {
  private previous: (LandmarkLike[] | null)[] = [null, null];
  private staleFrames = [0, 0];

  /** How many consecutive frames a block has been empty. */
  gapFor(block: number): number {
    return this.staleFrames[block];
  }

  reset(): void {
    this.previous = [null, null];
    this.staleFrames = [0, 0];
  }

  /**
   * Returns both representations per block, index 0 = Left, 1 = Right.
   *
   * The world landmarks travel with their own hand through the same decision, so
   * a swap corrected here moves the avatar's geometry and the recogniser's
   * feature vector together. Assigning them separately would let the two views
   * of the same frame disagree about which hand is which.
   */
  assign(
    landmarks: LandmarkLike[][],
    handedness: CategoryLike[][],
    worldLandmarks: LandmarkLike[][] = [],
  ): AssignedHands {
    const detections: Detection[] = [];
    for (let i = 0; i < landmarks.length; i += 1) {
      const category = handedness[i]?.[0];
      if (!category || !landmarks[i]?.length) continue;
      detections.push({
        landmarks: landmarks[i],
        world: worldLandmarks[i]?.length ? worldLandmarks[i] : null,
        label: category.categoryName,
        score: category.score,
      });
    }

    const assigned: AssignedHands = {
      image: [null, null],
      world: [null, null],
    };
    const put = (block: number, detection: Detection) => {
      assigned.image[block] = detection.landmarks;
      assigned.world[block] = detection.world;
    };

    if (detections.length === 0) {
      this.staleFrames = [this.staleFrames[0] + 1, this.staleFrames[1] + 1];
      // Keep `previous` so a brief dropout can still be matched on return.
      return assigned;
    }

    // The labelled assignment MediaPipe is proposing this frame.
    const byLabel: (Detection | null)[] = [null, null];
    for (const detection of detections) {
      const block = HAND_ORDER.indexOf(detection.label as (typeof HAND_ORDER)[number]);
      if (block < 0) continue;
      const current = byLabel[block];
      if (!current || detection.score > current.score) byLabel[block] = detection;
    }

    if (detections.length >= 2 && byLabel[0] && byLabel[1]) {
      // Two hands, one per block. Compare this labelling against its swap and
      // keep whichever moved less from the previous frame.
      const straight =
        this.travel(0, byLabel[0]!) + this.travel(1, byLabel[1]!);
      const swapped =
        this.travel(0, byLabel[1]!) + this.travel(1, byLabel[0]!);
      if (swapped + 1e-6 < straight) {
        // MediaPipe swapped the labels mid-sign; continuity says otherwise.
        put(0, byLabel[1]!);
        put(1, byLabel[0]!);
      } else {
        put(0, byLabel[0]!);
        put(1, byLabel[1]!);
      }
    } else {
      // One hand (or two that MediaPipe labelled identically). Place each
      // detection in the block it is nearest to, falling back to its label.
      for (const detection of detections) {
        const labelBlock = HAND_ORDER.indexOf(
          detection.label as (typeof HAND_ORDER)[number],
        );
        let block = labelBlock >= 0 ? labelBlock : 0;
        const near = this.nearestBlock(detection);
        if (near !== null && assigned.image[near] === null) block = near;
        if (assigned.image[block] !== null) block = block === 0 ? 1 : 0;
        if (assigned.image[block] === null) put(block, detection);
      }
    }

    for (let block = 0; block < 2; block += 1) {
      if (assigned.image[block]) {
        this.previous[block] = assigned.image[block];
        this.staleFrames[block] = 0;
      } else {
        this.staleFrames[block] += 1;
        // After a long gap the old position is no longer evidence of anything.
        if (this.staleFrames[block] > 10) this.previous[block] = null;
      }
    }

    return assigned;
  }

  /** Distance the wrist would have travelled if `detection` were `block`. */
  private travel(block: number, detection: Detection): number {
    const previous = this.previous[block];
    if (!previous) return 0; // no history: no opinion, so no penalty
    const a = previous[WRIST];
    const b = detection.landmarks[WRIST];
    return Math.hypot(a.x - b.x, a.y - b.y);
  }

  private nearestBlock(detection: Detection): number | null {
    let best: number | null = null;
    let bestDistance = MAX_WRIST_TRAVEL;
    for (let block = 0; block < 2; block += 1) {
      if (!this.previous[block]) continue;
      const distance = this.travel(block, detection);
      if (distance < bestDistance) {
        bestDistance = distance;
        best = block;
      }
    }
    return best;
  }
}

/* ------------------------------------------------------- frame assembly --- */

/**
 * Build the 141-float frame from already-assigned blocks.
 *
 * This is the stabilised counterpart to `buildFrameVector` in landmarks.ts: same
 * layout, same raw values, but the Left/Right decision has already been made by
 * `HandAssigner` instead of being read straight off MediaPipe's label.
 */
/**
 * The aspect ratio the feature contract is defined in — mirrors
 * REFERENCE_ASPECT in ml/collect.py.
 *
 * MediaPipe divides x by the frame width and y by the frame height, so a
 * normalized coordinate is only comparable across the two axes at one aspect
 * ratio. Every recording the model was trained on is 16:9. A camera with a
 * different shape is rescaled onto that, rather than being handed to a model
 * that has never seen a signer of those proportions.
 */
export const REFERENCE_ASPECT = 16 / 9;

export function aspectScale(width: number, height: number): number {
  if (!width || !height) return 1;
  return width / height / REFERENCE_ASPECT;
}

/* ------------------------------------------------------- the avatar's pose -- */

/**
 * The pose points the 3D figure needs, which are not the points the recogniser
 * needs.
 *
 * The recogniser's five (nose and the shoulder/elbow pairs) are an *anchor*: it
 * only has to know where the signing space is. A body has to be drawn, and for
 * that the wrists, hips and ears are not optional — see the format-2 note in
 * ml/build_motion.py. The order here is the same as POSE_INDICES there, because
 * the two produce the same block and the rig cannot tell them apart.
 */
export const AVATAR_POSE_INDICES = [
  0,   // nose
  11, 12,  // shoulders
  13, 14,  // elbows
  15, 16,  // wrists
  23, 24,  // hips
  7, 8,    // ears
  2, 5,    // eyes
] as const;

/**
 * Below this, MediaPipe is extrapolating a joint it cannot see.
 *
 * This one constant is the difference between a figure that stands calmly when
 * you step out of shot and one that thrashes. The pose model always returns 33
 * landmarks — when the signer walks away it keeps returning them, positioned by
 * inference from whatever is left in frame, and a rig that draws them without
 * asking produces exactly the abrupt contortion the old build did. Visibility
 * is the model's own answer to "did I actually see this", and it is reliable.
 */
export const POSE_VISIBILITY = 0.5;

/**
 * Build the avatar's 39-float pose block: 13 points x (x, y, z), square-scaled,
 * with anything the model could not actually see written as zeros.
 *
 * `squareScale` is the raw aspect ratio (width / height), NOT the recogniser's
 * aspect correction. MediaPipe normalizes x by width and y by height, so this
 * converts x — and z, which shares x's scale — into units where all three axes
 * measure the same distance. The rig computes real lengths and angles from
 * these, so anisotropic units there mean a figure that is wrong by 78%.
 */
export function assembleAvatarPose(
  pose: LandmarkLike[] | null | undefined,
  squareScale: number,
): { block: Float32Array; visible: number } {
  const block = new Float32Array(AVATAR_POSE_INDICES.length * 3);
  if (!pose) return { block, visible: 0 };

  let visible = 0;
  AVATAR_POSE_INDICES.forEach((index, slot) => {
    const lm = pose[index] as (LandmarkLike & { visibility?: number }) | undefined;
    if (!lm) return;
    if ((lm.visibility ?? 1) < POSE_VISIBILITY) return;
    const base = slot * 3;
    // Exact zero means "absent" everywhere downstream, so nudge a coordinate
    // that legitimately lands on the frame edge.
    block[base] = lm.x * squareScale || 1e-5;
    block[base + 1] = lm.y || 1e-5;
    block[base + 2] = lm.z * squareScale;
    visible += 1;
  });
  return { block, visible };
}

export function assembleFrame(
  hands: AssignedHands,
  pose: LandmarkLike[] | null | undefined,
  poseIndices: readonly number[],
  xScale = 1,
): Float32Array {
  const vec = new Float32Array(FEATURE_DIM);

  if (pose) {
    poseIndices.forEach((index, slot) => {
      const lm = pose[index];
      if (!lm) return;
      const base = HANDS_DIM + slot * COORDS_PER_LANDMARK;
      vec[base] = lm.x * xScale;
      vec[base + 1] = lm.y;
      vec[base + 2] = lm.z;
    });
  }

  for (let block = 0; block < 2; block += 1) {
    const landmarks = hands.image[block];
    if (!landmarks) continue;
    const offset = block * FEATURES_PER_HAND;
    const count = Math.min(LANDMARKS_PER_HAND, landmarks.length);
    for (let i = 0; i < count; i += 1) {
      const base = offset + i * COORDS_PER_LANDMARK;
      const lm = landmarks[i];
      vec[base] = lm.x * xScale;
      vec[base + 1] = lm.y;
      vec[base + 2] = lm.z;
    }
  }

  return vec;
}

/* --------------------------------------------------------- motion energy -- */

/**
 * How much the hands are moving, as a rolling figure in normalized units per
 * second. Used for two things the fixed sliding window cannot do on its own:
 * telling "holding a sign" apart from "hands resting", and finding the boundary
 * between two signs in continuous signing.
 */
export class MotionEnergy {
  private last: Float32Array | null = null;
  private lastTime = 0;
  private energy = 0;

  /** Exponential smoothing factor for the reported energy. */
  private static readonly SMOOTHING = 0.3;

  update(frame: Float32Array, timeMs: number): number {
    const previous = this.last;
    this.last = Float32Array.from(frame);

    if (!previous) {
      this.lastTime = timeMs;
      return this.energy;
    }
    const dt = Math.max(1 / 120, (timeMs - this.lastTime) / 1000);
    this.lastTime = timeMs;

    let sum = 0;
    let counted = 0;
    for (let i = 0; i < HANDS_DIM; i += COORDS_PER_LANDMARK) {
      // Only compare landmarks present in BOTH frames: a hand appearing or
      // vanishing is not movement, and counting it as movement would make
      // dropout look like signing.
      if (frame[i] === 0 || previous[i] === 0) continue;
      sum += Math.hypot(frame[i] - previous[i], frame[i + 1] - previous[i + 1]);
      counted += 1;
    }
    const instant = counted ? (sum / counted) / dt : 0;
    this.energy += MotionEnergy.SMOOTHING * (instant - this.energy);
    return this.energy;
  }

  get value(): number {
    return this.energy;
  }

  reset(): void {
    this.last = null;
    this.energy = 0;
  }
}

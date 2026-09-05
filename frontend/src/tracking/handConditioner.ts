/**
 * Making MediaPipe's hand usable as a source of joint angles.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS HAD TO EXIST BEFORE THE RIG COULD WORK
 * ---------------------------------------------------------------------------
 * The previous build drew cylinders directly between landmark positions. Landmark
 * noise showed up there as jitter — ugly, but bounded, because a wrong point is
 * just a point in a slightly wrong place.
 *
 * A rotation-driven rig is far less forgiving, and the reason is leverage. The
 * lateral deviation a landmark error induces in a bone's DIRECTION scales with
 * the inverse of that bone's length, and a distal phalanx is about 20 mm long.
 * Three millimetres of sideways error — routine for MediaPipe under motion blur
 * — is nine degrees on that bone. Worse, the error compounds: a finger is a
 * chain, and each joint's angle is measured relative to its parent's already-
 * wrong direction.
 *
 * Measured over all 239 signs of the library, the rotation the raw landmarks
 * demand at a PIP joint has a 5th–95th percentile spread of roughly ±40° about
 * the joint's lateral axis. A PIP is a hinge. It has perhaps 6° of lateral play.
 * So about 35° of that is pure noise, and a rig that renders it produces fingers
 * that splay, cross and shimmer — the exact "fingers intersect" and "joints
 * collapse" defects reported.
 *
 * ---------------------------------------------------------------------------
 * THREE STAGES, IN THIS ORDER
 * ---------------------------------------------------------------------------
 *   1. OUTLIER GATE. Reject a frame whose landmarks could not have followed the
 *      previous frame's. This runs FIRST and it must: the One Euro filter that
 *      follows is designed to get out of the way when it sees a large
 *      derivative, so feeding it an outlier makes it open its cutoff and pass
 *      the outlier straight through. One Euro without a gate in front of it is
 *      an outlier amplifier, which is why the previous build had "random jumps"
 *      despite having a One Euro filter.
 *
 *   2. SKELETON PROJECTION. A hand's bones do not change length. MediaPipe's do,
 *      frame to frame, by several percent. Re-projecting each frame onto the
 *      hand's own measured bone lengths removes that entire error mode and costs
 *      nothing — it is one pass down each finger. This is the single most
 *      effective denoiser here, because it uses a hard physical fact rather than
 *      a smoothness assumption, so unlike a filter it introduces no lag.
 *
 *   3. ONE EURO. Adaptive low-pass on what remains: heavy smoothing while the
 *      hand is still, backing off as it accelerates. A fixed filter cannot do
 *      both, and a sign's speed profile is part of its identity — smoothing a
 *      fast sign into a slow one changes what it means.
 *
 * Stage 2 is what makes stages 1 and 3 cheap: with lengths pinned, the only
 * remaining freedom is direction, so the filter has less to do and can be gentler
 * — which means less lag on exactly the fast movements that carry meaning.
 */

import { HAND, fingerChains } from "../core/humanoid";

/** MediaPipe hand topology as parent → child, in solve order from the wrist. */
const CHAIN: readonly (readonly [number, number])[] = (() => {
  const links: [number, number][] = [];
  for (const chain of fingerChains("left")) {
    // The wrist connects to each finger's first landmark; then down the finger.
    links.push([HAND.WRIST, chain.landmarks[0]]);
    links.push([chain.landmarks[0], chain.landmarks[1]]);
    links.push([chain.landmarks[1], chain.landmarks[2]]);
    links.push([chain.landmarks[2], chain.landmarks[3]]);
  }
  return links;
})();

const LANDMARKS = HAND.COUNT;
const FLOATS = LANDMARKS * 3;

/* ------------------------------------------------------------- one euro --- */

/**
 * Tuning. `minCutoff` sets how hard a still landmark is smoothed (lower =
 * smoother); `beta` how quickly the filter gets out of the way under
 * acceleration.
 *
 * Gentler than the values used for the recogniser's image-space frame, because
 * skeleton projection has already removed the error mode that needed heavy
 * smoothing. Buying stability with lag here would cost sign legibility.
 */
const MIN_CUTOFF = 2.4;
const BETA = 0.55;
const DERIVATIVE_CUTOFF = 1.2;

const TWO_PI = Math.PI * 2;

function alpha(cutoff: number, dt: number): number {
  const tau = 1 / (TWO_PI * cutoff);
  return 1 / (1 + tau / dt);
}

/* --------------------------------------------------------- the gate ------- */

/**
 * How far a landmark may move between frames, in metres per second, before the
 * frame is treated as a detection failure rather than a fast hand.
 *
 * A fingertip in a fast fingerspelled sequence peaks around 2.5 m/s. Eight is
 * comfortably above anything a hand does and comfortably below the teleport a
 * mis-detection produces — MediaPipe locking briefly onto a face or a second
 * person moves the whole set by 20 cm or more in one frame.
 */
const MAX_SPEED = 8.0;
/**
 * How much of the hand has to be implausible before the whole frame is refused.
 *
 * Per-landmark rejection was tried and is worse: it lets half a hand jump while
 * the other half holds, which produces a shape no hand can make. A hand is one
 * rigid-ish object and it succeeds or fails as one.
 */
const OUTLIER_FRACTION = 0.34;

/* ------------------------------------------------------- the conditioner -- */

export interface ConditionResult {
  /** False when this frame was rejected or there is nothing to report yet. */
  valid: boolean;
  /** 0-1. Falls while frames are being rejected, so the rig can back off. */
  confidence: number;
  /** True when the frame was refused by the outlier gate. */
  rejected: boolean;
}

/**
 * One hand's worth of conditioning state. Two instances per pipeline — the left
 * and right hands are independent and must never share history, or a hand that
 * disappears drags the other one toward it.
 */
export class HandConditioner {
  /** Filtered output, 21 landmarks × (x, y, z), scene space, metres. */
  readonly points = new Float32Array(FLOATS);

  private readonly filtered = new Float32Array(FLOATS);
  private readonly derivative = new Float32Array(FLOATS);
  private readonly previousRaw = new Float32Array(FLOATS);
  private readonly projected = new Float32Array(FLOATS);

  /** Running estimate of this hand's own bone lengths, one per CHAIN link. */
  private readonly boneLength = new Float32Array(CHAIN.length);
  private lengthSamples = 0;

  private started = false;
  private lastTime = 0;
  private confidence = 0;
  private consecutiveRejects = 0;

  reset(): void {
    this.started = false;
    this.lengthSamples = 0;
    this.confidence = 0;
    this.consecutiveRejects = 0;
    this.boneLength.fill(0);
    this.derivative.fill(0);
  }

  /**
   * Condition one frame.
   *
   * `raw` holds 21 scene-space landmarks in metres at `offset`. Returns whether
   * the result in `points` should be believed.
   */
  update(raw: Float32Array, offset: number, timeMs: number): ConditionResult {
    const dt = this.started
      ? Math.min(0.4, Math.max(1 / 240, (timeMs - this.lastTime) / 1000))
      : 1 / 30;
    this.lastTime = timeMs;

    /* ------------------------------------------------- 1. outlier gate -- */
    if (this.started) {
      let implausible = 0;
      for (let i = 0; i < LANDMARKS; i += 1) {
        const s = offset + i * 3;
        const dx = raw[s] - this.previousRaw[i * 3];
        const dy = raw[s + 1] - this.previousRaw[i * 3 + 1];
        const dz = raw[s + 2] - this.previousRaw[i * 3 + 2];
        if (Math.hypot(dx, dy, dz) / dt > MAX_SPEED) implausible += 1;
      }
      if (implausible / LANDMARKS > OUTLIER_FRACTION) {
        this.consecutiveRejects += 1;
        // Refuse the frame, but not forever. A genuine fast re-entry — the hand
        // leaving frame and coming back somewhere else — looks exactly like an
        // outlier, and after a few frames of agreement it IS the hand.
        if (this.consecutiveRejects < 4) {
          this.confidence *= 0.6;
          return { valid: this.confidence > 0.05, confidence: this.confidence, rejected: true };
        }
        // Sustained disagreement: the hand really did move. Re-seed.
        this.started = false;
        this.lengthSamples = 0;
      }
    }
    this.consecutiveRejects = 0;
    for (let i = 0; i < LANDMARKS; i += 1) {
      this.previousRaw[i * 3] = raw[offset + i * 3];
      this.previousRaw[i * 3 + 1] = raw[offset + i * 3 + 1];
      this.previousRaw[i * 3 + 2] = raw[offset + i * 3 + 2];
    }

    /* ------------------------------------------ 2. skeleton projection -- */
    this.learnLengths(raw, offset);
    this.project(raw, offset);

    /* ----------------------------------------------------- 3. one euro -- */
    if (!this.started) {
      this.started = true;
      this.filtered.set(this.projected);
      this.derivative.fill(0);
      this.points.set(this.projected);
      this.confidence = 0.5;
      return { valid: true, confidence: this.confidence, rejected: false };
    }

    const dAlpha = alpha(DERIVATIVE_CUTOFF, dt);
    for (let i = 0; i < FLOATS; i += 1) {
      const rate = (this.projected[i] - this.filtered[i]) / dt;
      this.derivative[i] += dAlpha * (rate - this.derivative[i]);
      const cutoff = MIN_CUTOFF + BETA * Math.abs(this.derivative[i]);
      this.filtered[i] += alpha(cutoff, dt) * (this.projected[i] - this.filtered[i]);
    }
    this.points.set(this.filtered);

    this.confidence = Math.min(1, this.confidence + dt * 4);
    return { valid: true, confidence: this.confidence, rejected: false };
  }

  /**
   * Learn this hand's bone lengths.
   *
   * A running MINIMUM-biased estimate rather than a mean, and the reason is
   * asymmetry in how the error behaves: a landmark seen clearly sits close to
   * the truth, while a landmark that is occluded or motion-blurred is
   * extrapolated and lands almost anywhere — usually further away. So the
   * distribution has a long upper tail and a hard lower edge, and the mean
   * tracks the tail. Easing toward the shorter of (estimate, measurement) fast
   * and toward the longer slowly puts the estimate near the mode.
   */
  private learnLengths(raw: Float32Array, offset: number): void {
    for (let link = 0; link < CHAIN.length; link += 1) {
      const [from, to] = CHAIN[link];
      const a = offset + from * 3;
      const b = offset + to * 3;
      const length = Math.hypot(raw[b] - raw[a], raw[b + 1] - raw[a + 1], raw[b + 2] - raw[a + 2]);
      if (!(length > 1e-4) || length > 0.12) continue;

      if (this.lengthSamples === 0) {
        this.boneLength[link] = length;
        continue;
      }
      const current = this.boneLength[link];
      const rate = length < current ? 0.25 : 0.03;
      this.boneLength[link] = current + rate * (length - current);
    }
    this.lengthSamples += 1;
  }

  /**
   * Re-place every landmark at its learned distance from its parent, keeping the
   * measured direction.
   *
   * The wrist is the root and does not move. Each link is then walked outward,
   * so a corrected parent is what the child is measured from — which means the
   * whole finger stays connected instead of each joint being fixed in isolation.
   */
  private project(raw: Float32Array, offset: number): void {
    for (let i = 0; i < LANDMARKS; i += 1) {
      this.projected[i * 3] = raw[offset + i * 3];
      this.projected[i * 3 + 1] = raw[offset + i * 3 + 1];
      this.projected[i * 3 + 2] = raw[offset + i * 3 + 2];
    }
    if (this.lengthSamples < 2) return;

    for (let link = 0; link < CHAIN.length; link += 1) {
      const [from, to] = CHAIN[link];
      const target = this.boneLength[link];
      if (!(target > 1e-4)) continue;
      const a = from * 3;
      const b = to * 3;
      const dx = this.projected[b] - this.projected[a];
      const dy = this.projected[b + 1] - this.projected[a + 1];
      const dz = this.projected[b + 2] - this.projected[a + 2];
      const length = Math.hypot(dx, dy, dz);
      if (length < 1e-6) {
        // Collapsed joint: continue straight on from the parent link so the
        // finger stays a finger rather than folding into a point.
        this.projected[b] = this.projected[a];
        this.projected[b + 1] = this.projected[a + 1] + target;
        this.projected[b + 2] = this.projected[a + 2];
        continue;
      }
      const scale = target / length;
      this.projected[b] = this.projected[a] + dx * scale;
      this.projected[b + 1] = this.projected[a + 1] + dy * scale;
      this.projected[b + 2] = this.projected[a + 2] + dz * scale;
    }
  }
}

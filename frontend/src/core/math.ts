/**
 * Rotation maths for a rig that is driven by measurements.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS FILE EXISTS AT ALL
 * ---------------------------------------------------------------------------
 * The previous build animated joint POSITIONS. Everything it could not do
 * followed from that one choice: a position has no roll, so forearm supination
 * was unrepresentable and palm orientation — one of the five phonological
 * parameters of a sign — was structurally missing. A position cannot be limited
 * either: an anatomical limit is a statement about an angle, so with only points
 * there is nothing to clamp and elbows bend sideways.
 *
 * So this rig animates ROTATIONS, and rotations need their own toolkit:
 *
 *   * You cannot lerp them. Component-wise blending of two rotations cuts the
 *     corner — the limb visibly shortens through the blend and snaps back at the
 *     end. `slerp` is the only correct answer, and every blend here uses it.
 *
 *   * You cannot spring them the way you spring a point. A critically damped
 *     spring needs a vector space; quaternions are not one. The trick used
 *     throughout is to work on the ROTATION VECTOR of the error — axis × angle,
 *     which *is* a vector space near the identity — and map back.
 *
 *   * You cannot clamp them per-axis without first deciding what "per-axis"
 *     means. Euler angles have gimbal lock and an order-dependent meaning, which
 *     makes them useless for a limit that must hold at every pose. Swing-twist
 *     decomposition is the standard answer and it is exact: see `swingTwist`.
 *
 * Nothing in this file allocates. Every function either writes into an output
 * argument or uses one of the module-scope scratch values, because all of it
 * runs 60 times a second across ~60 bones.
 */

import * as THREE from "three";

/* ------------------------------------------------------------- scalars --- */

export function clamp(value: number, low: number, high: number): number {
  return value < low ? low : value > high ? high : value;
}

export function smoothstep(x: number): number {
  const t = clamp(x, 0, 1);
  return t * t * (3 - 2 * t);
}

/** Smoothstep's continuous-acceleration sibling; used where velocity matters. */
export function smootherstep(x: number): number {
  const t = clamp(x, 0, 1);
  return t * t * t * (t * (t * 6 - 15) + 10);
}

/**
 * A soft limiter: the identity near zero, asymptotic to ±limit.
 *
 * Preferred over a hard clamp anywhere the input is a noisy estimate. A clamp
 * turns every overshoot into a value pinned to the limit, and a pinned value
 * reads on screen as a joint that has SNAPPED and is now stuck — which is much
 * more alarming than a joint that leaned a bit too far.
 */
export function soften(value: number, limit: number): number {
  return limit * Math.tanh(value / limit);
}

/**
 * Frame-rate independent exponential blend factor.
 *
 * `rate` is "fraction of the remaining distance covered per second". Naively
 * writing `value += 0.1 * (target - value)` per frame makes the smoothing speed
 * depend on the frame rate, so the avatar is smoother on a fast machine — which
 * is exactly backwards.
 */
export function expBlend(rate: number, dt: number): number {
  return 1 - Math.exp(-rate * dt);
}

/* ------------------------------------------------------ damped scalars --- */

/**
 * A critically damped spring, in the numerically stable form that survives
 * large timesteps (Game Programming Gems 4, ch. 1.10).
 *
 * `smoothTime` is roughly how long it takes to cover most of the remaining
 * distance, so it reads directly as "how much lag am I buying".
 *
 * Critically damped rather than exponential because the two differ exactly
 * where it matters: an exponential filter slow enough to kill tracker jitter
 * also visibly lags a fast sign, while a spring carries velocity through the
 * move and settles without overshoot.
 */
export class DampedScalar {
  value = 0;
  private velocity = 0;
  private live = false;

  get settled(): boolean {
    return this.live;
  }

  snap(value: number): void {
    this.value = value;
    this.velocity = 0;
    this.live = true;
  }

  step(target: number, smoothTime: number, dt: number): number {
    if (!this.live) {
      this.snap(target);
      return this.value;
    }
    const omega = 2 / Math.max(1e-3, smoothTime);
    const x = omega * dt;
    const decay = 1 / (1 + x + 0.48 * x * x + 0.235 * x * x * x);
    const change = this.value - target;
    const temp = (this.velocity + omega * change) * dt;
    this.velocity = (this.velocity - omega * temp) * decay;
    this.value = target + (change + temp) * decay;
    return this.value;
  }

  reset(): void {
    this.live = false;
    this.velocity = 0;
  }
}

/** The same spring, per axis of a vector. */
export class DampedVector3 {
  readonly value = new THREE.Vector3();
  private readonly velocity = new THREE.Vector3();
  private live = false;

  get settled(): boolean {
    return this.live;
  }

  snap(target: THREE.Vector3): THREE.Vector3 {
    this.value.copy(target);
    this.velocity.set(0, 0, 0);
    this.live = true;
    return this.value;
  }

  step(target: THREE.Vector3, smoothTime: number, dt: number): THREE.Vector3 {
    if (!this.live) return this.snap(target);
    const omega = 2 / Math.max(1e-3, smoothTime);
    const x = omega * dt;
    const decay = 1 / (1 + x + 0.48 * x * x + 0.235 * x * x * x);
    for (const axis of AXES) {
      const change = this.value[axis] - target[axis];
      const temp = (this.velocity[axis] + omega * change) * dt;
      this.velocity[axis] = (this.velocity[axis] - omega * temp) * decay;
      this.value[axis] = target[axis] + (change + temp) * decay;
    }
    return this.value;
  }

  /**
   * Advance under the current velocity with no target — the COASTING step.
   *
   * `decay` is the fraction of velocity retained per second, so 0.02 means the
   * limb keeps about 2% of its speed after one second. This is what makes a
   * hand that vanishes mid-movement keep moving the way it was moving and slow
   * to a stop, instead of freezing on the last frame and then teleporting to
   * wherever it reappears.
   */
  coast(dt: number, decay = 0.02): THREE.Vector3 {
    if (!this.live) return this.value;
    this.value.addScaledVector(this.velocity, dt);
    this.velocity.multiplyScalar(Math.pow(decay, dt));
    return this.value;
  }

  reset(): void {
    this.live = false;
    this.velocity.set(0, 0, 0);
  }
}

const AXES = ["x", "y", "z"] as const;

/* --------------------------------------------------------- quaternions --- */

/**
 * The rotation vector of a quaternion: the axis scaled by the angle, in
 * radians. This is the logarithm map, and it is what turns "rotations" into
 * something a spring can integrate.
 *
 * Near the identity it is well-conditioned and behaves like an ordinary 3-vector,
 * which is exactly the regime a smoothing filter operates in.
 */
export function quatToRotationVector(
  q: THREE.Quaternion,
  out: THREE.Vector3,
): THREE.Vector3 {
  // Take the shorter arc. Without this a quaternion and its negation — which
  // are the SAME rotation — produce error vectors pointing opposite ways, and
  // the spring spins the joint the long way round for one frame. That single
  // frame is a full-speed 360° flip on screen.
  let { x, y, z, w } = q;
  if (w < 0) {
    x = -x;
    y = -y;
    z = -z;
    w = -w;
  }
  const sinHalf = Math.sqrt(x * x + y * y + z * z);
  if (sinHalf < 1e-9) return out.set(0, 0, 0);
  const angle = 2 * Math.atan2(sinHalf, w);
  const scale = angle / sinHalf;
  return out.set(x * scale, y * scale, z * scale);
}

/** The exponential map: a rotation vector back to a quaternion. */
export function rotationVectorToQuat(
  v: THREE.Vector3,
  out: THREE.Quaternion,
): THREE.Quaternion {
  const angle = v.length();
  if (angle < 1e-9) return out.set(0, 0, 0, 1);
  const half = angle * 0.5;
  const scale = Math.sin(half) / angle;
  return out.set(v.x * scale, v.y * scale, v.z * scale, Math.cos(half));
}

/**
 * Angle between two rotations, in radians. Used for outlier gates ("no joint
 * moves 90° in one frame") and for the debug panel.
 */
export function quatAngle(a: THREE.Quaternion, b: THREE.Quaternion): number {
  const dot = Math.abs(clamp(a.dot(b), -1, 1));
  return 2 * Math.acos(dot);
}

/**
 * A critically damped spring on a rotation.
 *
 * Integrates the spring on the rotation vector of the error, then composes the
 * result back. This is the piece that guarantees the brief's "never snap":
 * every bone in the rig is driven through one of these, so a target that jumps
 * produces a fast but continuous move, never a discontinuity.
 */
export class DampedQuaternion {
  readonly value = new THREE.Quaternion();
  /** Angular velocity, radians/second, in the same frame as `value`. */
  private readonly velocity = new THREE.Vector3();
  private live = false;

  get settled(): boolean {
    return this.live;
  }

  /** Current angular speed, rad/s — surfaced to the debug panel. */
  get speed(): number {
    return this.velocity.length();
  }

  snap(target: THREE.Quaternion): THREE.Quaternion {
    this.value.copy(target);
    this.velocity.set(0, 0, 0);
    this.live = true;
    return this.value;
  }

  step(
    target: THREE.Quaternion,
    smoothTime: number,
    dt: number,
  ): THREE.Quaternion {
    if (!this.live) return this.snap(target);

    // The whole trick is to run the ordinary scalar spring in the tangent space
    // AT THE TARGET, so every quantity below is an ordinary 3-vector:
    //
    //     change = log(current ⋅ target⁻¹)     "how far current is from target"
    //     current = exp(change) ⋅ target       exactly recovers it
    //
    // The scalar recurrence then transfers line for line, and composing the
    // result back onto `target` is what keeps the motion on the rotation
    // manifold instead of drifting off it the way a component-wise blend does.
    Q.error.copy(target).invert().premultiply(this.value);
    quatToRotationVector(Q.error, V.error);

    const omega = 2 / Math.max(1e-3, smoothTime);
    const x = omega * dt;
    const decay = 1 / (1 + x + 0.48 * x * x + 0.235 * x * x * x);

    V.temp.copy(this.velocity).addScaledVector(V.error, omega).multiplyScalar(dt);
    this.velocity.addScaledVector(V.temp, -omega).multiplyScalar(decay);
    V.step.copy(V.error).add(V.temp).multiplyScalar(decay);

    rotationVectorToQuat(V.step, Q.delta);
    this.value.copy(target).premultiply(Q.delta).normalize();
    return this.value;
  }

  /** Continue turning under the current angular velocity. See DampedVector3.coast. */
  coast(dt: number, decay = 0.02): THREE.Quaternion {
    if (!this.live) return this.value;
    V.step.copy(this.velocity).multiplyScalar(dt);
    rotationVectorToQuat(V.step, Q.delta);
    this.value.premultiply(Q.delta).normalize();
    this.velocity.multiplyScalar(Math.pow(decay, dt));
    return this.value;
  }

  reset(): void {
    this.live = false;
    this.velocity.set(0, 0, 0);
  }
}

/* ------------------------------------------------------- swing / twist --- */

export interface SwingTwist {
  swing: THREE.Quaternion;
  twist: THREE.Quaternion;
}

/**
 * Split a rotation into the part that moves `axis` (swing) and the part that
 * spins about it (twist), such that `q = swing ⋅ twist`.
 *
 * This is the decomposition every anatomical joint limit is expressed in, and
 * the reason is anatomy rather than mathematics. A shoulder has a swing cone
 * (how far the arm can point away from rest) and a separate, much tighter twist
 * range (how far the humerus can rotate about its own length) — those two limits
 * are independent and interact through no Euler ordering. An elbow has almost no
 * swing off its hinge axis and a wide twist. Euler angles cannot express either
 * without choosing an order, and every order is wrong somewhere in the range.
 *
 * `axis` must be normalized.
 */
export function swingTwist(
  q: THREE.Quaternion,
  axis: THREE.Vector3,
  out: SwingTwist,
): SwingTwist {
  // Project the quaternion's vector part onto the axis: that is the twist.
  const dot = q.x * axis.x + q.y * axis.y + q.z * axis.z;
  out.twist.set(axis.x * dot, axis.y * dot, axis.z * dot, q.w);

  if (out.twist.lengthSq() < 1e-12) {
    // Exactly 180° of swing: the twist is undefined, so call it identity.
    out.twist.set(0, 0, 0, 1);
  } else {
    out.twist.normalize();
    // Keep the twist on the near side, so its angle reads as signed and small.
    if (dot < 0 !== out.twist.w < 0 && out.twist.w < 0) {
      out.twist.set(-out.twist.x, -out.twist.y, -out.twist.z, -out.twist.w);
    }
  }

  out.swing.copy(out.twist).invert().premultiply(q).normalize();
  return out;
}

export function makeSwingTwist(): SwingTwist {
  return { swing: new THREE.Quaternion(), twist: new THREE.Quaternion() };
}

/**
 * The signed angle of a twist about `axis`, in radians.
 *
 * Assumes `twist` is already a pure rotation about `axis` (i.e. came out of
 * `swingTwist`).
 */
export function twistAngle(
  twist: THREE.Quaternion,
  axis: THREE.Vector3,
): number {
  const dot = twist.x * axis.x + twist.y * axis.y + twist.z * axis.z;
  return 2 * Math.atan2(dot, twist.w);
}

/**
 * Clamp a rotation into an elliptical swing cone plus a twist range.
 *
 * The cone is elliptical rather than circular because joints are: a shoulder
 * swings much further forward than backward, and a finger MCP flexes 90° while
 * spreading only 20°. A circular cone sized for the larger axis lets the smaller
 * one go somewhere anatomy does not, which is precisely what makes fingers
 * intersect.
 *
 *   `axis`   — the bone's own long axis, the twist axis.
 *   `swingX` — half-angle limits about the local x axis, [negative, positive].
 *   `swingZ` — half-angle limits about the local z axis.
 *   `twist`  — [negative, positive] limits about `axis`.
 *
 * All limits in radians. Writes into `q`.
 */
export function clampSwingTwist(
  q: THREE.Quaternion,
  axis: THREE.Vector3,
  swingX: readonly [number, number],
  swingZ: readonly [number, number],
  twistRange: readonly [number, number],
): THREE.Quaternion {
  swingTwist(q, axis, ST.a);

  // The swing as a rotation vector: its components about the two axes
  // perpendicular to `axis` are the two swing angles.
  quatToRotationVector(ST.a.swing, V.swing);
  const x = clamp(V.swing.x, swingX[0], swingX[1]);
  const z = clamp(V.swing.z, swingZ[0], swingZ[1]);
  // The component along the twist axis belongs to the twist, not the swing;
  // after decomposition it is ~0, and forcing it to exactly 0 keeps the two
  // parts from fighting when the input is near a singularity.
  V.swing.set(x, 0, z);
  rotationVectorToQuat(V.swing, ST.a.swing);

  const angle = twistAngle(ST.a.twist, axis);
  const limited = clamp(angle, twistRange[0], twistRange[1]);
  if (limited !== angle) ST.a.twist.setFromAxisAngle(axis, limited);

  return q.copy(ST.a.swing).multiply(ST.a.twist).normalize();
}

/**
 * Clamp a local bone rotation using limits written in the bone's CANONICAL
 * frame (+Y along the bone, +X flexion, +Z spread).
 *
 * `frame` is the rotation that takes the canonical axes onto the model's actual
 * local axes for this bone — measured from the .vrm at load, never assumed. The
 * rotation is conjugated into canonical space, clamped there, and conjugated
 * back, so one set of clinical ranges is valid for every model.
 *
 * Conjugation rather than a change of the clamp's own axes because the swing
 * limits are *elliptical* (a finger flexes 90° and spreads 20°), and an ellipse
 * is only axis-aligned in the frame it was written for.
 */
export function clampInFrame(
  q: THREE.Quaternion,
  frame: THREE.Quaternion,
  flex: readonly [number, number],
  spread: readonly [number, number],
  twist: readonly [number, number],
): THREE.Quaternion {
  F.inverse.copy(frame).invert();
  // q_canonical = frame⁻¹ ⋅ q ⋅ frame
  F.canonical.copy(F.inverse).multiply(q).multiply(frame);
  clampSwingTwist(F.canonical, AXIS_Y, flex, spread, twist);
  return q.copy(frame).multiply(F.canonical).multiply(F.inverse).normalize();
}

const F = {
  inverse: new THREE.Quaternion(),
  canonical: new THREE.Quaternion(),
};

/**
 * The rotation that takes `from` to `to`, both unit vectors, with a stable
 * choice of perpendicular when they are opposed.
 *
 * three.js's own `setFromUnitVectors` handles the antiparallel case by picking
 * an axis off the largest component, which is fine in isolation and flickers
 * when the input passes through the singularity frame by frame — and a hand
 * pointing straight down the bone axis is not a rare pose in signing. `hint`
 * pins the choice.
 */
export function rotationBetween(
  from: THREE.Vector3,
  to: THREE.Vector3,
  hint: THREE.Vector3,
  out: THREE.Quaternion,
): THREE.Quaternion {
  const dot = clamp(from.dot(to), -1, 1);
  if (dot > 0.999999) return out.set(0, 0, 0, 1);
  if (dot < -0.999999) {
    V.axis.copy(hint).addScaledVector(from, -hint.dot(from));
    if (V.axis.lengthSq() < 1e-12) {
      // hint is parallel to `from`; any perpendicular will do, deterministically.
      V.axis.set(from.y, -from.x, 0);
      if (V.axis.lengthSq() < 1e-12) V.axis.set(0, from.z, -from.y);
    }
    V.axis.normalize();
    return out.setFromAxisAngle(V.axis, Math.PI);
  }
  V.axis.crossVectors(from, to);
  out.set(V.axis.x, V.axis.y, V.axis.z, 1 + dot);
  return out.normalize();
}

/**
 * Build a rotation from a forward direction and an up hint — the standard
 * "look along" basis, orthonormalized so a slightly-wrong `up` cannot shear it.
 *
 * This is how wrist orientation is recovered: `forward` is wrist → middle
 * knuckle, `up` is the palm normal. Getting a full 3-DOF orientation out of the
 * hand rather than just a direction is what makes palm facing readable, and
 * palm facing is one of the five parameters that distinguish one sign from
 * another.
 */
export function basisRotation(
  forward: THREE.Vector3,
  up: THREE.Vector3,
  out: THREE.Quaternion,
): THREE.Quaternion {
  B.z.copy(forward).normalize();
  B.x.crossVectors(up, B.z);
  if (B.x.lengthSq() < 1e-10) {
    // up is parallel to forward — pick any perpendicular, deterministically.
    B.x.set(B.z.y, -B.z.x, 0);
    if (B.x.lengthSq() < 1e-10) B.x.set(0, B.z.z, -B.z.y);
  }
  B.x.normalize();
  B.y.crossVectors(B.z, B.x).normalize();
  B.m.makeBasis(B.x, B.y, B.z);
  return out.setFromRotationMatrix(B.m);
}

/* ---------------------------------------------------------------- scratch -- */

const Q = {
  error: new THREE.Quaternion(),
  delta: new THREE.Quaternion(),
};

const V = {
  error: new THREE.Vector3(),
  temp: new THREE.Vector3(),
  step: new THREE.Vector3(),
  swing: new THREE.Vector3(),
  axis: new THREE.Vector3(),
};

const ST = { a: makeSwingTwist() };

const B = {
  x: new THREE.Vector3(),
  y: new THREE.Vector3(),
  z: new THREE.Vector3(),
  m: new THREE.Matrix4(),
};

/* ------------------------------------------------------------- constants -- */

export const DEG = Math.PI / 180;
export const AXIS_X = Object.freeze(new THREE.Vector3(1, 0, 0));
export const AXIS_Y = Object.freeze(new THREE.Vector3(0, 1, 0));
export const AXIS_Z = Object.freeze(new THREE.Vector3(0, 0, 1));

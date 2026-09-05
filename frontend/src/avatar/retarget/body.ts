/**
 * Torso, neck and head — and the reason a signing avatar needs all three.
 *
 * ---------------------------------------------------------------------------
 * SIGN LANGUAGE IS NOT ON THE HANDS ALONE
 * ---------------------------------------------------------------------------
 * In Indian Sign Language a raised brow is not an expression, it is a morpheme.
 * The same sequence of handshapes means different things depending on what the
 * face and body do over it:
 *
 *     YOU DOCTOR            neutral         →  "You are a doctor."
 *     YOU DOCTOR            brows raised    →  "Are you a doctor?"
 *     HOSPITAL YOU GO   + head shake over GO →  "You are NOT going."
 *
 * A reader given the hands alone gets the words and loses the sentence type. So
 * the head is not decoration on this rig; it carries clause type, and a head that
 * does not move is a sentence with its grammar deleted.
 *
 * ---------------------------------------------------------------------------
 * SIGNING SPACE, AND WHY THE FIGURE DOES NOT WANDER
 * ---------------------------------------------------------------------------
 * Everything here is expressed relative to the SHOULDER MIDPOINT, scaled by the
 * shoulder span, and then placed at the model's own rest shoulder position. That
 * has three consequences, all wanted:
 *
 *   * The avatar stands still while the signer moves around the frame. Where a
 *     sign is made relative to the BODY is meaning; where the body is relative to
 *     the camera is not.
 *   * A tall signer and a short one produce the same avatar pose, because the
 *     shoulder span scales with the person.
 *   * It is the same anchor ml/normalize.py uses for the recogniser, so the
 *     avatar and the recogniser agree about where a sign happened.
 */

import * as THREE from "three";
import type { BoneName, HumanoidPose } from "../../core/humanoid";
import { JOINT_LIMITS } from "../../core/humanoid";
import {
  basisRotation,
  clampInFrame,
  soften,
} from "../../core/math";
import type { NonManual } from "../nonManual";
import type { BodyFrame } from "./arm";
import type { RestPose } from "./rest";

/** Pose slots, matching the motion library's 13-point block. */
export const POSE = {
  NOSE: 0,
  L_SHOULDER: 1,
  R_SHOULDER: 2,
  L_ELBOW: 3,
  R_ELBOW: 4,
  L_WRIST: 5,
  R_WRIST: 6,
  L_HIP: 7,
  R_HIP: 8,
  L_EAR: 9,
  R_EAR: 10,
  L_EYE: 11,
  R_EYE: 12,
  COUNT: 13,
} as const;

/**
 * One frame of body measurement, already converted into model space.
 *
 * `null` positions mean the tracker did not see that point — never "it is at the
 * origin". Conflating the two is how the previous build ended up drawing a body
 * assembled from landmarks that were never observed.
 */
export interface BodyMeasurement {
  valid: boolean;
  shoulderL: THREE.Vector3;
  shoulderR: THREE.Vector3;
  shoulderMid: THREE.Vector3;
  pelvis: THREE.Vector3;
  side: THREE.Vector3;
  up: THREE.Vector3;
  forward: THREE.Vector3;
  wrist: [THREE.Vector3 | null, THREE.Vector3 | null];
  elbow: [THREE.Vector3 | null, THREE.Vector3 | null];
  head: THREE.Vector3 | null;
  /** Head yaw and pitch in radians, from the nose's offset between the ears. */
  headYaw: number;
  headPitch: number;
  headRoll: number;
  /** Metres per normalized unit for this frame. */
  scale: number;
  /**
   * The normalization centre this frame used, in image units.
   *
   * Exposed so a caller holding a point that is NOT in the pose block — the hand
   * landmarker's own wrist, which is more accurate than the pose model's — can
   * place it through the identical transform. Re-deriving it there would be two
   * implementations of one mapping, and they would drift.
   */
  centreX: number;
  centreY: number;
}

export function makeBodyMeasurement(): BodyMeasurement {
  return {
    valid: false,
    shoulderL: new THREE.Vector3(),
    shoulderR: new THREE.Vector3(),
    shoulderMid: new THREE.Vector3(),
    pelvis: new THREE.Vector3(),
    side: new THREE.Vector3(1, 0, 0),
    up: new THREE.Vector3(0, 1, 0),
    forward: new THREE.Vector3(0, 0, 1),
    wrist: [null, null],
    elbow: [null, null],
    head: null,
    headYaw: 0,
    headPitch: 0,
    headRoll: 0,
    scale: 1,
    centreX: 0,
    centreY: 0,
  };
}

/**
 * Where the nose sits between the ears on a head looking straight ahead, in
 * head-radii.
 *
 * MEASURED across the whole motion library, not chosen. Without subtracting it
 * the entire library plays back with the chin permanently raised about eleven
 * degrees, which reads — accurately but unhelpfully — as disdain.
 */
const NEUTRAL_NOSE_RISE = 0.158;

/** Below this the pose block is a misdetection, not a signer. */
const MIN_SHOULDER_SPAN = 0.02;

/**
 * Read one 13-point pose block into model space.
 *
 * `block` holds 39 floats — 13 points × (x, y, z) — in normalized image
 * coordinates with x and z already scaled so all three axes share a unit. A
 * point the tracker could not see is exact zeros, which is the absence
 * convention the whole pipeline uses.
 */
export function measureBody(
  block: Float32Array,
  offset: number,
  rest: RestPose,
  out: BodyMeasurement,
): BodyMeasurement {
  const present = (slot: number) =>
    block[offset + slot * 3] !== 0 || block[offset + slot * 3 + 1] !== 0;

  if (!present(POSE.L_SHOULDER) || !present(POSE.R_SHOULDER)) {
    out.valid = false;
    return out;
  }

  const lx = block[offset + POSE.L_SHOULDER * 3];
  const ly = block[offset + POSE.L_SHOULDER * 3 + 1];
  const lz = block[offset + POSE.L_SHOULDER * 3 + 2];
  const rx = block[offset + POSE.R_SHOULDER * 3];
  const ry = block[offset + POSE.R_SHOULDER * 3 + 1];
  const rz = block[offset + POSE.R_SHOULDER * 3 + 2];

  const span = Math.hypot(lx - rx, ly - ry);
  if (span < MIN_SHOULDER_SPAN) {
    out.valid = false;
    return out;
  }

  const cx = (lx + rx) / 2;
  const cy = (ly + ry) / 2;
  const cz = (lz + rz) / 2;

  /* Normalized units → metres. The model's own shoulder span is the ruler, so a
     signer at any distance produces an avatar of the right proportions. */
  const scale = rest.shoulderWidth / span;
  out.scale = scale;
  out.centreX = cx;
  out.centreY = cy;

  // The model's own shoulder line, so the figure is anchored to its own body
  // rather than to wherever the signer happened to stand in frame.
  const anchor = B.anchor.copy(rest.shoulderMid);

  // Image space → model space. y is flipped because image y grows downward, and
  // z because MediaPipe places the camera at negative z.
  const place = (slot: number, target: THREE.Vector3): THREE.Vector3 => {
    const base = offset + slot * 3;
    return target.set(
      (block[base] - cx) * scale + anchor.x,
      -(block[base + 1] - cy) * scale + anchor.y,
      -(block[base + 2] - cz) * scale + anchor.z,
    );
  };

  place(POSE.L_SHOULDER, out.shoulderL);
  place(POSE.R_SHOULDER, out.shoulderR);
  out.shoulderMid.copy(out.shoulderL).add(out.shoulderR).multiplyScalar(0.5);

  /* Hips. Out of shot in most sign footage — the reference clips are framed
     chest-up — so an anatomical fallback keeps the torso from being a guess that
     changes shape frame to frame. */
  if (present(POSE.L_HIP) && present(POSE.R_HIP)) {
    place(POSE.L_HIP, B.hipL);
    place(POSE.R_HIP, B.hipR);
    out.pelvis.copy(B.hipL).add(B.hipR).multiplyScalar(0.5);
  } else {
    out.pelvis.copy(out.shoulderMid);
    out.pelvis.y -= rest.torsoLength;
  }

  out.side.copy(out.shoulderL).sub(out.shoulderR);
  if (out.side.lengthSq() < 1e-10) out.side.set(1, 0, 0);
  out.side.normalize();

  B.down.copy(out.pelvis).sub(out.shoulderMid);
  if (B.down.lengthSq() < 1e-10) B.down.set(0, -1, 0);
  B.down.normalize();

  out.forward.crossVectors(B.down, out.side);
  if (out.forward.lengthSq() < 1e-10) out.forward.set(0, 0, 1);
  out.forward.normalize();
  out.up.crossVectors(out.forward, out.side).normalize();

  out.wrist[0] = present(POSE.L_WRIST) ? place(POSE.L_WRIST, B.wristL) : null;
  out.wrist[1] = present(POSE.R_WRIST) ? place(POSE.R_WRIST, B.wristR) : null;
  out.elbow[0] = present(POSE.L_ELBOW) ? place(POSE.L_ELBOW, B.elbowL) : null;
  out.elbow[1] = present(POSE.R_ELBOW) ? place(POSE.R_ELBOW, B.elbowR) : null;

  /* Depth has to be REBUILT, not believed. See `resolveDepth`. */
  for (let side = 0; side < 2; side += 1) {
    const shoulder = side === 0 ? out.shoulderL : out.shoulderR;
    const lengths = rest.armLengths[side];
    const elbow = out.elbow[side];
    const wrist = out.wrist[side];
    if (elbow) {
      resolveDepth(shoulder, elbow, lengths.upper);
      if (wrist) resolveDepth(elbow, wrist, lengths.lower);
    } else if (wrist) {
      // No elbow seen: place the wrist on the arm's full-reach sphere instead.
      // Less informative, but still guaranteed reachable, which is the property
      // that matters — an unreachable goal makes the IK snap the arm straight.
      resolveDepth(shoulder, wrist, lengths.upper + lengths.lower);
    }
  }

  /* ------------------------------------------------------------------ head -- */
  const hasEars = present(POSE.L_EAR) && present(POSE.R_EAR);
  const hasNose = present(POSE.NOSE);
  out.headYaw = 0;
  out.headPitch = 0;
  out.headRoll = 0;

  if (hasEars) {
    place(POSE.L_EAR, B.earL);
    place(POSE.R_EAR, B.earR);
    out.head = B.head.copy(B.earL).add(B.earR).multiplyScalar(0.5);
    // The ear canal sits below and behind the centre of the skull.
    out.head.y += rest.headHeight * 0.13;
    // Roll: how far the ear line departs from the shoulder line.
    B.earAxis.copy(B.earL).sub(B.earR).normalize();
    out.headRoll = soften(Math.asin(THREE.MathUtils.clamp(B.earAxis.dot(out.up), -1, 1)), 0.5);
  } else if (hasNose) {
    out.head = place(POSE.NOSE, B.head);
    out.head.y += rest.headHeight * 0.16;
  } else {
    out.head = null;
  }

  if (hasNose && hasEars) {
    place(POSE.NOSE, B.nose);
    const halfSpan = Math.max(1e-4, B.earL.distanceTo(B.earR) / 2);
    B.centre.copy(B.earL).add(B.earR).multiplyScalar(0.5);
    B.offset.copy(B.nose).sub(B.centre);

    /* Yaw from where the nose sits between the ears; pitch from how high it sits
       relative to them. Both are ratios of the head's own dimensions, so they do
       not change with distance from the camera.

       Softened rather than clamped. This estimate is noisy — the divisor is an
       ear separation that itself shrinks as the head turns, so it overshoots
       badly on the tail — and a hard clamp turns every overshoot into a head
       SNAPPED to its limit and sitting there. More than a tenth of the library's
       frames overshoot. A saturation turns those into a head that leans a lot,
       which is what the signer was doing. */
    out.headYaw = soften((B.offset.dot(out.side) / halfSpan) * 0.45, 0.45);
    const rise = B.offset.dot(out.up) / Math.max(1e-4, rest.headHeight * 0.5);
    out.headPitch = soften(-(rise + NEUTRAL_NOSE_RISE) * 0.9, 0.32);
  }

  out.valid = true;
  return out;
}

/**
 * Rebuild one bone's depth from its length, keeping the tracked screen position.
 *
 * ---------------------------------------------------------------------------
 * WHY POSE Z CANNOT BE USED AS A DEPTH
 * ---------------------------------------------------------------------------
 * MediaPipe's pose z is documented as roughly the scale of x. Measured over all
 * 14,299 shoulder-to-wrist pairs in this library it is not:
 *
 *     planar reach (x, y)   p50 0.75   p90 1.47   shoulder widths
 *     |pose dz|             p50 2.83   p90 4.08   shoulder widths
 *     arm length                 1.50             shoulder widths
 *
 * The planar figures are sensible and bounded by the arm. The depths are around
 * twice the length of the whole arm — pose z is an ORDERING signal (which side
 * of the body a joint is on), not a metric. Trusting it as a position put wrist
 * targets 1.4 m in front of the chest, which an end-to-end check caught as the
 * hand landing 145% of a shoulder width from where the sign put it.
 *
 * So z is rebuilt rather than read:
 *
 *     dz    = clamp(z_child − z_parent, ±√(L² − d²))
 *     scale = √(L² − dz²) / d
 *     child = parent + (dxy · scale, dz)
 *
 * where `d` is the on-screen distance and `L` the bone's true length. The result
 * is exactly L long whatever the tracker said, it leans the way the tracker
 * leaned, and it degrades to a flat-but-correct arm when there is no depth hint
 * at all. Screen position — the part MediaPipe is genuinely good at — is
 * preserved up to that one scale factor.
 *
 * The wrist is then always inside the arm's reach, which matters more than it
 * sounds: an unreachable goal makes an IK solver snap the limb straight, and it
 * does so on exactly the frames where the hand is nearest the camera.
 */
export function resolveDepth(
  parent: THREE.Vector3,
  child: THREE.Vector3,
  length: number,
): void {
  const dx = child.x - parent.x;
  const dy = child.y - parent.y;
  const planar = Math.hypot(dx, dy);
  const maxDz = Math.sqrt(Math.max(0, length * length - planar * planar));

  // The hint's magnitude is meaningless, so only its sign and its saturation are
  // used: a large reported depth becomes "as far forward as the bone allows".
  const hint = child.z - parent.z;
  const dz = Math.abs(hint) > 1e-6 ? Math.sign(hint) * maxDz : maxDz * 0.35;

  const remaining = Math.sqrt(Math.max(0, length * length - dz * dz));
  if (planar > 1e-5) {
    const scale = remaining / planar;
    child.set(parent.x + dx * scale, parent.y + dy * scale, parent.z + dz);
  } else {
    // Directly on top of the parent on screen: hang the bone downward, which is
    // where an arm goes when it is pointing at the camera and foreshortened to
    // nothing.
    child.set(parent.x, parent.y - remaining, parent.z + dz);
  }
}

/**
 * Turn a body measurement into spine rotations, and produce the frame the arms
 * are solved against.
 */
export function solveBody(
  measurement: BodyMeasurement,
  rest: RestPose,
  pose: HumanoidPose,
  frame: BodyFrame,
): void {
  /* Chest orientation. `basisRotation(forward, up)` is the identity when the
     figure faces +Z with +Y up, which is exactly the VRM rest pose — so a signer
     square to the camera produces zero spine rotation rather than a small
     permanent twist. */
  basisRotation(measurement.forward, measurement.up, B.chest);

  /* Distribute the turn down the spine instead of putting it all in one joint.
     A torso that rotates entirely at the waist reads as a figure on a turntable;
     real rotation accumulates through the lumbar and thoracic spine, and the
     weights below are roughly the proportion of axial rotation each region
     actually contributes. */
  const share: [BoneName, number][] = [
    ["hips", 0.12],
    ["spine", 0.30],
    ["chest", 0.32],
    ["upperChest", 0.26],
  ];

  B.accumulated.identity();
  for (const [bone, weight] of share) {
    const rotation = pose.rotations.get(bone);
    const boneRest = rest.bones.get(bone);
    if (!rotation || !boneRest) continue;
    // The remaining rotation still to be distributed, in this bone's parent
    // frame — so the shares compose to the target rather than each aiming at it.
    B.remaining.copy(B.accumulated).invert().multiply(B.chest);
    rotation.identity().slerp(B.remaining, weight / remainingWeight(share, bone));
    const limit = JOINT_LIMITS[bone];
    if (limit) {
      clampInFrame(rotation, boneRest.frame, limit.flex, limit.spread, limit.twist);
    }
    B.accumulated.multiply(rotation);
  }

  frame.chestRotation.copy(B.accumulated);
  frame.chestSide.copy(measurement.side);
  frame.shoulder[0].copy(measurement.shoulderL);
  frame.shoulder[1].copy(measurement.shoulderR);
}

/** How much of the total turn is left to give out from this bone onward. */
function remainingWeight(
  share: readonly [BoneName, number][],
  from: BoneName,
): number {
  let total = 0;
  let seen = false;
  for (const [bone, weight] of share) {
    if (bone === from) seen = true;
    if (seen) total += weight;
  }
  return total || 1;
}

/**
 * Neck and head, including the non-manual markers that carry clause type.
 *
 * `time` drives the periodic markers — the negation head shake runs at about
 * 2.6 Hz, which is the rate a signed one actually runs at. Much faster reads as
 * a shiver; much slower reads as ordinary disagreement rather than as the
 * grammatical negation it is.
 */
export function solveHead(
  measurement: BodyMeasurement,
  expression: NonManual,
  time: number,
  rest: RestPose,
  chestRotation: THREE.Quaternion,
  pose: HumanoidPose,
): void {
  const yaw = measurement.headYaw + Math.sin(time * 16) * expression.headShake * 0.22;
  const pitch =
    measurement.headPitch +
    expression.headForward +
    Math.sin(time * 13) * expression.headNod * 0.13;
  const roll = measurement.headRoll + expression.headTilt;

  // Split between neck and head. The neck takes the smaller share: a head that
  // turns entirely at the base of the neck reads as a bobblehead, and one that
  // turns entirely at the skull reads as a periscope.
  const split: [BoneName, number][] = [
    ["neck", 0.4],
    ["head", 0.6],
  ];

  B.accumulated.copy(chestRotation);
  for (const [bone, weight] of split) {
    const rotation = pose.rotations.get(bone);
    const boneRest = rest.bones.get(bone);
    if (!rotation || !boneRest) continue;

    // Build the share in the bone's own canonical frame, so "pitch" means
    // nodding forward on every model rather than whichever axis the artist used.
    B.euler.set(pitch * weight, yaw * weight, roll * weight);
    B.local.setFromAxisAngle(CANONICAL_X, B.euler.x);
    B.spin.setFromAxisAngle(CANONICAL_Y, B.euler.y);
    B.local.multiply(B.spin);
    B.spin.setFromAxisAngle(CANONICAL_Z, B.euler.z);
    B.local.multiply(B.spin);

    B.inverseFrame.copy(boneRest.frame).invert();
    rotation.copy(boneRest.frame).multiply(B.local).multiply(B.inverseFrame);

    const limit = JOINT_LIMITS[bone];
    if (limit) {
      clampInFrame(rotation, boneRest.frame, limit.flex, limit.spread, limit.twist);
    }
    B.accumulated.multiply(rotation);
  }
}

const CANONICAL_X = new THREE.Vector3(1, 0, 0);
const CANONICAL_Y = new THREE.Vector3(0, 1, 0);
const CANONICAL_Z = new THREE.Vector3(0, 0, 1);

const B = {
  anchor: new THREE.Vector3(),
  hipL: new THREE.Vector3(),
  hipR: new THREE.Vector3(),
  down: new THREE.Vector3(),
  wristL: new THREE.Vector3(),
  wristR: new THREE.Vector3(),
  elbowL: new THREE.Vector3(),
  elbowR: new THREE.Vector3(),
  earL: new THREE.Vector3(),
  earR: new THREE.Vector3(),
  earAxis: new THREE.Vector3(),
  nose: new THREE.Vector3(),
  head: new THREE.Vector3(),
  centre: new THREE.Vector3(),
  offset: new THREE.Vector3(),
  euler: new THREE.Vector3(),
  chest: new THREE.Quaternion(),
  accumulated: new THREE.Quaternion(),
  remaining: new THREE.Quaternion(),
  local: new THREE.Quaternion(),
  spin: new THREE.Quaternion(),
  inverseFrame: new THREE.Quaternion(),
};

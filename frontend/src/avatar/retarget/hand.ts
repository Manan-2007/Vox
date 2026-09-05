/**
 * Twenty-one landmarks in, fifteen bone rotations out.
 *
 * ---------------------------------------------------------------------------
 * THIS IS THE FILE THE PRODUCT IS ABOUT
 * ---------------------------------------------------------------------------
 * Vox exists so a Deaf user can read a sign. Handshape is one of the five
 * parameters that distinguish one sign from another — with location, movement,
 * orientation and non-manuals — and it is the one the previous build could not
 * render at all. Its hands were cylinders drawn between landmark positions, so a
 * bent finger was a chain of beads that gapped on the outside of the curve and
 * interpenetrated on the inside, and a palm was a box with no way to tell which
 * face was the palm.
 *
 * The fix is not better geometry. It is a change of representation: solve for
 * ROTATIONS and let a skinned mesh deform. A rotation can be limited to what a
 * knuckle can actually do, blended without shortening the finger, and retargeted
 * onto any hand — none of which a position can.
 *
 * ---------------------------------------------------------------------------
 * THE ALGORITHM
 * ---------------------------------------------------------------------------
 * Two steps, both exact — no fitting, no optimisation, no iteration.
 *
 * 1. WRIST. Build an orthonormal frame from the measured hand:
 *
 *        up      = wrist → middle knuckle       (the hand's long axis)
 *        forward = the palmar normal            (which way the palm faces)
 *
 *    The model's rest hand has the same frame, measured at load
 *    (`rest.ts` → `BoneRest.frame`). The world rotation of the hand bone is
 *    whatever takes one onto the other:
 *
 *        handWorld = measuredFrame ⋅ restFrame⁻¹
 *
 *    Taking a full 3-DOF frame rather than just a direction is what makes palm
 *    orientation readable. A direction alone leaves the hand free to spin about
 *    its own axis, and palm-up versus palm-down is the whole difference between
 *    several pairs of signs.
 *
 * 2. FINGERS. Walk each chain from knuckle to tip. For bone i, the measured
 *    direction is landmark[i] → landmark[i+1]. Express it in the parent's
 *    current frame, take the minimal rotation from the bone's rest direction,
 *    clamp it to anatomy, and compose forward so the next bone inherits the
 *    clamped result:
 *
 *        desiredLocal = parentWorld⁻¹ ⋅ measuredDirection
 *        local        = minimalRotation(restAxis → desiredLocal)
 *        local        = clampToAnatomy(local)
 *        parentWorld  = parentWorld ⋅ local
 *
 *    Minimal rotation is the right choice specifically because it introduces no
 *    twist: a phalanx is a cylinder and carries no measurable roll, so any twist
 *    invented here would be noise rendered as a finger rotating about itself.
 *    Clamping BEFORE composing forward is what stops one bad joint estimate from
 *    cascading down the finger.
 *
 * ---------------------------------------------------------------------------
 * WHY WORLD LANDMARKS
 * ---------------------------------------------------------------------------
 * MediaPipe's normalized image landmarks have a per-point z that is a weak
 * guess, so a hand built from them is a flat constellation. `hand_world_landmarks`
 * are 21 points in metres with real depth and real proportions. Shape comes from
 * those; WHERE the hand is comes from the arm solve. See ml/build_motion.py.
 */

import * as THREE from "three";
import type { BoneName, HumanoidPose, Side } from "../../core/humanoid";
import { HAND, JOINT_LIMITS, fingerChains } from "../../core/humanoid";
import {
  basisRotation,
  clamp,
  clampInFrame,
  makeSwingTwist,
  quatToRotationVector,
  rotationBetween,
  rotationVectorToQuat,
  smoothstep,
  swingTwist,
} from "../../core/math";
import type { RestPose } from "./rest";

/**
 * MediaPipe world landmarks → scene space.
 *
 * MediaPipe reports x right, y DOWN and z toward the camera; three.js is y-up.
 * The conversion is a 180° rotation about x, which flips y and z together.
 *
 * Flipping BOTH is essential and easy to get wrong. Flipping y alone is a
 * reflection: its determinant is −1, so it turns a left hand into a right hand.
 * Nothing about that failure is obvious on screen — the fingers still bend, the
 * palm still faces somewhere — but every handshape comes out mirrored, and a
 * mirrored handshape is a different sign or no sign at all. Flipping y and z
 * together has determinant +1 and preserves chirality.
 *
 * `src` and `dst` may be the same array.
 */
export function mediapipeToScene(
  src: Float32Array,
  srcOffset: number,
  dst: Float32Array,
  dstOffset = 0,
): void {
  for (let i = 0; i < HAND.COUNT; i += 1) {
    const s = srcOffset + i * 3;
    const d = dstOffset + i * 3;
    const x = src[s];
    const y = src[s + 1];
    const z = src[s + 2];
    dst[d] = x;
    dst[d + 1] = -y;
    dst[d + 2] = -z;
  }
}

/** The measured geometry of one hand, in scene space. */
export interface HandFrame {
  /** False when the landmarks are absent or degenerate. */
  valid: boolean;
  /** Wrist → middle knuckle, normalized. */
  axis: THREE.Vector3;
  /** Unit normal pointing out of the PALM. */
  palmar: THREE.Vector3;
  /** Wrist-to-middle-knuckle distance in metres — the hand's own ruler. */
  span: number;
  /** World rotation for the hand bone. */
  rotation: THREE.Quaternion;
}

export function makeHandFrame(): HandFrame {
  return {
    valid: false,
    axis: new THREE.Vector3(0, 1, 0),
    palmar: new THREE.Vector3(0, 0, 1),
    span: 0.09,
    rotation: new THREE.Quaternion(),
  };
}

/**
 * The smallest hand, in metres, that is worth believing.
 *
 * MediaPipe occasionally returns a collapsed world-landmark set — every point
 * within a millimetre of the origin — when it is tracking something that is not
 * a hand. Rendering one produces a spike of garbage rotations, because every
 * direction in the solve is then the normalization of near-zero noise.
 */
const MIN_SPAN = 0.02;
/** And the largest. An adult hand is ~9 cm wrist to middle knuckle. */
const MAX_SPAN = 0.20;

/**
 * Measure one hand and derive its wrist rotation.
 *
 * `points` holds 21 scene-space landmarks (see `mediapipeToScene`) starting at
 * `offset`.
 */
export function measureHand(
  points: Float32Array,
  offset: number,
  side: Side,
  rest: RestPose,
  out: HandFrame,
): HandFrame {
  const read = (index: number, target: THREE.Vector3) =>
    target.set(
      points[offset + index * 3],
      points[offset + index * 3 + 1],
      points[offset + index * 3 + 2],
    );

  read(HAND.WRIST, S.wrist);
  read(HAND.MIDDLE_MCP, S.middle);
  read(HAND.INDEX_MCP, S.index);
  read(HAND.LITTLE_MCP, S.little);

  S.axis.copy(S.middle).sub(S.wrist);
  const span = S.axis.length();
  if (!(span > MIN_SPAN) || span > MAX_SPAN) {
    out.valid = false;
    return out;
  }
  S.axis.divideScalar(span);

  // The knuckle line, made perpendicular to the hand's long axis so the two
  // together span the palm plane rather than a skewed pair.
  S.knuckle.copy(S.little).sub(S.index);
  S.knuckle.addScaledVector(S.axis, -S.knuckle.dot(S.axis));
  if (S.knuckle.lengthSq() < 1e-8) {
    out.valid = false;
    return out;
  }
  S.knuckle.normalize();

  // Same handedness rule as the rest capture, so the two frames are comparable.
  // Getting this backwards on one side and not the other is the classic way to
  // end up with one correct hand and one that signs everything inside out.
  if (side === "left") S.dorsal.crossVectors(S.axis, S.knuckle);
  else S.dorsal.crossVectors(S.knuckle, S.axis);
  if (S.dorsal.lengthSq() < 1e-8) {
    out.valid = false;
    return out;
  }
  S.dorsal.normalize();

  out.axis.copy(S.axis);
  out.palmar.copy(S.dorsal).negate();
  out.span = span;
  out.valid = true;

  // measuredFrame ⋅ restFrame⁻¹ — see the module header.
  basisRotation(out.palmar, out.axis, S.measured);
  const handBone: BoneName = side === "left" ? "leftHand" : "rightHand";
  const restFrame = rest.bones.get(handBone)?.frame;
  if (restFrame) {
    S.inverseRest.copy(restFrame).invert();
    out.rotation.copy(S.measured).multiply(S.inverseRest).normalize();
  } else {
    out.rotation.copy(S.measured);
  }
  return out;
}

/**
 * Solve the fifteen finger bones of one hand and write them into `pose`.
 *
 * `handWorld` is the world rotation the hand bone ended up with AFTER the arm
 * solve and its own limits — not the raw measurement. Passing the clamped value
 * is what keeps the fingers attached to the wrist the viewer can actually see.
 */
export function solveFingers(
  points: Float32Array,
  offset: number,
  side: Side,
  rest: RestPose,
  handWorld: THREE.Quaternion,
  pose: HumanoidPose,
): void {
  const chains = side === "left" ? LEFT_CHAINS : RIGHT_CHAINS;

  for (const chain of chains) {
    S.parentWorld.copy(handWorld);

    // The three measured segment directions, in scene space. Solving needs the
    // FIRST TWO together — see `solveProximal` — so they are gathered up front
    // rather than one at a time.
    let usable = 0;
    for (let segment = 0; segment < 3; segment += 1) {
      const from = chain.landmarks[segment];
      const to = chain.landmarks[segment + 1];
      const base = offset + from * 3;
      const tip = offset + to * 3;
      const d = S.segments[segment];
      d.set(
        points[tip] - points[base],
        points[tip + 1] - points[base + 1],
        points[tip + 2] - points[base + 2],
      );
      if (d.lengthSq() < 1e-10) break;
      d.normalize();
      usable = segment + 1;
    }

    for (let segment = 0; segment < 3; segment += 1) {
      const bone = chain.bones[segment];
      const target = pose.rotations.get(bone);
      const boneRest = rest.bones.get(bone);
      if (!target || !boneRest) continue;

      if (segment >= usable) {
        // Nothing measured for this segment. Leaving the bone at its previous
        // value rather than resetting it keeps a momentary landmark collapse
        // from flicking the finger straight for one frame.
        S.parentWorld.multiply(target);
        continue;
      }

      S.inverseParent.copy(S.parentWorld).invert();
      S.local.copy(S.segments[segment]).applyQuaternion(S.inverseParent);

      if (segment === 0) {
        solveProximal(chain.finger, boneRest, usable, target);
      } else {
        // Minimal rotation for the hinges. With the proximal's roll now solved
        // from the bend plane, "minimal" here lands on almost pure flexion,
        // which is what a hinge is.
        S.hint.set(0, 0, 1).applyQuaternion(boneRest.frame);
        rotationBetween(boneRest.axis, S.local, S.hint, target);
      }

      const limit = JOINT_LIMITS[bone];
      if (limit) {
        clampInFrame(
          target,
          boneRest.frame,
          limit.flex,
          limit.spread,
          limit.twist,
        );
      }

      S.parentWorld.multiply(target);
    }
  }
}

/**
 * The proximal phalanx, whose ROLL cannot be measured directly and must be
 * inferred from the joint below it.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS IS NOT JUST "POINT THE BONE THE RIGHT WAY"
 * ---------------------------------------------------------------------------
 * A landmark pair fixes a bone's DIRECTION and says nothing about its rotation
 * about that direction — a cylinder looks the same however it is rolled. For the
 * middle and distal phalanges that genuinely does not matter. For the proximal
 * it matters enormously, because the proximal's roll determines the axis the PIP
 * hinge swings on, and every joint below inherits it.
 *
 * Taking the minimal rotation (zero roll) is the obvious choice and it is
 * measurably wrong. Run it over the whole library and the rotation the PIP then
 * demands has a 5th–95th percentile spread of ±40° about its lateral axis. A PIP
 * has perhaps 6°. The excess is not noise — temporal smoothing barely touches it
 * — it is pure flexion being mis-attributed to spread because it is being
 * measured against a hinge axis pointing the wrong way.
 *
 * The fix is that the bend itself reveals the roll. Two successive segment
 * directions define a plane, and its normal IS the hinge axis:
 *
 *     hinge = normalize(proximalDirection × intermediateDirection)
 *
 * Rolling the proximal so its canonical flexion axis lands on that hinge makes
 * the PIP a true hinge, and the spread collapses to the few degrees of real
 * lateral play.
 *
 * A straight finger has no bend plane, so the normal is undefined and noisy —
 * hence the smooth blend back to the rest axis as the bend vanishes. That blend
 * is not cosmetic: a hard switch would make a finger visibly snap about its own
 * axis every time it straightened.
 */
function solveProximal(
  finger: string,
  boneRest: { axis: THREE.Vector3; frame: THREE.Quaternion },
  usable: number,
  out: THREE.Quaternion,
): void {
  // Where the flexion axis sits with the bone unrolled — the fallback, and the
  // reference that keeps the solved hinge on the correct side.
  S.expected.set(1, 0, 0).applyQuaternion(boneRest.frame);
  S.expected.applyQuaternion(S.inverseParent).normalize();

  S.hinge.copy(S.expected);

  if (usable >= 2) {
    S.local2.copy(S.segments[1]).applyQuaternion(S.inverseParent);
    S.bend.crossVectors(S.local, S.local2);
    const bend = S.bend.length();
    if (bend > 1e-6) {
      S.bend.divideScalar(bend);
      // A hinge opposed to the rest axis means the joint read as hyperextended,
      // which a PIP cannot be. That is a bad measurement, not a bad finger.
      if (S.bend.dot(S.expected) < 0) S.bend.copy(S.expected);
      // `bend` is sin(angle between the segments): fade the measured hinge in
      // between about 6° and 15° of flexion, below which it is mostly noise.
      const weight = smoothstep((bend - 0.10) / 0.16);
      S.hinge.lerpVectors(S.expected, S.bend, weight);
      if (S.hinge.lengthSq() < 1e-8) S.hinge.copy(S.expected);
      S.hinge.normalize();
    }
  }

  // The thumb's metacarpal rolls to bring the pad round to face the fingers —
  // that is opposition, and it is a genuine degree of freedom rather than an
  // artefact of the bend plane. Trusting the measured hinge fully here would
  // make the thumb's roll follow its bend, which flattens opposition into a
  // simple curl and turns every pinch into a claw.
  if (finger === "thumb") {
    S.hinge.lerp(S.expected, 0.45);
    if (S.hinge.lengthSq() < 1e-8) S.hinge.copy(S.expected);
    S.hinge.normalize();
  }

  // Make the hinge perpendicular to the bone before building the frame; the two
  // are never exactly perpendicular in measured data.
  S.hinge.addScaledVector(S.local, -S.hinge.dot(S.local));
  if (S.hinge.lengthSq() < 1e-8) S.hinge.copy(S.expected);
  S.hinge.normalize();

  // Build the rotation taking the canonical axes onto (hinge, direction, ...),
  // then remove the bone's own rest frame so what is left is a local rotation.
  //
  //   basisRotation(forward, up) maps  Y → up,  Z → forward,  X → up × forward
  //   so `forward = hinge × direction` puts X on the hinge.
  S.forward.crossVectors(S.hinge, S.local);
  basisRotation(S.forward, S.local, S.measured);
  S.inverseRest.copy(boneRest.frame).invert();
  out.copy(S.measured).multiply(S.inverseRest).normalize();
}

/**
 * Push neighbouring fingers apart when their spread rotations would cross.
 *
 * Landmark noise routinely puts the ring finger a couple of degrees inside the
 * middle finger, and a skinned mesh renders that as two fingers occupying the
 * same space. The eye reads intersecting fingers as a rendering fault and stops
 * trusting the handshape, which for this product is the whole failure.
 *
 * Deliberately a *minimum separation on the spread angle* rather than mesh
 * collision. Collision is expensive, needs per-model geometry, and — the real
 * objection — resolves by pushing fingers out of the palm plane, which changes
 * the handshape. Separating on the one axis that caused the overlap preserves
 * flexion, which is where the meaning lives.
 */
export function separateFingers(
  side: Side,
  rest: RestPose,
  pose: HumanoidPose,
  minimumRadians = 0.012,
): void {
  const chains = side === "left" ? LEFT_CHAINS : RIGHT_CHAINS;
  // Index, middle, ring, little. The thumb is excluded: it crosses the palm on
  // purpose in most handshapes, and separating it from the index would break
  // every pinch and every closed fist.
  const knuckles = SEPARATION;
  let count = 0;

  for (let i = 1; i < chains.length && count < knuckles.length; i += 1) {
    const bone = chains[i].bones[0];
    const rotation = pose.rotations.get(bone);
    const boneRest = rest.bones.get(bone);
    if (!rotation || !boneRest) continue;
    const entry = knuckles[count];
    entry.bone = bone;
    entry.rest = boneRest;
    decompose(rotation, boneRest.frame, entry.swing, entry.twist);
    // Work inside the joint's own range from the start. Separating first and
    // clamping afterwards silently undoes the correction whenever a knuckle is
    // already at its spread limit, which is exactly when fingers are most
    // splayed and most likely to have crossed.
    const limit = JOINT_LIMITS[bone];
    entry.spread = limit
      ? clamp(entry.swing.z, limit.spread[0], limit.spread[1])
      : entry.swing.z;
    count += 1;
  }
  if (count < 2) return;

  /* Fingers run index → little, so their spread angles must stay in that order.
     `sign` is which way the order runs in the canonical frame, which mirrors
     across the body — hence the per-side flip.

     The minimum gap is small on purpose. At rest the four fingers are PARALLEL
     and simply offset across the palm by about 17 mm, so equal spread angles do
     not put them in the same place — a flat hand is exactly that. Demanding a
     large positive gap would fan the fingers permanently apart, which is a
     different handshape. All that has to be prevented is the ORDER inverting,
     which is a genuine crossing.

     Two passes, because one is not a fixed point: pushing the pair at i apart
     can close the pair at i−1, and a single forward sweep leaves that behind.
     A third pass never changes anything measurably. */
  const sign = side === "left" ? -1 : 1;
  let corrected = false;

  for (let pass = 0; pass < 2; pass += 1) {
    for (let i = 1; i < count; i += 1) {
      const previous = knuckles[i - 1];
      const current = knuckles[i];
      const gap = (current.spread - previous.spread) * sign;
      if (gap >= minimumRadians) continue;
      // Split the correction between the pair, so neither finger is singled out
      // and the hand stays symmetric about the middle of the overlap.
      const push = (minimumRadians - gap) * 0.5 * sign;
      previous.spread -= push;
      current.spread += push;
      corrected = true;
    }
  }
  if (!corrected) return;

  for (let i = 0; i < count; i += 1) {
    const entry = knuckles[i];
    const rotation = pose.rotations.get(entry.bone);
    if (!rotation || !entry.rest) continue;
    entry.swing.z = entry.spread;
    recompose(rotation, entry.rest.frame, entry.swing, entry.twist);
    // Re-clamp. Pushing a finger apart can push it past its own limit, and an
    // over-spread knuckle is exactly as wrong as the overlap it was fixing.
    const limit = JOINT_LIMITS[entry.bone];
    if (limit) {
      clampInFrame(rotation, entry.rest.frame, limit.flex, limit.spread, limit.twist);
    }
  }
}

/**
 * Split a local rotation into its canonical swing vector (x = flex, z = spread)
 * and its twist, so one axis can be edited without disturbing the others.
 *
 * Reading the spread off the quaternion's z component directly is the tempting
 * shortcut and it is wrong: that is only the spread angle when the rotation is a
 * pure z rotation, and for a flexed finger it is off by tens of degrees. Editing
 * on that reading corrupts the flexion, which is where the meaning is.
 */
function decompose(
  rotation: THREE.Quaternion,
  frame: THREE.Quaternion,
  swingOut: THREE.Vector3,
  twistOut: THREE.Quaternion,
): void {
  S.inverseFrame.copy(frame).invert();
  S.canonical.copy(S.inverseFrame).multiply(rotation).multiply(frame);
  swingTwist(S.canonical, CANONICAL_Y, S.split);
  quatToRotationVector(S.split.swing, swingOut);
  twistOut.copy(S.split.twist);
}

/** The inverse of `decompose`. */
function recompose(
  out: THREE.Quaternion,
  frame: THREE.Quaternion,
  swing: THREE.Vector3,
  twist: THREE.Quaternion,
): void {
  rotationVectorToQuat(swing, S.split.swing);
  S.canonical.copy(S.split.swing).multiply(twist);
  S.inverseFrame.copy(frame).invert();
  out.copy(frame).multiply(S.canonical).multiply(S.inverseFrame).normalize();
}

const CANONICAL_Y = new THREE.Vector3(0, 1, 0);

interface SeparationEntry {
  bone: BoneName;
  rest: { frame: THREE.Quaternion } | null;
  swing: THREE.Vector3;
  twist: THREE.Quaternion;
  spread: number;
}

const SEPARATION: SeparationEntry[] = Array.from({ length: 4 }, () => ({
  bone: "leftIndexProximal" as BoneName,
  rest: null,
  swing: new THREE.Vector3(),
  twist: new THREE.Quaternion(),
  spread: 0,
}));

const LEFT_CHAINS = fingerChains("left");
const RIGHT_CHAINS = fingerChains("right");

const S = {
  wrist: new THREE.Vector3(),
  middle: new THREE.Vector3(),
  index: new THREE.Vector3(),
  little: new THREE.Vector3(),
  axis: new THREE.Vector3(),
  knuckle: new THREE.Vector3(),
  dorsal: new THREE.Vector3(),
  segments: [new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()],
  local: new THREE.Vector3(),
  local2: new THREE.Vector3(),
  hint: new THREE.Vector3(),
  hinge: new THREE.Vector3(),
  bend: new THREE.Vector3(),
  expected: new THREE.Vector3(),
  forward: new THREE.Vector3(),
  measured: new THREE.Quaternion(),
  inverseRest: new THREE.Quaternion(),
  parentWorld: new THREE.Quaternion(),
  inverseParent: new THREE.Quaternion(),
  inverseFrame: new THREE.Quaternion(),
  canonical: new THREE.Quaternion(),
  split: makeSwingTwist(),
};

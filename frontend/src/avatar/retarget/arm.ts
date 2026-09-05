/**
 * Arms and torso: two-bone IK, plus the spine and head that carry ISL's
 * non-manual grammar.
 *
 * ---------------------------------------------------------------------------
 * WHY IK AND NOT "COPY THE TRACKED ELBOW"
 * ---------------------------------------------------------------------------
 * MediaPipe reports an elbow position, and using it directly is what the previous
 * build did. Three things go wrong, and all three were visible on screen:
 *
 *   * RUBBER BONES. A limb whose ends are two independently tracked points is
 *     as long as the tracker says. Under foreshortening — constant in signing,
 *     because the hands come toward the camera — the upper arm stretched and the
 *     figure read as elastic.
 *   * THE ELBOW IS THE WORST-TRACKED JOINT. It is frequently occluded by the
 *     torso or the other arm, and its depth is close to a guess.
 *   * NO ORIENTATION. A tracked elbow gives a position, and a position cannot
 *     say how the upper arm is ROLLED. Humeral rotation is what puts the elbow
 *     out to the side rather than under the shoulder, so without it the whole
 *     arm reads as pinned to the body.
 *
 * So the wrist is treated as the goal and the elbow is SOLVED. Bone lengths come
 * from the model and never change; the elbow lands on the circle those two
 * lengths allow; and which point on that circle is chosen by a pole vector — the
 * tracked elbow when it is trustworthy, an anatomical default when it is not.
 * The arm is then exactly as long as the avatar's arm, always.
 *
 * ---------------------------------------------------------------------------
 * WHY THE WRIST DRIVES THE ARM, NOT THE OTHER WAY ROUND
 * ---------------------------------------------------------------------------
 * In sign language the hand's LOCATION is a phonological parameter: HOME and
 * SCHOOL can share a handshape and differ by where they are made. So the thing
 * that must be right is where the hand ends up, and everything else in the arm
 * exists to deliver it there. Forward kinematics from the shoulder puts the
 * accumulated error in the hand, which is precisely the wrong place for it.
 */

import * as THREE from "three";
import type { ArmChain, BoneName, HumanoidPose, Side } from "../../core/humanoid";
import { ARMS, JOINT_LIMITS } from "../../core/humanoid";
import {
  basisRotation,
  clamp,
  clampInFrame,
  rotationBetween,
  smoothstep,
} from "../../core/math";
import type { BoneRest, RestPose } from "./rest";

/** Everything the arm solve needs about one side, per frame. */
export interface ArmTarget {
  /** Wrist position in model space, metres. */
  wrist: THREE.Vector3;
  /** Desired world rotation of the hand bone, from the hand solve. */
  handRotation: THREE.Quaternion;
  /** Tracked elbow, when the pose model actually saw it. */
  elbow: THREE.Vector3 | null;
  /** 0-1. Below ~0.3 the elbow hint is ignored in favour of anatomy. */
  elbowConfidence: number;
  /** Whether there is a usable hand rotation at all. */
  hasHand: boolean;
}

export function makeArmTarget(): ArmTarget {
  return {
    wrist: new THREE.Vector3(),
    handRotation: new THREE.Quaternion(),
    elbow: null,
    elbowConfidence: 0,
    hasHand: false,
  };
}

/** Where each shoulder sits in model space, and the frame the chest is in. */
export interface BodyFrame {
  shoulder: [THREE.Vector3, THREE.Vector3];
  /** Chest orientation as a world rotation. */
  chestRotation: THREE.Quaternion;
  /** Unit vector toward the figure's own LEFT, for choosing elbow poles. */
  chestSide: THREE.Vector3;
}

export function makeBodyFrame(): BodyFrame {
  return {
    shoulder: [new THREE.Vector3(), new THREE.Vector3()],
    chestRotation: new THREE.Quaternion(),
    chestSide: new THREE.Vector3(1, 0, 0),
  };
}

/**
 * Solve one arm and write `shoulder`, `upperArm`, `lowerArm` and `hand` into the
 * pose.
 *
 * Returns the world rotation the hand bone ended up with, AFTER limits — the
 * finger solve must be given this rather than the raw measurement, or the hand
 * detaches from the wrist the viewer can see.
 */
export function solveArm(
  arm: ArmChain,
  target: ArmTarget,
  body: BodyFrame,
  rest: RestPose,
  pose: HumanoidPose,
  out: THREE.Quaternion,
): THREE.Quaternion {
  const upperRest = rest.bones.get(arm.upper);
  const lowerRest = rest.bones.get(arm.lower);
  const handRest = rest.bones.get(arm.hand);
  if (!upperRest || !lowerRest || !handRest) return out.identity();

  const index = arm.side === "left" ? 0 : 1;
  const upperLength = rest.armLengths[index].upper;
  const lowerLength = rest.armLengths[index].lower;
  const shoulder = body.shoulder[index];

  /* --------------------------------------------------------- reachability -- */
  A.toWrist.copy(target.wrist).sub(shoulder);
  const reach = A.toWrist.length();
  const maxReach = (upperLength + lowerLength) * 0.999;
  const minReach = Math.abs(upperLength - lowerLength) * 1.001 + 1e-4;

  if (reach < 1e-5) {
    // Wrist on top of the shoulder: no direction to solve. Leave the arm alone
    // rather than producing an arbitrary one.
    return out.copy(pose.rotations.get(arm.hand) ?? IDENTITY);
  }
  A.toWrist.divideScalar(reach);

  // Clamping the GOAL rather than letting the solve fail is what stops the arm
  // snapping straight when the tracker puts the hand a centimetre beyond reach —
  // which it does constantly, because the avatar's arm is not the user's arm.
  const clamped = clamp(reach, minReach, maxReach);

  /* ------------------------------------------------------------ the elbow -- */
  // Cosine rule: distance along the shoulder→wrist line to the elbow's circle,
  // and that circle's radius.
  const along =
    (clamped * clamped + upperLength * upperLength - lowerLength * lowerLength) /
    (2 * clamped);
  const radius = Math.sqrt(Math.max(0, upperLength * upperLength - along * along));

  pickPole(arm, target, body, shoulder, A.toWrist, clamped, along, A.pole);

  A.elbow
    .copy(shoulder)
    .addScaledVector(A.toWrist, along)
    .addScaledVector(A.pole, radius);

  /* ------------------------------------------------------- the two bones -- */
  // The shoulder blade first: a small lift toward a high target. Without it the
  // arm reaches overhead by rotating in the socket alone, which is anatomically
  // impossible past about 120° and reads as a doll's arm.
  solveShoulderGirdle(arm, target, body, rest, pose);

  // Recompute the chest-relative parent rotation AFTER the girdle moved.
  A.parentWorld.copy(body.chestRotation);
  const shoulderRotation = pose.rotations.get(arm.shoulder);
  if (shoulderRotation) A.parentWorld.multiply(shoulderRotation);

  // Upper arm: point it at the elbow, and ROLL it so its own flexion axis lies
  // perpendicular to the arm's plane. That roll is humeral rotation, and it is
  // the difference between an elbow that sticks out to the side and one welded
  // to the ribs.
  A.upperDirection.copy(A.elbow).sub(shoulder).normalize();
  /* The elbow's hinge axis, and its SIGN is load-bearing.
   *
   * `applyDirected` builds the bone's frame so that canonical +Z — the direction
   * flexion carries the tip — comes out as `cross(flexAxis, direction)`. The
   * elbow bulges out along +pole, so the forearm bends back the OTHER way, and
   * the flex direction is −pole.
   *
   *     cross(pole, upper) → flex direction −pole   ✓
   *     cross(upper, pole) → flex direction +pole   ✗
   *
   * Taking the second gives an elbow whose every bend reads as hyperextension.
   * The limit then clamps it to −4°, and the arm renders permanently straight —
   * measured at 0.538 m against a 0.540 m maximum reach, with the hand landing a
   * whole shoulder width from where the sign put it. Nothing about that failure
   * looks like a sign error; it looks like the IK is ignoring its target.
   */
  A.planeNormal.crossVectors(A.pole, A.upperDirection);
  if (A.planeNormal.lengthSq() < 1e-8) A.planeNormal.copy(body.chestSide);
  A.planeNormal.normalize();

  applyDirected(
    arm.upper,
    upperRest,
    A.parentWorld,
    A.upperDirection,
    A.planeNormal,
    pose,
  );
  A.parentWorld.multiply(pose.rotations.get(arm.upper)!);

  // Forearm: a hinge, so the minimal rotation is exactly right. Its twist —
  // pronation and supination — is set afterwards from the hand, because that is
  // where the measurement of it lives.
  A.lowerDirection.copy(target.wrist).sub(A.elbow);
  if (A.lowerDirection.lengthSq() < 1e-10) A.lowerDirection.copy(A.upperDirection);
  A.lowerDirection.normalize();

  A.inverseParent.copy(A.parentWorld).invert();
  A.local.copy(A.lowerDirection).applyQuaternion(A.inverseParent);
  A.hint.set(0, 0, 1).applyQuaternion(lowerRest.frame);
  const lower = pose.rotations.get(arm.lower)!;
  rotationBetween(lowerRest.axis, A.local, A.hint, lower);

  if (target.hasHand) {
    // Forearm twist from the hand's measured roll.
    //
    // Supination is a FOREARM motion — the radius crossing the ulna — not a
    // wrist one. The previous build had no forearm twist at all and tried to
    // absorb palm orientation at the wrist, where the range is about ±12°; that
    // is why it could not turn a palm over, and palm orientation is one of the
    // five parameters that distinguish one sign from another.
    applyForearmTwist(arm, target, lowerRest, A.parentWorld, lower);
  }

  clampBone(arm.lower, lowerRest, lower);
  A.parentWorld.multiply(lower);

  /* -------------------------------------------------------------- the hand -- */
  const hand = pose.rotations.get(arm.hand)!;
  if (target.hasHand) {
    A.inverseParent.copy(A.parentWorld).invert();
    hand.copy(A.inverseParent).multiply(target.handRotation);
  } else {
    hand.identity();
  }
  clampBone(arm.hand, handRest, hand);

  return out.copy(A.parentWorld).multiply(hand);
}

/**
 * Choose which point on the elbow circle to use.
 *
 * The tracked elbow when it is believable, an anatomical default otherwise, and
 * a smooth blend between them — never a switch. A hard switch is visible as the
 * elbow flicking between two positions every time confidence crosses the
 * threshold, and confidence crosses thresholds constantly.
 */
function pickPole(
  arm: ArmChain,
  target: ArmTarget,
  body: BodyFrame,
  shoulder: THREE.Vector3,
  toWrist: THREE.Vector3,
  reach: number,
  along: number,
  out: THREE.Vector3,
): void {
  /* The anatomical default: down, outward, and slightly back. This is where a
     human elbow goes when the hand is out in front, and solving for it is what
     stops a lost elbow collapsing the arm into a straight line through the
     chest. */
  const side = body.chestSide;
  A.defaultPole.copy(side).multiplyScalar(arm.outward * 0.55);
  A.defaultPole.y -= 1;
  A.defaultPole.z -= 0.35;
  // Make it perpendicular to the shoulder→wrist axis; only that component
  // selects a point on the circle.
  A.defaultPole.addScaledVector(toWrist, -A.defaultPole.dot(toWrist));
  if (A.defaultPole.lengthSq() < 1e-8) A.defaultPole.set(0, -1, 0);
  A.defaultPole.normalize();

  out.copy(A.defaultPole);
  if (!target.elbow || target.elbowConfidence < 0.05) return;

  // The tracked elbow, reduced to the component that actually selects a point on
  // the circle: everything along the shoulder→wrist axis is irrelevant to it.
  A.trackedPole.copy(target.elbow).sub(shoulder);
  A.trackedPole.addScaledVector(toWrist, -A.trackedPole.dot(toWrist));
  if (A.trackedPole.lengthSq() < 1e-8) return;
  A.trackedPole.normalize();

  /* Weight the tracked hint down when it is least reliable.

     Near full extension the elbow circle collapses to a point and its position
     carries almost no information about the pole — but the ANGLE of the pole is
     then extremely sensitive to elbow noise, so a confident-looking tracked
     elbow produces a wildly swinging arm roll. `spread` measures how far from
     collapsed the circle is. */
  const spread = smoothstep((reach - Math.abs(along)) / (reach * 0.25 + 1e-6));
  const weight = clamp(target.elbowConfidence, 0, 1) * spread;
  out.lerp(A.trackedPole, weight);
  if (out.lengthSq() < 1e-8) out.copy(A.defaultPole);
  out.normalize();
  // Re-orthogonalize: lerping two perpendicular-ish vectors does not stay
  // perpendicular, and a pole with an along-axis component shortens the arm.
  out.addScaledVector(toWrist, -out.dot(toWrist));
  if (out.lengthSq() < 1e-8) out.copy(A.defaultPole);
  out.normalize();
}

/**
 * Scapular elevation: the shoulder rises as the hand goes high.
 *
 * Small — up to about 20° — and the single cheapest thing that stops a rigged
 * avatar reading as a mannequin. The glenohumeral joint alone tops out near 120°
 * of elevation; everything above that comes from the shoulder blade, and a rig
 * that skips it either cannot reach high signs or reaches them by tearing the
 * arm out of the socket.
 */
function solveShoulderGirdle(
  arm: ArmChain,
  target: ArmTarget,
  body: BodyFrame,
  rest: RestPose,
  pose: HumanoidPose,
): void {
  const shoulderRest = rest.bones.get(arm.shoulder);
  const rotation = pose.rotations.get(arm.shoulder);
  if (!shoulderRest || !rotation) return;

  const index = arm.side === "left" ? 0 : 1;
  const shoulder = body.shoulder[index];
  const height = target.wrist.y - shoulder.y;
  // Fade in over the last third of the arm's length: no lift for a hand at chest
  // height, full lift for one above the head.
  const lift = smoothstep(height / (rest.armLengths[index].upper * 1.2));
  const limit = JOINT_LIMITS[arm.shoulder];
  const maxLift = limit ? limit.spread[1] : 0.35;

  A.axis.set(0, 0, 1).applyQuaternion(shoulderRest.frame);
  rotation.setFromAxisAngle(A.axis, lift * maxLift);
  clampBone(arm.shoulder, shoulderRest, rotation);
}

/**
 * Set the forearm's axial twist so the hand can reach its measured roll.
 *
 * Solved rather than measured: there is no landmark on the radius. What IS known
 * is where the hand needs to end up, so the twist is whatever gets the wrist's
 * residual rotation inside the wrist's own small range — which is the correct
 * division of labour, since the forearm has ±85° and the wrist ±12°.
 */
function applyForearmTwist(
  arm: ArmChain,
  target: ArmTarget,
  lowerRest: BoneRest,
  parentWorld: THREE.Quaternion,
  lower: THREE.Quaternion,
): void {
  // The forearm's world rotation with no twist applied yet.
  A.noTwist.copy(parentWorld).multiply(lower);
  // What the hand would need locally, given that.
  A.residual.copy(A.noTwist).invert().multiply(target.handRotation);

  // Project that residual onto the forearm's own axis: the part of it that a
  // twist can absorb.
  A.axis.copy(lowerRest.axis);
  const dot =
    A.residual.x * A.axis.x + A.residual.y * A.axis.y + A.residual.z * A.axis.z;
  const angle = 2 * Math.atan2(dot, A.residual.w);
  const limit = JOINT_LIMITS[arm.lower];
  const twist = limit ? clamp(angle, limit.twist[0], limit.twist[1]) : angle;

  A.twist.setFromAxisAngle(A.axis, twist);
  lower.multiply(A.twist);
}

/**
 * Point a bone along `direction` and roll it so its canonical flexion axis lands
 * on `flexAxis`, then clamp.
 *
 * The same construction the finger proximals use: direction alone leaves the
 * bone free to spin about itself, and for a bone with children that freedom is
 * inherited by everything below it.
 */
function applyDirected(
  bone: BoneName,
  boneRest: BoneRest,
  parentWorld: THREE.Quaternion,
  direction: THREE.Vector3,
  flexAxis: THREE.Vector3,
  pose: HumanoidPose,
): void {
  const rotation = pose.rotations.get(bone);
  if (!rotation) return;

  A.inverseParent.copy(parentWorld).invert();
  A.local.copy(direction).applyQuaternion(A.inverseParent);
  A.localFlex.copy(flexAxis).applyQuaternion(A.inverseParent);
  A.localFlex.addScaledVector(A.local, -A.localFlex.dot(A.local));
  if (A.localFlex.lengthSq() < 1e-8) {
    A.hint.set(0, 0, 1).applyQuaternion(boneRest.frame);
    rotationBetween(boneRest.axis, A.local, A.hint, rotation);
  } else {
    A.localFlex.normalize();
    // basisRotation(forward, up) maps Y → up and X → up × forward, so
    // forward = flex × direction puts the canonical flexion axis on `flexAxis`.
    A.forward.crossVectors(A.localFlex, A.local);
    basisRotation(A.forward, A.local, A.measured);
    A.inverseRest.copy(boneRest.frame).invert();
    rotation.copy(A.measured).multiply(A.inverseRest).normalize();
  }
  clampBone(bone, boneRest, rotation);
}

function clampBone(
  bone: BoneName,
  boneRest: BoneRest,
  rotation: THREE.Quaternion,
): void {
  const limit = JOINT_LIMITS[bone];
  if (!limit) return;
  clampInFrame(rotation, boneRest.frame, limit.flex, limit.spread, limit.twist);
}

/* ------------------------------------------------------------------ misc -- */

const IDENTITY = new THREE.Quaternion();

/** Both arms, in one call. */
export function solveArms(
  targets: [ArmTarget, ArmTarget],
  body: BodyFrame,
  rest: RestPose,
  pose: HumanoidPose,
  handWorld: [THREE.Quaternion, THREE.Quaternion],
): void {
  for (let i = 0; i < ARMS.length; i += 1) {
    solveArm(ARMS[i], targets[i], body, rest, pose, handWorld[i]);
  }
}

export function sideIndex(side: Side): number {
  return side === "left" ? 0 : 1;
}

const A = {
  toWrist: new THREE.Vector3(),
  elbow: new THREE.Vector3(),
  pole: new THREE.Vector3(),
  defaultPole: new THREE.Vector3(),
  trackedPole: new THREE.Vector3(),
  upperDirection: new THREE.Vector3(),
  lowerDirection: new THREE.Vector3(),
  planeNormal: new THREE.Vector3(),
  local: new THREE.Vector3(),
  localFlex: new THREE.Vector3(),
  forward: new THREE.Vector3(),
  hint: new THREE.Vector3(),
  axis: new THREE.Vector3(),
  parentWorld: new THREE.Quaternion(),
  inverseParent: new THREE.Quaternion(),
  inverseRest: new THREE.Quaternion(),
  measured: new THREE.Quaternion(),
  noTwist: new THREE.Quaternion(),
  residual: new THREE.Quaternion(),
  twist: new THREE.Quaternion(),
};

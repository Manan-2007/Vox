/**
 * Measuring a skeleton, so the retargeter never has to assume one.
 *
 * ---------------------------------------------------------------------------
 * THE PROBLEM THIS SOLVES
 * ---------------------------------------------------------------------------
 * "Bend the index finger 40°" is not a well-formed instruction until you know
 * which way the finger points and which way its palm faces. Those differ between
 * models: one artist builds a left forearm running down +X, another down −Z, and
 * a limit written against the first silently applies flexion to the second's
 * twist axis. The arm then locks instead of bending, and there is nothing in the
 * output that says why.
 *
 * The fix is to stop assuming and start measuring. At load, this module walks
 * the model's own normalized bone hierarchy and derives, for every driven bone, a
 * CANONICAL FRAME:
 *
 *     +Y   along the bone, toward its child
 *     +Z   the direction the bone's tip travels when the joint FLEXES
 *     +X   Y × Z, the flexion axis itself
 *
 * Every anatomical limit in core/humanoid.ts is written in that frame, so the
 * same clinical ranges are correct for every model. Swapping the .vrm changes
 * nothing except these measurements.
 *
 * ---------------------------------------------------------------------------
 * WHY THE NORMALIZED RIG
 * ---------------------------------------------------------------------------
 * three-vrm exposes two hierarchies. The RAW one is the artist's, with whatever
 * bind rotations they used. The NORMALIZED one has identity rest rotation on
 * every bone, which gives two properties this file depends on:
 *
 *   1. A bone's local space at rest IS world space at rest, so a direction
 *      measured in world coordinates can be used directly as a local axis.
 *   2. World rotation composes as `parent ⋅ local`, with no bind-pose term.
 *
 * Both stop being true on the raw rig, and re-deriving them per model is exactly
 * the work this design exists to avoid.
 */

import * as THREE from "three";
import type { BoneName, Side } from "../../core/humanoid";
import {
  ARMS,
  DRIVEN_BONES,
  HAND,
  fingerChains,
} from "../../core/humanoid";
import { basisRotation } from "../../core/math";

/** Where a bone points and how its joint is allowed to move. */
export interface BoneRest {
  name: BoneName;
  /** Unit direction toward the child, in local (== rest world) space. */
  axis: THREE.Vector3;
  /**
   * Rotation taking the canonical axes onto this bone's local axes. Passed
   * straight to `clampInFrame`.
   */
  frame: THREE.Quaternion;
  /** Bone length in metres. Zero for terminal bones. */
  length: number;
  /** Nearest driven ancestor, for composing world rotations. */
  parent: BoneName | null;
  /**
   * Rest position relative to that ancestor, in metres.
   *
   * Needed so the retargeter can run a short forward-kinematic chain of its own
   * down to the shoulders. It has to know where the arms will ACTUALLY start
   * once the spine has turned, not where they start at rest — see the anchoring
   * note in `retarget/index.ts`.
   */
  offset: THREE.Vector3;
  /** True when the axis was inherited rather than measured (terminal bones). */
  inferred: boolean;
}

export interface RestPose {
  bones: Map<BoneName, BoneRest>;
  /** Shoulder-to-shoulder distance in metres — the model's own unit of scale. */
  shoulderWidth: number;
  /** Wrist-to-middle-knuckle, per side, in metres. */
  handLength: [number, number];
  /** Upper-arm and forearm lengths per side, for IK reach. */
  armLengths: [{ upper: number; lower: number }, { upper: number; lower: number }];
  /** Hip position at rest, the root the whole figure hangs from. */
  hipsPosition: THREE.Vector3;
  /**
   * Midpoint of the two shoulder joints at rest.
   *
   * This is the ANCHOR the whole retarget hangs off: measured signing space is
   * placed here, so the avatar stands still while the signer moves around the
   * camera frame. Where a sign happens relative to the body is meaning; where the
   * body happens to be in frame is not.
   */
  shoulderMid: THREE.Vector3;
  /** Shoulder line to hip joints, metres — the torso's own length. */
  torsoLength: number;
  /** Chin to crown, metres. Head aim is expressed as a fraction of this. */
  headHeight: number;
  /** Bones the model was missing. Empty on a well-formed humanoid. */
  missing: BoneName[];
}

/** Supplies a bone's node, or null when the model does not have it. */
export type BoneLookup = (name: BoneName) => THREE.Object3D | null;

/**
 * The node whose space the rig's rotations compose in.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS IS NOT SIMPLY WORLD SPACE
 * ---------------------------------------------------------------------------
 * Everything here rests on one property of the normalized rig: a bone's LOCAL
 * space at rest is the same as the space its measured axis lives in, so a
 * direction read off the model can be used directly as a local axis.
 *
 * That holds relative to the top of the normalized hierarchy — NOT relative to
 * the world. `VRMUtils.rotateVRM0` turns a VRM 0.x model 180° about Y so it
 * faces +Z like a 1.0 model, and every normalized bone sits under that rotation.
 * Measuring in world space then bakes the 180° into every axis, while the local
 * rotations the rig applies do not have it — so the two disagree by exactly a
 * half turn.
 *
 * The symptom is spectacular and does not look like a coordinate bug: the idle
 * pose put the hands ABOVE the head and swapped left for right. It cost a real
 * .vrm to find, because a procedurally built skeleton has an unrotated root and
 * the two spaces coincide.
 */
export type ReferenceRoot = THREE.Object3D | null;

/**
 * The model's facing direction IN WORLD SPACE. VRM 1.0 requires +Z, and
 * `VRMUtils.rotateVRM0` brings 0.x models into line, so this much can be assumed.
 *
 * It cannot be used directly, though. Measurements are taken in the reference
 * root's space (see ReferenceRoot), and for a 0.x model that root carries the
 * 180° turn — so inside it the model still faces −Z. Assuming +Z there inverts
 * the flexion direction of every joint in the body, which is how an idle pose
 * asking for 72° of arm-lowering produced 72° of arm-RAISING.
 *
 * So the facing is transformed into the measurement space rather than assumed,
 * which costs one quaternion multiply at load and works for both versions.
 */
const WORLD_ANTERIOR = new THREE.Vector3(0, 0, 1);
const UP = new THREE.Vector3(0, 1, 0);

/**
 * Which bone each driven bone hangs from, for composing world rotations.
 *
 * Deliberately NOT read from the scene graph. A .vrm may put non-humanoid nodes
 * between two humanoid bones — a twist helper, a corrective joint — and walking
 * `.parent` would then compose through transforms this rig does not control.
 * The humanoid spec defines the logical hierarchy, so that is what is used.
 */
const PARENT: Partial<Record<BoneName, BoneName>> = {
  spine: "hips",
  chest: "spine",
  upperChest: "chest",
  neck: "upperChest",
  head: "neck",
  leftShoulder: "upperChest",
  leftUpperArm: "leftShoulder",
  leftLowerArm: "leftUpperArm",
  leftHand: "leftLowerArm",
  rightShoulder: "upperChest",
  rightUpperArm: "rightShoulder",
  rightLowerArm: "rightUpperArm",
  rightHand: "rightLowerArm",
};

for (const side of ["left", "right"] as const) {
  const hand: BoneName = side === "left" ? "leftHand" : "rightHand";
  for (const chain of fingerChains(side)) {
    PARENT[chain.bones[0]] = hand;
    PARENT[chain.bones[1]] = chain.bones[0];
    PARENT[chain.bones[2]] = chain.bones[1];
  }
}

/**
 * Fall back through the hierarchy for models that omit optional bones.
 *
 * `upperChest` and `chest` are both optional in the VRM spec, and `shoulder`
 * often is too. Without this a model missing `upperChest` would have both arms
 * parented to a bone that does not exist, and the arms would be posed in world
 * space while the torso turned underneath them.
 */
function resolveParent(
  name: BoneName,
  present: (bone: BoneName) => boolean,
): BoneName | null {
  let parent = PARENT[name] ?? null;
  while (parent && !present(parent)) parent = PARENT[parent] ?? null;
  return parent;
}

/* ------------------------------------------------------------- measuring -- */

/**
 * Capture the rest geometry of a humanoid.
 *
 * `lookup` must return NORMALIZED bone nodes with the model in its rest pose —
 * i.e. before any rotation this rig applies. Call once, at load.
 */
export function captureRest(
  lookup: BoneLookup,
  reference: ReferenceRoot = null,
): RestPose {
  const present = (bone: BoneName) => lookup(bone) !== null;
  const world = new Map<BoneName, THREE.Vector3>();
  const missing: BoneName[] = [];

  /* Everything is measured in the REFERENCE node's space — see ReferenceRoot.
     With no reference supplied this is world space, which is correct for a rig
     whose root carries no rotation. */
  const intoReference = new THREE.Matrix4();
  const anterior = WORLD_ANTERIOR.clone();
  if (reference) {
    reference.updateWorldMatrix(true, false);
    intoReference.copy(reference.matrixWorld).invert();
    // The facing direction, expressed in the space everything else is measured
    // in. Rotation only — a translation cannot change which way a body faces.
    const spin = reference.getWorldQuaternion(new THREE.Quaternion()).invert();
    anterior.applyQuaternion(spin).normalize();
  }

  for (const bone of DRIVEN_BONES) {
    const node = lookup(bone);
    if (!node) {
      missing.push(bone);
      continue;
    }
    const position = node.getWorldPosition(new THREE.Vector3());
    if (reference) position.applyMatrix4(intoReference);
    world.set(bone, position);
  }

  /* Palm normals, measured per hand from the knuckle line and the finger
     direction. Everything about finger flexion is expressed against these, so
     they are computed first and shared. */
  const palmar: Record<Side, THREE.Vector3> = {
    left: measurePalmar("left", world, lookup),
    right: measurePalmar("right", world, lookup),
  };

  const bones = new Map<BoneName, BoneRest>();

  for (const bone of DRIVEN_BONES) {
    const here = world.get(bone);
    if (!here) continue;

    const childName = childOf(bone, present);
    const childPos = childName
      ? world.get(childName)
      : childPosition(bone, lookup, reference ? intoReference : null);

    const axis = new THREE.Vector3();
    let length = 0;
    let inferred = false;

    if (childPos && childPos.distanceToSquared(here) > 1e-10) {
      axis.copy(childPos).sub(here);
      length = axis.length();
      axis.normalize();
    } else {
      // Terminal bone with no usable child node — continue straight on from the
      // parent. A distal phalanx that inherits its proximal's direction is
      // exactly right: at rest a finger is straight.
      const parentName = resolveParent(bone, present);
      const parentRest = parentName ? bones.get(parentName) : null;
      axis.copy(parentRest?.axis ?? UP);
      inferred = true;
    }

    const parentName = resolveParent(bone, present);
    const parentPos = parentName ? world.get(parentName) : null;
    bones.set(bone, {
      name: bone,
      axis,
      frame: canonicalFrame(bone, axis, palmar, anterior),
      length,
      parent: parentName,
      offset: parentPos ? here.clone().sub(parentPos) : here.clone(),
      inferred,
    });
  }

  /* ------------------------------------------------------ scale measures -- */

  const shoulderL = world.get("leftUpperArm");
  const shoulderR = world.get("rightUpperArm");
  const shoulderWidth =
    shoulderL && shoulderR ? shoulderL.distanceTo(shoulderR) : 0.4;

  const handLength: [number, number] = [
    measureHandLength("left", world),
    measureHandLength("right", world),
  ];

  const armLengths = ARMS.map((arm) => ({
    upper: bones.get(arm.upper)?.length ?? 0,
    lower: bones.get(arm.lower)?.length ?? 0,
  })) as RestPose["armLengths"];

  const shoulderMid = new THREE.Vector3();
  if (shoulderL && shoulderR) {
    shoulderMid.copy(shoulderL).add(shoulderR).multiplyScalar(0.5);
  } else {
    shoulderMid.copy(world.get("upperChest") ?? world.get("chest") ?? new THREE.Vector3());
  }

  const hips = world.get("hips");
  const torsoLength = hips ? Math.max(0.1, shoulderMid.y - hips.y) : 0.46;

  /* Head height, chin to crown. Taken from the neck-to-head distance where the
     model provides it, because a head is about 2.7 times the length of that
     segment on a normally proportioned figure. Head aim is expressed as a
     fraction of this, so it only has to be right to within a few percent — but
     it does have to scale with the model, or a stylised head with a large skull
     would under-rotate on every sign. */
  const neck = world.get("neck");
  const head = world.get("head");
  const headHeight =
    neck && head ? Math.max(0.12, head.distanceTo(neck) * 2.7) : 0.23;

  return {
    bones,
    shoulderWidth,
    handLength,
    armLengths,
    hipsPosition: hips?.clone() ?? new THREE.Vector3(),
    shoulderMid,
    torsoLength,
    headHeight,
    missing,
  };
}

/**
 * The logical child of a bone, for measuring its direction.
 *
 * Arms point at the next arm bone; a hand points at its MIDDLE finger, because
 * the middle metacarpal is the hand's own long axis and the one a wrist rotates
 * about. Using the index instead — the obvious alternative — tilts the whole
 * hand frame toward the thumb by about eight degrees, and every finger rotation
 * then inherits that error.
 */
function childOf(
  bone: BoneName,
  present: (name: BoneName) => boolean,
): BoneName | null {
  const direct: Partial<Record<BoneName, BoneName[]>> = {
    hips: ["spine"],
    spine: ["chest", "upperChest", "neck"],
    chest: ["upperChest", "neck"],
    upperChest: ["neck"],
    neck: ["head"],
    leftShoulder: ["leftUpperArm"],
    leftUpperArm: ["leftLowerArm"],
    leftLowerArm: ["leftHand"],
    leftHand: ["leftMiddleProximal", "leftIndexProximal"],
    rightShoulder: ["rightUpperArm"],
    rightUpperArm: ["rightLowerArm"],
    rightLowerArm: ["rightHand"],
    rightHand: ["rightMiddleProximal", "rightIndexProximal"],
  };

  for (const side of ["left", "right"] as const) {
    for (const chain of fingerChains(side)) {
      direct[chain.bones[0]] = [chain.bones[1]];
      direct[chain.bones[1]] = [chain.bones[2]];
    }
  }

  for (const candidate of direct[bone] ?? []) {
    if (present(candidate)) return candidate;
  }
  return null;
}

/**
 * The world position of a terminal bone's tip, from the model's own scene graph.
 *
 * Most rigs carry an "end" node past the last phalanx. Where one exists it gives
 * the distal segment a real length and a real direction; where it does not, the
 * caller falls back to inheriting the parent's axis.
 */
function childPosition(
  bone: BoneName,
  lookup: BoneLookup,
  intoReference: THREE.Matrix4 | null = null,
): THREE.Vector3 | null {
  const node = lookup(bone);
  if (!node || node.children.length === 0) return null;
  // Take the farthest child: some rigs hang IK targets or twist helpers off a
  // phalanx, and those sit almost on top of the joint.
  let best: THREE.Object3D | null = null;
  let bestDistance = 1e-6;
  const here = node.getWorldPosition(new THREE.Vector3());
  const probe = new THREE.Vector3();
  for (const child of node.children) {
    const distance = child.getWorldPosition(probe).distanceTo(here);
    if (distance > bestDistance) {
      bestDistance = distance;
      best = child;
    }
  }
  if (!best) return null;
  const position = best.getWorldPosition(new THREE.Vector3());
  if (intoReference) position.applyMatrix4(intoReference);
  return position;
}

/**
 * The palmar direction of one hand: which way the palm faces at rest.
 *
 * Derived from the plane of the knuckles. `knuckle` runs index → little, and the
 * cross product with the finger direction gives the dorsal (back-of-hand)
 * normal — with the handedness flip that a mirrored limb requires. Palmar is its
 * negation, and it is the direction a finger tip travels when the finger curls.
 */
function measurePalmar(
  side: Side,
  world: Map<BoneName, THREE.Vector3>,
  lookup: BoneLookup,
): THREE.Vector3 {
  const chains = fingerChains(side);
  const index = world.get(chains[1].bones[0]);
  const middle = world.get(chains[2].bones[0]);
  const little = world.get(chains[4].bones[0]);
  const hand = world.get(side === "left" ? "leftHand" : "rightHand");

  // Default matches the VRM 1.0 T-pose convention: palms face down.
  const fallback = new THREE.Vector3(0, -1, 0);
  if (!index || !little || !hand || !middle) {
    void lookup;
    return fallback;
  }

  const fingerDirection = middle.clone().sub(hand);
  if (fingerDirection.lengthSq() < 1e-10) return fallback;
  fingerDirection.normalize();

  const knuckle = little.clone().sub(index);
  // Remove any component along the fingers, so the two axes are genuinely the
  // plane of the palm rather than a skewed pair.
  knuckle.addScaledVector(fingerDirection, -knuckle.dot(fingerDirection));
  if (knuckle.lengthSq() < 1e-10) return fallback;
  knuckle.normalize();

  const dorsal =
    side === "left"
      ? new THREE.Vector3().crossVectors(fingerDirection, knuckle)
      : new THREE.Vector3().crossVectors(knuckle, fingerDirection);
  if (dorsal.lengthSq() < 1e-10) return fallback;
  return dorsal.normalize().negate();
}

function measureHandLength(
  side: Side,
  world: Map<BoneName, THREE.Vector3>,
): number {
  const hand = world.get(side === "left" ? "leftHand" : "rightHand");
  const middle = world.get(fingerChains(side)[2].bones[0]);
  if (!hand || !middle) return 0.09;
  const length = hand.distanceTo(middle);
  return length > 1e-4 ? length : 0.09;
}

/* ---------------------------------------------------------------- frames -- */

/**
 * The canonical frame for one bone: +Y along it, +Z the direction its tip moves
 * when the joint flexes, +X = Y × Z.
 *
 * `basisRotation` is given (forward = Z, up = Y) and orthonormalizes, so a flex
 * direction that is not exactly perpendicular to the bone — which it never is on
 * a real model — still produces a clean frame rather than a sheared one.
 */
function canonicalFrame(
  bone: BoneName,
  axis: THREE.Vector3,
  palmar: Record<Side, THREE.Vector3>,
  anterior: THREE.Vector3,
): THREE.Quaternion {
  const flexDirection = flexDirectionFor(bone, palmar, anterior);
  // Project the flex direction perpendicular to the bone. Without this, a bone
  // that happens to lie near the flex direction produces a degenerate frame and
  // the joint's limits become meaningless exactly where it is most bent.
  const forward = flexDirection.clone();
  forward.addScaledVector(axis, -forward.dot(axis));
  if (forward.lengthSq() < 1e-8) {
    // Bone is parallel to its own flex direction — impossible anatomically, so
    // this is a malformed rig. Pick any perpendicular deterministically.
    forward.set(axis.y, -axis.x, 0);
    if (forward.lengthSq() < 1e-8) forward.set(0, axis.z, -axis.y);
  }
  forward.normalize();
  return basisRotation(forward, axis, new THREE.Quaternion());
}

/**
 * Which way a joint's tip travels when it flexes.
 *
 *   fingers, thumb, wrist — toward the palm
 *   everything else       — anterior, the way the model faces
 *
 * "Anterior" covers the elbow (forearm swings forward), the shoulder (arm
 * raises forward), and the whole spine and neck (nodding forward). That those
 * all share one answer is not a simplification: flexion is *defined* as motion
 * in the sagittal plane toward the front of the body.
 */
function flexDirectionFor(
  bone: BoneName,
  palmar: Record<Side, THREE.Vector3>,
  anterior: THREE.Vector3,
): THREE.Vector3 {
  const isLeft = bone.startsWith("left");
  const isRight = bone.startsWith("right");
  if (!isLeft && !isRight) return anterior.clone();

  const side: Side = isLeft ? "left" : "right";
  const stem = bone.slice(side.length);
  const isHandOrFinger =
    stem === "Hand" ||
    stem.startsWith("Thumb") ||
    stem.startsWith("Index") ||
    stem.startsWith("Middle") ||
    stem.startsWith("Ring") ||
    stem.startsWith("Little");

  return isHandOrFinger ? palmar[side].clone() : anterior.clone();
}

/* ---------------------------------------------------------------- helpers -- */

/** The 21-landmark index of a bone's own joint, for hands. Debug use. */
export const LANDMARK_OF_BONE: Partial<Record<BoneName, number>> = (() => {
  const map: Partial<Record<BoneName, number>> = {};
  for (const side of ["left", "right"] as const) {
    map[side === "left" ? "leftHand" : "rightHand"] = HAND.WRIST;
    for (const chain of fingerChains(side)) {
      map[chain.bones[0]] = chain.landmarks[0];
      map[chain.bones[1]] = chain.landmarks[1];
      map[chain.bones[2]] = chain.landmarks[2];
    }
  }
  return map;
})();

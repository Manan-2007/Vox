/**
 * A reference humanoid skeleton, built from anthropometric measurements.
 *
 * ---------------------------------------------------------------------------
 * WHAT THIS IS FOR — AND WHAT IT IS NOT
 * ---------------------------------------------------------------------------
 * This is NOT the avatar. It carries no mesh and is never rendered as a figure.
 * It is a skeleton in the VRM humanoid's own conventions, and it exists for
 * three jobs:
 *
 *   1. HEADLESS VERIFICATION. The retargeter can be run over the real motion
 *      library in node, with no browser and no .vrm, and its output checked
 *      against what the handshape is known to be. That is the only way to assert
 *      "the letter B renders as a flat hand" in an automated test, and it is how
 *      every constant in the solve was validated.
 *
 *   2. THE DEBUG SKELETON VIEW. Bone axes and joint gizmos, drawn before any
 *      model is loaded, so the rig can be inspected independently of whatever
 *      mesh is hanging off it. Required by the developer-mode brief and useful
 *      exactly when a model is misbehaving.
 *
 *   3. A LOADING SKELETON. Something coherent to hold the pose while a .vrm is
 *      still downloading, so the first frame after load is a blend rather than a
 *      pop.
 *
 * ---------------------------------------------------------------------------
 * WHY BUILD REAL BONES RATHER THAN A REST POSE DIRECTLY
 * ---------------------------------------------------------------------------
 * It would be shorter to write the `RestPose` out by hand. It would also mean
 * two implementations of "measure a skeleton" — this one and the one that runs
 * on the real model — which would drift, and the tests would then be verifying
 * the wrong one. So this builds an actual `THREE.Bone` hierarchy and hands it to
 * the same `captureRest` the VRM path uses. If rest capture is wrong, the checks
 * fail; they cannot pass by agreeing with themselves.
 *
 * ---------------------------------------------------------------------------
 * THE MEASUREMENTS
 * ---------------------------------------------------------------------------
 * A 50th-percentile adult, in metres. Segment lengths are from standard
 * anthropometric tables (Pheasant, *Bodyspace*); hand proportions from Buchholz
 * & Armstrong's hand-model data. Stature 1.72 m, biacromial breadth 0.40 m —
 * the same 0.40 m the motion library's shoulder-width normalization assumes, so
 * a sign recorded there arrives here at the right scale.
 *
 * Pose is VRM 1.0's required T-pose: arms along ±X, palms down, model facing +Z.
 */

import * as THREE from "three";
import type { BoneName, Side } from "../../core/humanoid";
import { fingerChains } from "../../core/humanoid";
import { captureRest, type RestPose } from "./rest";

/** Biacromial breadth. Everything else is expressed against it downstream. */
export const CANONICAL_SHOULDER_WIDTH = 0.40;

/**
 * Bone positions, in metres, in the model's own space. Each entry is the
 * WORLD position of that joint in the T-pose; the builder converts to local
 * offsets when it parents them.
 */
type Vec3 = readonly [number, number, number];

const SPINE: Partial<Record<BoneName, Vec3>> = {
  hips: [0, 0.94, 0],
  spine: [0, 1.05, 0],
  chest: [0, 1.16, 0],
  upperChest: [0, 1.28, 0],
  neck: [0, 1.44, 0],
  head: [0, 1.53, 0],
};

/**
 * Arm chain for the figure's LEFT side; the right is mirrored in x.
 *
 * The shoulder joint sits inboard of the acromion, which is why `leftUpperArm`
 * is at x = 0.18 rather than at half the 0.40 biacromial breadth. An arm hinged
 * at the very corner of the silhouette reads as a doll's.
 */
const LEFT_ARM: Partial<Record<BoneName, Vec3>> = {
  leftShoulder: [0.04, 1.40, 0],
  leftUpperArm: [0.18, 1.40, 0],
  leftLowerArm: [0.47, 1.40, 0],
  leftHand: [0.72, 1.40, 0],
};

/**
 * Finger geometry for the left hand.
 *
 * `knuckle` is the world position of the MCP joint; `phalanges` are the three
 * segment lengths, proximal to distal. Fingers run along +x in the T-pose.
 *
 * The knuckle line is spread across z — index anterior, little posterior —
 * because with the palm down and the model facing +z, the thumb side of the hand
 * IS the anterior side. That relationship is what `rest.ts` reads to work out
 * which way the palm faces, so it has to be geometrically true here rather than
 * merely plausible.
 */
interface FingerGeometry {
  knuckle: Vec3;
  phalanges: readonly [number, number, number];
}

const LEFT_FINGER_GEOMETRY: Record<string, FingerGeometry> = {
  index: { knuckle: [0.801, 1.400, 0.019], phalanges: [0.045, 0.026, 0.021] },
  middle: { knuckle: [0.806, 1.400, 0.000], phalanges: [0.050, 0.031, 0.022] },
  ring: { knuckle: [0.800, 1.400, -0.018], phalanges: [0.046, 0.029, 0.021] },
  little: { knuckle: [0.788, 1.400, -0.035], phalanges: [0.036, 0.020, 0.019] },
};

/**
 * The thumb, which is not a short finger.
 *
 * Its metacarpal leaves the wrist at a large angle — forward (anterior, +z),
 * outward (+x) and slightly palmar (−y) — and that angle is most of what makes
 * opposition possible. Modelling the thumb parallel to the fingers, which is the
 * tempting simplification, makes every pinch and every closed fist wrong.
 */
const LEFT_THUMB: readonly Vec3[] = [
  [0.742, 1.394, 0.024], // metacarpal base (CMC)
  [0.786, 1.386, 0.049], // proximal base (MCP)
  [0.812, 1.381, 0.063], // distal base (IP)
  [0.834, 1.377, 0.075], // tip
];

/* ------------------------------------------------------------- building --- */

export interface CanonicalSkeleton {
  root: THREE.Bone;
  lookup: (name: BoneName) => THREE.Object3D | null;
  bones: Map<BoneName, THREE.Bone>;
}

function mirror(position: Vec3): Vec3 {
  return [-position[0], position[1], position[2]];
}

/**
 * Build the bone hierarchy. Every bone is created with an identity local
 * rotation, which is what makes this a NORMALIZED rig in three-vrm's sense and
 * therefore directly comparable with the normalized hierarchy of a real model.
 */
export function buildCanonicalSkeleton(): CanonicalSkeleton {
  const bones = new Map<BoneName, THREE.Bone>();
  const worldPositions = new Map<BoneName, THREE.Vector3>();

  const place = (name: BoneName, position: Vec3) => {
    const bone = new THREE.Bone();
    bone.name = name;
    bones.set(name, bone);
    worldPositions.set(name, new THREE.Vector3(...position));
  };

  for (const [name, position] of Object.entries(SPINE)) {
    place(name as BoneName, position as Vec3);
  }

  for (const side of ["left", "right"] as const) {
    for (const [name, position] of Object.entries(LEFT_ARM)) {
      const bone = (side === "left"
        ? name
        : name.replace(/^left/, "right")) as BoneName;
      place(bone, side === "left" ? (position as Vec3) : mirror(position as Vec3));
    }

    for (const chain of fingerChains(side)) {
      if (chain.finger === "thumb") {
        LEFT_THUMB.forEach((position, i) => {
          if (i === 3) return; // the tip is a leaf node, added below
          place(chain.bones[i], side === "left" ? position : mirror(position));
        });
        continue;
      }
      const geometry = LEFT_FINGER_GEOMETRY[chain.finger];
      let x = geometry.knuckle[0];
      for (let i = 0; i < 3; i += 1) {
        const position: Vec3 = [x, geometry.knuckle[1], geometry.knuckle[2]];
        place(chain.bones[i], side === "left" ? position : mirror(position));
        x += geometry.phalanges[i];
      }
    }
  }

  /* Parent everything, converting world positions to local offsets. Because
     every local rotation is identity, a local offset is just the difference of
     the two world positions — which is the property the whole normalized-rig
     approach rests on. */
  const parentOf = canonicalParents();
  for (const [name, bone] of bones) {
    const here = worldPositions.get(name)!;
    const parentName = parentOf(name);
    const parent = parentName ? bones.get(parentName) : null;
    if (!parent || !parentName) {
      bone.position.copy(here);
      continue;
    }
    bone.position.copy(here).sub(worldPositions.get(parentName)!);
    parent.add(bone);
  }

  /* Leaf tips, so the distal phalanges have a measurable direction and length
     rather than inheriting their parent's. A rig without these makes every
     fingertip segment "inferred", and the last joint of every finger then
     cannot be limited correctly. */
  for (const side of ["left", "right"] as const) {
    for (const chain of fingerChains(side)) {
      const distal = bones.get(chain.bones[2]);
      if (!distal) continue;
      const tip = new THREE.Bone();
      tip.name = `${chain.bones[2]}_end`;
      if (chain.finger === "thumb") {
        const from = side === "left" ? LEFT_THUMB[2] : mirror(LEFT_THUMB[2]);
        const to = side === "left" ? LEFT_THUMB[3] : mirror(LEFT_THUMB[3]);
        tip.position.set(to[0] - from[0], to[1] - from[1], to[2] - from[2]);
      } else {
        const geometry = LEFT_FINGER_GEOMETRY[chain.finger];
        tip.position.set(
          (side === "left" ? 1 : -1) * geometry.phalanges[2],
          0,
          0,
        );
      }
      distal.add(tip);
    }
  }

  // A head tip, for the same reason.
  const head = bones.get("head");
  if (head) {
    const crown = new THREE.Bone();
    crown.name = "head_end";
    crown.position.set(0, 0.15, 0);
    head.add(crown);
  }

  const root = bones.get("hips")!;
  root.updateMatrixWorld(true);

  return {
    root,
    bones,
    lookup: (name: BoneName) => bones.get(name) ?? null,
  };
}

/**
 * The canonical hierarchy, as a function so it stays in one place.
 *
 * Mirrors the humanoid spec exactly. `rest.ts` has its own copy for resolving
 * parents on models with optional bones missing; this one is complete because
 * this skeleton always has every bone.
 */
function canonicalParents(): (name: BoneName) => BoneName | null {
  const parent: Partial<Record<BoneName, BoneName>> = {
    spine: "hips",
    chest: "spine",
    upperChest: "chest",
    neck: "upperChest",
    head: "neck",
  };
  for (const side of ["left", "right"] as const) {
    const S = side === "left" ? "left" : "right";
    parent[`${S}Shoulder` as BoneName] = "upperChest";
    parent[`${S}UpperArm` as BoneName] = `${S}Shoulder` as BoneName;
    parent[`${S}LowerArm` as BoneName] = `${S}UpperArm` as BoneName;
    parent[`${S}Hand` as BoneName] = `${S}LowerArm` as BoneName;
    for (const chain of fingerChains(side)) {
      parent[chain.bones[0]] = `${S}Hand` as BoneName;
      parent[chain.bones[1]] = chain.bones[0];
      parent[chain.bones[2]] = chain.bones[1];
    }
  }
  return (name) => parent[name] ?? null;
}

/** The reference rest pose, measured from the reference skeleton. */
export function canonicalRest(): { skeleton: CanonicalSkeleton; rest: RestPose } {
  const skeleton = buildCanonicalSkeleton();
  return { skeleton, rest: captureRest(skeleton.lookup) };
}

/**
 * Wear a pose and recompute world transforms.
 *
 * The point of doing this on real bones rather than by multiplying quaternions
 * by hand is that it is the SAME operation three.js performs on the .vrm — so a
 * headless check exercises the actual composition order, including any place the
 * hierarchy disagrees with what the retargeter assumed.
 */
export function applyPose(
  skeleton: CanonicalSkeleton,
  pose: { rotations: Map<BoneName, THREE.Quaternion> },
): void {
  for (const [name, bone] of skeleton.bones) {
    const rotation = pose.rotations.get(name);
    if (rotation) bone.quaternion.copy(rotation);
  }
  skeleton.root.updateMatrixWorld(true);
}

/** World position of one bone after `applyPose`. */
export function bonePosition(
  skeleton: CanonicalSkeleton,
  name: BoneName,
  out: THREE.Vector3,
): THREE.Vector3 | null {
  const bone = skeleton.bones.get(name);
  if (!bone) return null;
  return bone.getWorldPosition(out);
}

/** Which side a bone belongs to, or null for the spine. */
export function sideOf(bone: BoneName): Side | null {
  if (bone.startsWith("left")) return "left";
  if (bone.startsWith("right")) return "right";
  return null;
}

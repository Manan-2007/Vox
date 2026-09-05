/**
 * The bone contract: what a signing avatar must have, and how far each joint
 * may bend.
 *
 * ---------------------------------------------------------------------------
 * WHY VRM, AND WHY IT IS NOT A COSMETIC CHOICE
 * ---------------------------------------------------------------------------
 * The VRM humanoid specification defines fifteen finger bones per hand:
 *
 *     thumb   metacarpal / proximal / distal
 *     index   proximal / intermediate / distal
 *     middle  proximal / intermediate / distal
 *     ring    proximal / intermediate / distal
 *     little  proximal / intermediate / distal
 *
 * MediaPipe reports twenty-one hand landmarks, which describe exactly fifteen
 * phalanx segments plus the wrist and the five fingertips. The two agree
 * one-for-one:
 *
 *     landmarks 1→2→3→4      thumb   CMC / MCP / IP
 *     landmarks 5→6→7→8      index   MCP / PIP / DIP
 *     landmarks 9→10→11→12   middle  MCP / PIP / DIP
 *     landmarks 13→14→15→16  ring    MCP / PIP / DIP
 *     landmarks 17→18→19→20  little  MCP / PIP / DIP
 *
 * That correspondence is the entire argument. It means a tracked hand can be
 * turned into fifteen joint rotations with no fitting, no optimisation and no
 * approximation — and a rotation is the only thing that can be limited, blended
 * or retargeted onto a different body. Positions cannot be.
 *
 * ---------------------------------------------------------------------------
 * NORMALIZED BONES
 * ---------------------------------------------------------------------------
 * Everything here is expressed against three-vrm's NORMALIZED humanoid rig, not
 * the raw one. In the normalized rig every bone's rest rotation is the identity
 * and the hierarchy is a plain T-pose, which means a rotation computed for one
 * model is valid for every other model. Retargeting against the raw rig would
 * bake in one particular artist's bind pose, and swapping the .vrm would then
 * require re-deriving every constant in this file.
 *
 * ---------------------------------------------------------------------------
 * WHERE THE LIMITS COME FROM
 * ---------------------------------------------------------------------------
 * The ranges below are clinical goniometry norms (AAOS / Kapandji), converted to
 * radians at the bottom of the file. They are not style choices, and they are
 * not there to make the avatar look nice — they are there because MediaPipe
 * produces anatomically impossible hands several times a second under motion
 * blur, and an unlimited rig renders every one of them. A finger that bends
 * backwards through its own knuckle is the single most legibility-destroying
 * artefact a signing avatar can have, because the reader's first assumption is
 * that they misread the handshape.
 */

import type { VRMHumanBoneName } from "@pixiv/three-vrm";
import * as THREE from "three";
import { DEG } from "./math";

/* ------------------------------------------------------------- the bones -- */

export type BoneName = VRMHumanBoneName;

export type Side = "left" | "right";
export type FingerName = "thumb" | "index" | "middle" | "ring" | "little";

/**
 * Bones this rig drives. Legs are deliberately absent: signing is a seated or
 * standing-still activity, the reference library is filmed chest-up, and an
 * unmeasured leg driven by inference is a leg that twitches. They keep their
 * rest pose.
 */
export const DRIVEN_BONES: readonly BoneName[] = [
  "hips",
  "spine",
  "chest",
  "upperChest",
  "neck",
  "head",
  "leftShoulder",
  "leftUpperArm",
  "leftLowerArm",
  "leftHand",
  "rightShoulder",
  "rightUpperArm",
  "rightLowerArm",
  "rightHand",
  "leftThumbMetacarpal",
  "leftThumbProximal",
  "leftThumbDistal",
  "leftIndexProximal",
  "leftIndexIntermediate",
  "leftIndexDistal",
  "leftMiddleProximal",
  "leftMiddleIntermediate",
  "leftMiddleDistal",
  "leftRingProximal",
  "leftRingIntermediate",
  "leftRingDistal",
  "leftLittleProximal",
  "leftLittleIntermediate",
  "leftLittleDistal",
  "rightThumbMetacarpal",
  "rightThumbProximal",
  "rightThumbDistal",
  "rightIndexProximal",
  "rightIndexIntermediate",
  "rightIndexDistal",
  "rightMiddleProximal",
  "rightMiddleIntermediate",
  "rightMiddleDistal",
  "rightRingProximal",
  "rightRingIntermediate",
  "rightRingDistal",
  "rightLittleProximal",
  "rightLittleIntermediate",
  "rightLittleDistal",
] as const;

/**
 * Bones without which the avatar cannot sign at all. A .vrm missing any of
 * these is rejected at load with a message naming the bone, rather than being
 * loaded into a rig that silently does nothing with the fingers — which is the
 * failure mode that is hardest to diagnose from a screenshot.
 */
export const REQUIRED_BONES: readonly BoneName[] = [
  "hips",
  "spine",
  "head",
  "leftUpperArm",
  "leftLowerArm",
  "leftHand",
  "rightUpperArm",
  "rightLowerArm",
  "rightHand",
  "leftIndexProximal",
  "leftIndexIntermediate",
  "leftIndexDistal",
  "leftMiddleProximal",
  "leftThumbProximal",
  "rightIndexProximal",
  "rightIndexIntermediate",
  "rightIndexDistal",
  "rightMiddleProximal",
  "rightThumbProximal",
] as const;

/* ---------------------------------------------------------- finger chains -- */

/**
 * One finger: three bones, and the four MediaPipe landmarks whose three
 * segments they correspond to.
 *
 * `landmarks[i] → landmarks[i+1]` is the direction of `bones[i]`. That is the
 * whole retargeting rule for fingers, and it is exact — no fitting involved.
 */
export interface FingerChain {
  finger: FingerName;
  bones: readonly [BoneName, BoneName, BoneName];
  landmarks: readonly [number, number, number, number];
  /**
   * Which MCP landmark this finger's knuckle sits at, for measuring spread
   * against its neighbour.
   */
  knuckle: number;
}

const FINGER_LANDMARKS: Record<
  FingerName,
  readonly [number, number, number, number]
> = {
  thumb: [1, 2, 3, 4],
  index: [5, 6, 7, 8],
  middle: [9, 10, 11, 12],
  ring: [13, 14, 15, 16],
  little: [17, 18, 19, 20],
};

/** MediaPipe hand landmark indices, named. */
export const HAND = {
  WRIST: 0,
  THUMB_CMC: 1,
  THUMB_MCP: 2,
  THUMB_IP: 3,
  THUMB_TIP: 4,
  INDEX_MCP: 5,
  INDEX_PIP: 6,
  INDEX_DIP: 7,
  INDEX_TIP: 8,
  MIDDLE_MCP: 9,
  MIDDLE_PIP: 10,
  MIDDLE_DIP: 11,
  MIDDLE_TIP: 12,
  RING_MCP: 13,
  RING_PIP: 14,
  RING_DIP: 15,
  RING_TIP: 16,
  LITTLE_MCP: 17,
  LITTLE_PIP: 18,
  LITTLE_DIP: 19,
  LITTLE_TIP: 20,
  COUNT: 21,
} as const;

const FINGER_ORDER: readonly FingerName[] = [
  "thumb",
  "index",
  "middle",
  "ring",
  "little",
];

/** Capitalise for VRM's bone naming, which is `leftIndexProximal` etc. */
function boneOf(side: Side, finger: FingerName, segment: number): BoneName {
  const Side = side === "left" ? "left" : "right";
  const Finger = finger[0].toUpperCase() + finger.slice(1);
  // The thumb's three bones are named differently from the other four fingers:
  // metacarpal/proximal/distal rather than proximal/intermediate/distal. That is
  // not a quirk of the spec, it reflects anatomy — the thumb's first segment is
  // a metacarpal that moves, where the other fingers' metacarpals are fixed in
  // the palm.
  const segments =
    finger === "thumb"
      ? ["Metacarpal", "Proximal", "Distal"]
      : ["Proximal", "Intermediate", "Distal"];
  return `${Side}${Finger}${segments[segment]}` as BoneName;
}

export function fingerChains(side: Side): readonly FingerChain[] {
  return FINGER_ORDER.map((finger) => ({
    finger,
    bones: [boneOf(side, finger, 0), boneOf(side, finger, 1), boneOf(side, finger, 2)] as const,
    landmarks: FINGER_LANDMARKS[finger],
    knuckle: FINGER_LANDMARKS[finger][0],
  }));
}

export const LEFT_FINGERS = fingerChains("left");
export const RIGHT_FINGERS = fingerChains("right");

/* -------------------------------------------------------------- the arms -- */

export interface ArmChain {
  side: Side;
  shoulder: BoneName;
  upper: BoneName;
  lower: BoneName;
  hand: BoneName;
  /** +1 for the figure's left, -1 for its right — which way "outward" points. */
  outward: number;
}

export const ARMS: readonly ArmChain[] = [
  {
    side: "left",
    shoulder: "leftShoulder",
    upper: "leftUpperArm",
    lower: "leftLowerArm",
    hand: "leftHand",
    outward: 1,
  },
  {
    side: "right",
    shoulder: "rightShoulder",
    upper: "rightUpperArm",
    lower: "rightLowerArm",
    hand: "rightHand",
    outward: -1,
  },
];

/* ------------------------------------------------------------ the limits -- */

/**
 * An anatomical limit on one bone, expressed in that bone's CANONICAL frame:
 * +Y along the bone, +X the flexion axis, +Z the spread axis.
 *
 * Canonical rather than raw-local, because the raw local frame is whatever the
 * artist who built the .vrm chose. A left forearm might point down +X in one
 * model and down −Z in another, and limits written against one of those are
 * silently wrong for the other — the elbow would clamp its flexion range onto
 * the twist axis and the arm would lock. The frame that maps a given model into
 * this canonical one is measured from the model itself at load
 * (`avatar/retarget/rest.ts`), so every constant below is model-independent.
 *
 *   flex   — rotation about +X. Positive is the direction the joint naturally
 *            closes: elbow bending, finger curling, head nodding forward.
 *   spread — rotation about +Z. Abduction/adduction; lateral head tilt.
 *   twist  — rotation about +Y, the bone's own length. Forearm supination,
 *            humeral rotation, head shake.
 */
export interface JointLimit {
  flex: readonly [number, number];
  spread: readonly [number, number];
  twist: readonly [number, number];
}


/**
 * Ranges are clinical norms, converted from degrees.
 *
 * A note on the hinges. An elbow, a PIP and a DIP each have essentially ONE
 * degree of freedom, and every degree of freedom that is not clamped is a
 * degree of freedom noise will find. Giving the elbow ±4° of off-axis swing
 * rather than 0 is deliberate: exactly zero looks mechanical, because a real
 * elbow does carry a few degrees of play under load, and — more practically —
 * a hard zero makes the joint fight the IK solver every frame when the measured
 * hand position is a millimetre outside the arm's reach.
 */
function limit(
  flex: [number, number],
  spread: [number, number],
  twist: [number, number],
): JointLimit {
  return {
    flex: [flex[0] * DEG, flex[1] * DEG],
    spread: [spread[0] * DEG, spread[1] * DEG],
    twist: [twist[0] * DEG, twist[1] * DEG],
  };
}

/**
 * Mirror a limit for the other side.
 *
 * Flexion is NOT mirrored — an elbow closes the same way on both arms, and the
 * canonical frame is built per-side so that "positive flex" already means
 * "closing" on each. Spread and twist ARE mirrored, because abduction and
 * internal rotation are defined relative to the body's midline and therefore
 * swap sign across it.
 */
function mirrored(source: JointLimit): JointLimit {
  return {
    flex: source.flex,
    spread: [-source.spread[1], -source.spread[0]],
    twist: [-source.twist[1], -source.twist[0]],
  };
}

const LEFT_LIMITS: Partial<Record<BoneName, JointLimit>> = {
  // Scapular movement. Small, and almost entirely elevation and protraction —
  // but not negligible: a shoulder that never rises makes every high sign look
  // like it is being made by someone holding very still, which is the single
  // most common tell of a rigged avatar.
  leftShoulder: limit([-12, 12], [-8, 22], [-8, 8]),

  /* Glenohumeral.
   *
   * These are the one place clinical figures cannot be used directly, and
   * getting it wrong is silent and total. Goniometry measures from the
   * ANATOMICAL POSITION — arms hanging at the sides — where abduction runs 0° to
   * 180°. A VRM rest pose is a T-POSE, arms already horizontal, which is 90° of
   * abduction before the rig has done anything.
   *
   * Writing `spread: [-45, 180]` from the clinical table therefore allows only
   * 45° of downward travel from horizontal, and the arm physically cannot reach
   * the signer's side. It gets stuck out and forward — which is exactly what it
   * did, until an end-to-end check measured the wrist landing 150% of a shoulder
   * width from where the sign put it.
   *
   * Re-expressed against the T-pose:
   *   spread  −140  arm down past the side and across the body
   *            +95  arm raised overhead
   *   flex     −90  arm swung behind the coronal plane
   *           +135  arm swung forward and across
   *
   * Deliberately generous. The shoulder genuinely has the largest range in the
   * body, and it is the joint where over-constraining does the most damage and
   * anatomical limits buy the least — an impossible shoulder reads as an odd
   * posture, while an impossible finger destroys a handshape.
   */
  leftUpperArm: limit([-90, 135], [-140, 95], [-90, 90]),

  /* Elbow: flexion 150 from straight, a few degrees of hyperextension, and the
     forearm's pronation/supination as twist (±85).

     The T-pose has the elbow straight, so here the clinical figures DO transfer
     directly. Putting supination on this bone rather than the wrist is what
     makes palm orientation reachable at all: it is a forearm motion, and
     modelling it at the wrist — which has about ±12° — is why the previous build
     could not turn a palm over. Lateral play is ±8 rather than ±4 because the
     twist solve nudges the hinge slightly off-axis to reach a measured roll. */
  leftLowerArm: limit([-4, 150], [-8, 8], [-85, 85]),

  // Wrist: flexion 80 / extension 70, radial 20 / ulnar 30. Twist is tiny
  // because the real twist happened at the elbow.
  leftHand: limit([-70, 80], [-20, 30], [-12, 12]),
};

/**
 * Finger limits, by joint type.
 *
 *   MCP — flexion 90, hyperextension 30, abduction/adduction ±20. Spread is the
 *         axis that makes a "4" handshape different from a flat closed hand, so
 *         it must exist; it is also the axis that lets fingers pass through each
 *         other if it is not bounded.
 *   PIP — a pure hinge, 110 of flexion, no hyperextension.
 *   DIP — a hinge, 80 of flexion and about 10 of hyperextension, which real
 *         fingers do have and which is visible in a flat "B" handshape.
 */
const MCP = (): JointLimit => limit([-30, 90], [-20, 20], [-6, 6]);
const PIP = (): JointLimit => limit([0, 110], [-3, 3], [-3, 3]);
const DIP = (): JointLimit => limit([-10, 80], [-3, 3], [-3, 3]);

for (const finger of FINGER_ORDER) {
  if (finger === "thumb") continue;
  LEFT_LIMITS[boneOf("left", finger, 0)] = MCP();
  LEFT_LIMITS[boneOf("left", finger, 1)] = PIP();
  LEFT_LIMITS[boneOf("left", finger, 2)] = DIP();
}

/* The thumb is not a short finger and cannot use the finger limits.
   Its carpometacarpal joint is a saddle with two large, coupled ranges plus a
   real axial rotation — that rotation IS opposition, the motion that brings the
   thumb pad round to face the fingers. Opposition is what separates a "pinch"
   from a "claw", and both occur in the manual alphabet, so it has to be a
   modelled degree of freedom rather than something that falls out of the swing. */
LEFT_LIMITS.leftThumbMetacarpal = limit([-40, 60], [-45, 45], [-45, 45]);
LEFT_LIMITS.leftThumbProximal = limit([-10, 60], [-12, 12], [-10, 10]);
LEFT_LIMITS.leftThumbDistal = limit([-15, 85], [-5, 5], [-5, 5]);

/* Spine and head. Kept modest on purpose: these carry non-manual grammar (a
   negation head shake, a topic nod, a questioning forward lean), and grammar has
   to be unmistakable without the figure appearing to sway. */
const AXIAL_LIMITS: Partial<Record<BoneName, JointLimit>> = {
  hips: limit([-10, 10], [-10, 10], [-20, 20]),
  spine: limit([-15, 20], [-12, 12], [-20, 20]),
  chest: limit([-12, 15], [-10, 10], [-18, 18]),
  upperChest: limit([-10, 12], [-10, 10], [-15, 15]),
  neck: limit([-25, 25], [-18, 18], [-30, 30]),
  head: limit([-35, 30], [-25, 25], [-55, 55]),
};

export const JOINT_LIMITS: Partial<Record<BoneName, JointLimit>> = (() => {
  const all: Partial<Record<BoneName, JointLimit>> = { ...AXIAL_LIMITS };
  for (const [name, value] of Object.entries(LEFT_LIMITS)) {
    if (!value) continue;
    all[name as BoneName] = value;
    const right = name.replace(/^left/, "right") as BoneName;
    all[right] = mirrored(value);
  }
  return all;
})();

/* --------------------------------------------------------------- the pose -- */

/**
 * How much of a pose is measurement and how much is inference.
 *
 * Carried per body part rather than as one number because the parts fail
 * independently and should degrade independently: losing a hand to occlusion
 * must not make the torso relax. That single distinction is most of the fix for
 * "the avatar freaks out when tracking is lost".
 */
export interface PoseConfidence {
  body: number;
  /** [left, right] */
  arms: [number, number];
  /** [left, right] */
  hands: [number, number];
  head: number;
}

export const FULL_CONFIDENCE: PoseConfidence = {
  body: 1,
  arms: [1, 1],
  hands: [1, 1],
  head: 1,
};

export const NO_CONFIDENCE: PoseConfidence = {
  body: 0,
  arms: [0, 0],
  hands: [0, 0],
  head: 0,
};

export function blankConfidence(): PoseConfidence {
  return { body: 0, arms: [0, 0], hands: [0, 0], head: 0 };
}

/**
 * A complete humanoid pose as LOCAL bone rotations against the normalized rig,
 * plus the root offset.
 *
 * This is the interface between everything that produces motion (live tracking,
 * the recorded library, synthesised fingerspelling) and everything that consumes
 * it (the blender, the avatar, the debug view). Nothing downstream of here knows
 * or cares where a pose came from — which is what lets the same blending, the
 * same limits and the same avatar serve all three sources.
 */
export interface HumanoidPose {
  rotations: Map<BoneName, THREE.Quaternion>;
  /** Hip offset from rest, in metres. Small: weight shift and lean only. */
  rootOffset: THREE.Vector3;
  confidence: PoseConfidence;
}

export function blankPose(): HumanoidPose {
  const rotations = new Map<BoneName, THREE.Quaternion>();
  for (const bone of DRIVEN_BONES) rotations.set(bone, new THREE.Quaternion());
  return {
    rotations,
    rootOffset: new THREE.Vector3(),
    confidence: blankConfidence(),
  };
}

/** Reset every rotation to identity without reallocating the map. */
export function resetPose(pose: HumanoidPose): HumanoidPose {
  for (const q of pose.rotations.values()) q.set(0, 0, 0, 1);
  pose.rootOffset.set(0, 0, 0);
  pose.confidence.body = 0;
  pose.confidence.head = 0;
  pose.confidence.arms[0] = pose.confidence.arms[1] = 0;
  pose.confidence.hands[0] = pose.confidence.hands[1] = 0;
  return pose;
}

/**
 * Which confidence channel governs a bone.
 *
 * The blender uses this to decide how hard to trust an incoming rotation and how
 * fast to let that bone relax when the measurement stops. Grouping is by failure
 * mode, not by anatomy: the wrist bone belongs to the ARM group because it is
 * placed by the arm solve, while the fifteen finger bones belong to the HAND
 * group because they come from the hand landmarker and disappear with it.
 */
export type ConfidenceChannel = "body" | "head" | "armL" | "armR" | "handL" | "handR";

export function channelOf(bone: BoneName): ConfidenceChannel {
  if (bone === "neck" || bone === "head") return "head";
  if (bone.startsWith("left")) {
    return isFingerBone(bone) ? "handL" : "armL";
  }
  if (bone.startsWith("right")) {
    return isFingerBone(bone) ? "handR" : "armR";
  }
  return "body";
}

const FINGER_TOKENS = ["Thumb", "Index", "Middle", "Ring", "Little"];

export function isFingerBone(bone: BoneName): boolean {
  return FINGER_TOKENS.some((token) => bone.includes(token));
}

export function readConfidence(
  confidence: PoseConfidence,
  channel: ConfidenceChannel,
): number {
  switch (channel) {
    case "body":
      return confidence.body;
    case "head":
      return confidence.head;
    case "armL":
      return confidence.arms[0];
    case "armR":
      return confidence.arms[1];
    case "handL":
      return confidence.hands[0];
    case "handR":
      return confidence.hands[1];
  }
}

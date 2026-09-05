/**
 * The pose the signer relaxes into when nobody is signing.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS FILE EXISTS
 * ---------------------------------------------------------------------------
 * A VRM rests in a T-POSE: arms straight out, palms down, fingers extended. It
 * is a rigging convention, not a posture — no person has ever stood like that —
 * and the brief names it explicitly:
 *
 *     "never snap, never teleport, never rotate 90°, never enter T-pose"
 *
 * The blender relaxes toward an idle pose whenever a channel loses tracking, so
 * whatever is passed as "idle" is what the figure becomes when the camera cannot
 * see anyone. Passing the identity rotation makes that a T-pose, which is both
 * the thing the brief forbids and, incidentally, terrible for framing: arms held
 * straight out make the figure nearly five times wider than it is when standing
 * normally, so the camera pulls back and the hands — the part that matters —
 * become small.
 *
 * ---------------------------------------------------------------------------
 * WHAT A SIGNER AT REST ACTUALLY DOES
 * ---------------------------------------------------------------------------
 * Not arms hanging dead at the sides either. Between utterances a signer holds a
 * READY position: hands at roughly waist height, slightly forward of the body,
 * elbows softly bent, fingers relaxed into a natural curve. It is the posture
 * you take when you are listening and expect to reply.
 *
 * That matters for more than looks. Returning to a consistent neutral is how a
 * reader knows an utterance has ENDED — in signing, the return to rest is
 * punctuation. A figure that drops its arms limply reads as disengaged; one that
 * holds ready reads as waiting for you.
 *
 * Every angle below is expressed in the CANONICAL frame — +X flexion, +Y along
 * the bone, +Z spread — so one description works for any model. See
 * `retarget/rest.ts`.
 */

import * as THREE from "three";
import type { BoneName, HumanoidPose } from "../../core/humanoid";
import { blankPose, fingerChains } from "../../core/humanoid";
import { DEG } from "../../core/math";
import type { RestPose } from "../retarget/rest";

/**
 * One joint's resting angles, in degrees, in its canonical frame.
 *
 * `flex` closes the joint, `spread` abducts it, `twist` rotates it about its own
 * length — the same three axes every anatomical limit is written in.
 */
interface Resting {
  flex?: number;
  spread?: number;
  twist?: number;
}

/**
 * The arm, from a T-pose.
 *
 * `spread: -72` is the big one: it swings the whole arm down from horizontal to
 * near the side. It is not −90 because a relaxed arm hangs a little away from
 * the body, and an arm pinned flat to the ribs reads as a soldier at attention.
 */
const ARM: Record<string, Resting> = {
  Shoulder: { spread: -3 },
  UpperArm: { spread: -72, flex: 14, twist: 12 },
  // A soft elbow. Fully straight arms are the other half of the mannequin look,
  // and the bend also puts the hands where the next sign will start from.
  LowerArm: { flex: 34, twist: 22 },
  Hand: { flex: -6, spread: 4 },
};

/**
 * A relaxed hand is not a flat hand.
 *
 * Left to itself a hand curls: the fingers rest at roughly 20-40° of flexion,
 * increasing from index to little, and the thumb lies across rather than beside
 * them. A signer's neutral hand does exactly this, and rendering a flat paddle
 * instead is one of the clearest tells that a figure is rigged rather than
 * alive — the eye reads a fully extended hand as a deliberate handshape, which
 * means the avatar appears to be signing something during its own pauses.
 */
const FINGER_REST: Record<string, [number, number, number]> = {
  //            MCP   PIP   DIP
  index: [18, 26, 12],
  middle: [20, 30, 14],
  ring: [24, 34, 16],
  little: [28, 38, 18],
};

/** The thumb rests across the palm, opposed rather than splayed. */
const THUMB_REST: [Resting, Resting, Resting] = [
  { flex: 16, spread: 20, twist: 14 },
  { flex: 14 },
  { flex: 12 },
];

/** A little slump, so the figure is not standing to attention. */
const SPINE: Partial<Record<BoneName, Resting>> = {
  spine: { flex: 2 },
  chest: { flex: 1 },
  neck: { flex: 3 },
  head: { flex: -2 },
};

/**
 * Build the idle pose for a particular body.
 *
 * Takes a `RestPose` because the canonical frames are measured per model — the
 * same angles produce the same posture on any humanoid, which is the whole
 * reason they are expressed this way.
 */
export function buildIdlePose(rest: RestPose): HumanoidPose {
  const pose = blankPose();

  const set = (bone: BoneName, resting: Resting) => {
    const boneRest = rest.bones.get(bone);
    const rotation = pose.rotations.get(bone);
    if (!boneRest || !rotation) return;

    // Compose in canonical space, then conjugate into the bone's own frame.
    C.flex.setFromAxisAngle(AXIS_X, (resting.flex ?? 0) * DEG);
    C.twist.setFromAxisAngle(AXIS_Y, (resting.twist ?? 0) * DEG);
    C.spread.setFromAxisAngle(AXIS_Z, (resting.spread ?? 0) * DEG);
    C.canonical.copy(C.spread).multiply(C.flex).multiply(C.twist);

    C.inverse.copy(boneRest.frame).invert();
    rotation.copy(boneRest.frame).multiply(C.canonical).multiply(C.inverse).normalize();
  };

  for (const [bone, resting] of Object.entries(SPINE)) {
    set(bone as BoneName, resting);
  }

  for (const side of ["left", "right"] as const) {
    const S = side === "left" ? "left" : "right";
    // Spread and twist are defined relative to the body's midline, so they
    // mirror; flexion does not — an elbow closes the same way on both arms.
    const mirror = side === "left" ? 1 : -1;

    for (const [stem, resting] of Object.entries(ARM)) {
      set(`${S}${stem}` as BoneName, {
        flex: resting.flex,
        spread: (resting.spread ?? 0) * mirror,
        twist: (resting.twist ?? 0) * mirror,
      });
    }

    for (const chain of fingerChains(side)) {
      if (chain.finger === "thumb") {
        THUMB_REST.forEach((resting, i) => {
          set(chain.bones[i], {
            flex: resting.flex,
            spread: (resting.spread ?? 0) * mirror,
            twist: (resting.twist ?? 0) * mirror,
          });
        });
        continue;
      }
      const curl = FINGER_REST[chain.finger];
      chain.bones.forEach((bone, i) => set(bone, { flex: curl[i] }));
    }
  }

  // Idle is entirely inference, so it claims no confidence. The blender reads
  // this when deciding how much of the figure is real.
  pose.confidence.body = 0;
  pose.confidence.head = 0;
  pose.confidence.arms[0] = pose.confidence.arms[1] = 0;
  pose.confidence.hands[0] = pose.confidence.hands[1] = 0;
  return pose;
}

const AXIS_X = new THREE.Vector3(1, 0, 0);
const AXIS_Y = new THREE.Vector3(0, 1, 0);
const AXIS_Z = new THREE.Vector3(0, 0, 1);

const C = {
  flex: new THREE.Quaternion(),
  twist: new THREE.Quaternion(),
  spread: new THREE.Quaternion(),
  canonical: new THREE.Quaternion(),
  inverse: new THREE.Quaternion(),
};

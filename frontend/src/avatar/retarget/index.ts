/**
 * The retargeter: one frame of tracking in, one complete humanoid pose out.
 *
 * ---------------------------------------------------------------------------
 * THE POINT OF THIS BOUNDARY
 * ---------------------------------------------------------------------------
 * Everything upstream of here produces measurements — of a live signer, of a
 * recorded sign from the library, or of a synthesised fingerspelled letter.
 * Everything downstream consumes bone ROTATIONS and does not know or care which
 * of those it is holding.
 *
 * That is what lets one avatar, one blender, one set of joint limits and one
 * debug view serve all three sources. The previous build had the live path and
 * the replay path each reaching into the geometry separately, which is why the
 * live view and the reference view of the same sign were different shapes.
 *
 * ---------------------------------------------------------------------------
 * ORDER MATTERS
 * ---------------------------------------------------------------------------
 *   body → arms → hands → fingers
 *
 * Each stage needs the CLAMPED result of the one before it. Solving the fingers
 * against the raw measured wrist rather than the wrist the arm solve actually
 * produced is how a hand ends up floating a few centimetres off the end of its
 * own forearm — visible immediately, and impossible to explain from a still.
 */

import * as THREE from "three";
import type { BoneName, HumanoidPose } from "../../core/humanoid";
import { ARMS, resetPose } from "../../core/humanoid";
import { HandConditioner } from "../../tracking/handConditioner";
import { NEUTRAL_FACE, type NonManual } from "../nonManual";
import {
  makeArmTarget,
  makeBodyFrame,
  solveArm,
  type ArmTarget,
  type BodyFrame,
} from "./arm";
import {
  makeBodyMeasurement,
  measureBody,
  resolveDepth,
  solveBody,
  solveHead,
  POSE,
  type BodyMeasurement,
} from "./body";
import {
  makeHandFrame,
  measureHand,
  mediapipeToScene,
  separateFingers,
  solveFingers,
  type HandFrame,
} from "./hand";
import type { RestPose } from "./rest";

/* --------------------------------------------------- the frame contract --- */

/**
 * The motion library's frame layout, which the live path also produces.
 *
 * Keeping ONE layout for both is deliberate. A learner comparing their own hands
 * to the reference should be looking at the same object posed two ways, not at a
 * polished avatar next to a wireframe of themselves.
 */
export const FRAME = {
  LEFT_WRIST: 0,
  LEFT_WORLD: 2,
  RIGHT_WRIST: 65,
  RIGHT_WORLD: 67,
  POSE_BLOCK: 130,
  FLOATS: 169,
} as const;

export interface RetargetOptions {
  /** Non-manual markers to carry on the head. */
  expression?: NonManual;
  /** Seconds, for the periodic markers. */
  time?: number;
  /** Monotonic clock in ms, for the landmark filters. */
  timeMs?: number;
  /** Multiplies every confidence — the tracker's own opinion of this frame. */
  quality?: number;
}

/**
 * Stateful because the hand conditioners are: filtering needs history, and a
 * fresh instance per frame would filter nothing.
 *
 * One instance per pose SOURCE, not per avatar. The live camera and the library
 * player each need their own, or a sign starting while the camera is running
 * would be filtered against the user's hand.
 */
export class PoseRetargeter {
  private readonly rest: RestPose;
  private readonly conditioners = [new HandConditioner(), new HandConditioner()];
  private readonly handFrames: [HandFrame, HandFrame] = [
    makeHandFrame(),
    makeHandFrame(),
  ];
  private readonly armTargets: [ArmTarget, ArmTarget] = [
    makeArmTarget(),
    makeArmTarget(),
  ];
  private readonly measurement: BodyMeasurement = makeBodyMeasurement();
  private readonly bodyFrame: BodyFrame = makeBodyFrame();
  private readonly handWorld: [THREE.Quaternion, THREE.Quaternion] = [
    new THREE.Quaternion(),
    new THREE.Quaternion(),
  ];
  private readonly scene = new Float32Array(63);
  private readonly shoulderWorld: [THREE.Vector3, THREE.Vector3] = [
    new THREE.Vector3(),
    new THREE.Vector3(),
  ];
  private readonly shoulderMid = new THREE.Vector3();
  private clock = 0;

  constructor(rest: RestPose) {
    this.rest = rest;
  }

  reset(): void {
    for (const conditioner of this.conditioners) conditioner.reset();
    this.clock = 0;
  }

  /** The last body measurement, for the debug view. */
  get lastMeasurement(): BodyMeasurement {
    return this.measurement;
  }

  /**
   * The goals the arm IK was actually given.
   *
   * Exposed because "did the solver reach its target" and "was the target right"
   * are different questions, and conflating them hid a real defect: a check
   * comparing the rig against the POSE model's wrist was really measuring the
   * disagreement between two trackers, not the solver's error.
   */
  get lastArmTargets(): readonly ArmTarget[] {
    return this.armTargets;
  }

  /**
   * Retarget one frame into `pose`.
   *
   * Returns false when the frame carries no usable body, in which case `pose` is
   * left untouched — NOT zeroed. A caller that zeroed it would snap the avatar
   * to its rest pose the instant a shoulder was occluded, which is precisely the
   * lurch this rebuild exists to remove. Deciding what to do about a missing
   * frame belongs to the blender, which has the history to do it smoothly.
   */
  solve(
    frame: Float32Array,
    pose: HumanoidPose,
    options: RetargetOptions = {},
  ): boolean {
    const expression = options.expression ?? NEUTRAL_FACE;
    const time = options.time ?? 0;
    this.clock = options.timeMs ?? this.clock + 1000 / 30;
    const quality = options.quality ?? 1;

    const measurement = measureBody(frame, FRAME.POSE_BLOCK, this.rest, this.measurement);
    if (!measurement.valid) {
      pose.confidence.body = 0;
      pose.confidence.head = 0;
      pose.confidence.arms[0] = pose.confidence.arms[1] = 0;
      pose.confidence.hands[0] = pose.confidence.hands[1] = 0;
      return false;
    }

    /* ------------------------------------------------------------- body -- */
    solveBody(measurement, this.rest, pose, this.bodyFrame);

    /* Re-anchor signing space onto the shoulders the avatar will actually have.
     *
     * The spine solve turns the torso, which SWINGS THE SHOULDERS along an arc —
     * up to about 12 cm on a normal signing turn. Measured signing space is
     * anchored at the model's REST shoulder midpoint, so unless it is moved with
     * them, the IK solves an arm whose origin is not where the arm is. The hand
     * then lands consistently short, and it looks like an IK failure rather than
     * an anchoring one: measured at 33% of a shoulder width, it was the largest
     * single source of error in the whole pipeline.
     *
     * A short forward-kinematic walk down the spine gives the real positions;
     * every target then shifts by the same amount, so the sign keeps its place
     * relative to the body — which is where its meaning lives. */
    this.locateShoulders(pose);
    S.shift.copy(this.shoulderMid).sub(this.rest.shoulderMid);
    measurement.shoulderL.add(S.shift);
    measurement.shoulderR.add(S.shift);
    measurement.shoulderMid.add(S.shift);
    for (let i = 0; i < 2; i += 1) {
      measurement.wrist[i]?.add(S.shift);
      measurement.elbow[i]?.add(S.shift);
    }
    measurement.head?.add(S.shift);
    this.bodyFrame.shoulder[0].copy(this.shoulderWorld[0]);
    this.bodyFrame.shoulder[1].copy(this.shoulderWorld[1]);
    solveHead(
      measurement,
      expression,
      time,
      this.rest,
      this.bodyFrame.chestRotation,
      pose,
    );
    pose.confidence.body = quality;
    pose.confidence.head = measurement.head ? quality : quality * 0.4;

    /* -------------------------------------------------------- the hands -- */
    // Measured BEFORE the arms, because the arm solve needs the wrist rotation
    // the hand implies. The finger solve then runs after, against the arm's
    // clamped output.
    for (let block = 0; block < 2; block += 1) {
      const wristSlot = block === 0 ? FRAME.LEFT_WRIST : FRAME.RIGHT_WRIST;
      const worldSlot = block === 0 ? FRAME.LEFT_WORLD : FRAME.RIGHT_WORLD;
      const side = block === 0 ? "left" : "right";
      const handFrame = this.handFrames[block];
      const target = this.armTargets[block];

      const seen = frame[wristSlot] !== 0 || frame[wristSlot + 1] !== 0;
      if (!seen) {
        this.conditioners[block].reset();
        handFrame.valid = false;
        pose.confidence.hands[block] = 0;
      } else {
        mediapipeToScene(frame, worldSlot, this.scene, 0);
        const conditioned = this.conditioners[block].update(this.scene, 0, this.clock);
        if (!conditioned.valid) {
          handFrame.valid = false;
          pose.confidence.hands[block] = 0;
        } else {
          measureHand(this.conditioners[block].points, 0, side, this.rest, handFrame);
          pose.confidence.hands[block] = handFrame.valid
            ? conditioned.confidence * quality
            : 0;
        }
      }

      /* Where the wrist actually is.

         The hand landmarker's own wrist is preferred for x and y: it is the
         point the hand geometry is built around, so using anything else detaches
         the hand from its own arm, and it tracks better than the pose model's
         wrist — which is routinely extrapolated below the bottom of a chest-up
         frame.

         DEPTH IS NOT TAKEN FROM EITHER. Neither the hand block nor the pose
         block carries a usable metric z (see `resolveDepth`), so it is rebuilt
         from the arm's own bone lengths, using the pose hint only for which way
         to lean. That is what keeps the target inside the arm's reach. */
      const poseWrist = measurement.wrist[block];
      const poseElbow = measurement.elbow[block];
      if (seen) {
        const scale = measurement.scale;
        const anchor = this.rest.shoulderMid;
        // The image-space wrist shares the pose block's normalization, so it is
        // placed through the same transform.
        target.wrist.set(
          (frame[wristSlot] - measurement.centreX) * scale + anchor.x,
          -(frame[wristSlot + 1] - measurement.centreY) * scale + anchor.y,
          // Seed with the pose wrist's depth so `resolveDepth` inherits its
          // sign; the magnitude it carries is discarded.
          poseWrist ? poseWrist.z : anchor.z,
        );
        target.wrist.add(S.shift);
        const lengths = this.rest.armLengths[block];
        const shoulder = this.shoulderWorld[block];

        /* Do the two trackers agree this is the same arm?
         *
         * The hand landmarker and the pose model run independently, and they
         * disagree more often than is comfortable — the hand model will happily
         * report two hands on the same side of the body, or label a right hand
         * "Left". When that happens the hand block and the pose block describe
         * DIFFERENT arms, and resolving depth against the wrong elbow throws the
         * wrist right across the signing space. Measured over the library this is
         * the entire tail of the placement error: the median is 10% of a shoulder
         * width and the 95th percentile was 128%.
         *
         * Half a shoulder width is far larger than the two trackers' honest
         * disagreement about one wrist and far smaller than the distance to the
         * other arm, so it separates the two cases cleanly. On a mismatch the
         * pose model wins: it is the one that knows which arm is which. */
        const disagreement = poseWrist ? poseWrist.distanceTo(target.wrist) : 0;
        const paired = !poseWrist || disagreement < this.rest.shoulderWidth * 0.5;

        if (!paired && poseWrist) {
          target.wrist.copy(poseWrist);
          // The hand's SHAPE is still usable — it is a real hand, just not the
          // one the pose model was describing — but it must not be trusted to
          // place the arm, so its confidence is cut rather than zeroed.
          pose.confidence.hands[block] *= 0.35;
        } else if (poseElbow) {
          resolveDepth(poseElbow, target.wrist, lengths.lower);
        } else {
          resolveDepth(shoulder, target.wrist, lengths.upper + lengths.lower);
        }
      } else if (poseWrist) {
        target.wrist.copy(poseWrist);
      } else {
        pose.confidence.arms[block] = 0;
        continue;
      }

      target.elbow = measurement.elbow[block];
      target.elbowConfidence = measurement.elbow[block] ? quality : 0;
      target.hasHand = handFrame.valid;
      target.handRotation.copy(handFrame.rotation);

      pose.confidence.arms[block] =
        quality * (0.5 + (measurement.elbow[block] ? 0.25 : 0) + (seen ? 0.25 : 0));
    }

    /* --------------------------------------------------------- the arms -- */
    for (let i = 0; i < ARMS.length; i += 1) {
      if (pose.confidence.arms[i] <= 0) continue;
      solveArm(
        ARMS[i],
        this.armTargets[i],
        this.bodyFrame,
        this.rest,
        pose,
        this.handWorld[i],
      );
    }

    /* ------------------------------------------------------ the fingers -- */
    for (let block = 0; block < 2; block += 1) {
      if (!this.handFrames[block].valid) continue;
      const side = block === 0 ? "left" : "right";
      solveFingers(
        this.conditioners[block].points,
        0,
        side,
        this.rest,
        this.handWorld[block],
        pose,
      );
      separateFingers(side, this.rest, pose);
    }

    return true;
  }

  /**
   * Walk hips → spine → chest → upperChest → shoulder → upperArm and record
   * where each shoulder joint ends up once the spine solve has been applied.
   *
   * Only six bones, so it is far cheaper than it looks — and it has to be done
   * here rather than read back from the avatar, because the avatar has not been
   * posed yet this frame and the arms are solved before it is.
   */
  private locateShoulders(pose: HumanoidPose): void {
    for (let side = 0; side < 2; side += 1) {
      const chain = side === 0 ? LEFT_ROOT : RIGHT_ROOT;
      S.world.identity();
      S.position.set(0, 0, 0);
      for (const bone of chain) {
        const boneRest = this.rest.bones.get(bone);
        if (!boneRest) continue;
        // Offsets are expressed in the parent's frame, so they rotate with it.
        S.step.copy(boneRest.offset).applyQuaternion(S.world);
        S.position.add(S.step);
        const rotation = pose.rotations.get(bone);
        if (rotation) S.world.multiply(rotation);
      }
      this.shoulderWorld[side].copy(S.position);
    }
    this.shoulderMid
      .copy(this.shoulderWorld[0])
      .add(this.shoulderWorld[1])
      .multiplyScalar(0.5);
  }

  /** Reset a pose to rest — used when a source is swapped out entirely. */
  static clear(pose: HumanoidPose): HumanoidPose {
    return resetPose(pose);
  }
}

/** Spine chains down to each shoulder joint, in composition order. */
const LEFT_ROOT: BoneName[] = [
  "hips", "spine", "chest", "upperChest", "leftShoulder", "leftUpperArm",
];
const RIGHT_ROOT: BoneName[] = [
  "hips", "spine", "chest", "upperChest", "rightShoulder", "rightUpperArm",
];

const S = {
  shift: new THREE.Vector3(),
  world: new THREE.Quaternion(),
  position: new THREE.Vector3(),
  step: new THREE.Vector3(),
};

export { POSE };
export type { BodyMeasurement } from "./body";
export type { RestPose, BoneRest } from "./rest";

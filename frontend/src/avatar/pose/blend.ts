/**
 * What the avatar actually wears — and what happens when the tracker stops.
 *
 * ---------------------------------------------------------------------------
 * THE FAILURE THIS REPLACES
 * ---------------------------------------------------------------------------
 * The previous build's response to losing tracking was:
 *
 *     tracking disappears → arms jump → body twitches → pose goes random
 *
 * Three separate causes, all of them structural:
 *
 *   1. THE SWITCH WAS BINARY. `measured ?? REST` — the instant both shoulders
 *      dropped below the visibility threshold, the target flipped from the
 *      tracked pose to a rest pose most of a body-width away, and a spring
 *      started dragging the arms down. A momentary shoulder occlusion, which
 *      happens constantly, produced a visible lurch.
 *
 *   2. THE WHOLE BODY RELAXED TOGETHER. One global smoothing constant, so losing
 *      a HAND relaxed the torso. The parts fail independently and must degrade
 *      independently.
 *
 *   3. THERE WAS NO PREDICTION. A hand that vanished mid-movement froze on its
 *      last frame and then teleported to wherever it reappeared. Freezing is not
 *      a neutral choice: a hand stopping dead mid-sign is a strong, wrong signal,
 *      because in sign language a hold IS a phoneme.
 *
 * ---------------------------------------------------------------------------
 * THE STATE MACHINE
 * ---------------------------------------------------------------------------
 *                    confidence high, 3 frames
 *        ACQUIRING ─────────────────────────────▶ TRACKING
 *            ▲                                        │ confidence low
 *            │ confidence recovers                    ▼
 *            │                                    COASTING
 *            │                              (predict, decaying)
 *            │                                        │ 350 ms
 *            └──────────────── IDLE ◀─────────────────┘
 *
 * Per CHANNEL, not per body: `handL` can be COASTING while `body` is TRACKING.
 * That single change is most of the fix — the torso simply does not know that a
 * hand was lost.
 *
 * Every transition is a duration, never an assignment. Nothing in this file can
 * produce a discontinuity, because every bone is driven through a critically
 * damped quaternion spring and a spring has no way to jump.
 */

import * as THREE from "three";
import type {
  BoneName,
  ConfidenceChannel,
  HumanoidPose,
} from "../../core/humanoid";
import {
  DRIVEN_BONES,
  channelOf,
  readConfidence,
} from "../../core/humanoid";
import {
  DampedQuaternion,
  clamp,
  clampInFrame,
  quatAngle,
  quatToRotationVector,
  rotationVectorToQuat,
  smoothstep,
} from "../../core/math";
import { JOINT_LIMITS } from "../../core/humanoid";
import type { RestPose } from "../retarget/rest";

export type TrackingState = "idle" | "acquiring" | "tracking" | "coasting";

/** Confidence above which a channel is believed. */
const ACQUIRE_THRESHOLD = 0.55;
/** Confidence below which a channel starts coasting. */
const LOSE_THRESHOLD = 0.30;
/** Frames of agreement before ACQUIRING becomes TRACKING. */
const ACQUIRE_FRAMES = 3;
/**
 * How long a channel coasts before giving up and easing to idle.
 *
 * 350 ms is chosen from what it has to hide: MediaPipe's typical dropout while a
 * hand crosses the face is 3–8 frames, which at 30 fps is 100–270 ms. Covering
 * that makes the majority of real occlusions invisible. Much longer and a hand
 * that genuinely left the frame hangs in the air, which is worse than a hand
 * that put itself away.
 */
const COAST_SECONDS = 0.35;

/**
 * Hard ceiling on how fast any bone may turn, in radians per second.
 *
 * The springs alone do NOT guarantee this, and an end-to-end run over the real
 * library is what proved it: the motion library is 15 fps, the rig renders at
 * 60, and a fast sign moves a joint far enough between source frames that even a
 * 55 ms spring covers 39° on the first frame after the jump. That is a snap by
 * any definition.
 *
 * So the guarantee is made structural rather than emergent. Whatever the source
 * does — a 15 fps recording, a dropped frame, a teleporting mis-detection, a
 * queue switching signs mid-motion — no bone can exceed a speed a human joint
 * can reach. 18 rad/s is above a fast fingerspelling flick and far below a jump.
 *
 * This costs nothing when it is not needed: ordinary signing never approaches it,
 * so the limiter is inert during normal playback and only engages on the
 * discontinuities it exists to absorb.
 */
const MAX_ANGULAR_SPEED = 18;

/**
 * Smoothing times, in seconds, per state.
 *
 * TRACKING is fast because a sign's speed profile is part of its identity —
 * smoothing a fast sign into a slow one changes what it means. IDLE is slow
 * because dropping the arms is a large move and doing it at tracking speed is
 * exactly the lurch being designed out.
 */
const SMOOTHING = {
  tracking: 0.055,
  acquiring: 0.12,
  coasting: 0.10,
  idle: 0.45,
} as const;

/** Per-channel bookkeeping. */
interface Channel {
  state: TrackingState;
  /** 0-1, smoothed. Drives how far toward idle the channel has drifted. */
  presence: number;
  confidence: number;
  agreeFrames: number;
  coastTime: number;
}

function makeChannel(): Channel {
  return {
    state: "idle",
    presence: 0,
    confidence: 0,
    agreeFrames: 0,
    coastTime: 0,
  };
}

const CHANNELS: ConfidenceChannel[] = [
  "body",
  "head",
  "armL",
  "armR",
  "handL",
  "handR",
];

export interface BlendReport {
  states: Record<ConfidenceChannel, TrackingState>;
  presence: Record<ConfidenceChannel, number>;
  /** Largest per-frame rotation applied to any bone, radians — a snap detector. */
  peakStep: number;
}

/**
 * Blends measured poses into the pose the avatar wears.
 *
 * Owns one spring per bone. `apply` is called every animation frame — including
 * frames with no new measurement, because the relaxation toward idle has to keep
 * running when the input stops.
 */
export class PoseBlender {
  /** The pose the avatar should wear right now. */
  readonly output: HumanoidPose;

  private readonly springs = new Map<BoneName, DampedQuaternion>();
  private readonly channels = new Map<ConfidenceChannel, Channel>();
  private readonly idle: HumanoidPose;
  private readonly rest: RestPose | null;
  private readonly target = new THREE.Quaternion();
  private readonly previous = new THREE.Quaternion();
  private readonly delta = new THREE.Quaternion();
  private readonly axis = new THREE.Vector3();
  private report: BlendReport;
  /**
   * False until the first frame has been applied.
   *
   * The first frame has no previous pose, so the springs seed directly onto the
   * measurement and every bone "moves" the whole way there at once. That is a
   * cold start, not a snap — there was no earlier pose for a viewer to see it
   * jump from — and reporting it as one would make the snap detector cry wolf on
   * every single session.
   */
  private warm = false;

  /**
   * @param blank    a fresh pose to own as output
   * @param idlePose the pose to relax into — normally the model's rest pose
   */
  constructor(blank: HumanoidPose, idlePose: HumanoidPose, rest: RestPose | null = null) {
    this.output = blank;
    this.idle = idlePose;
    this.rest = rest;
    for (const bone of DRIVEN_BONES) this.springs.set(bone, new DampedQuaternion());
    for (const channel of CHANNELS) this.channels.set(channel, makeChannel());
    this.report = {
      states: Object.fromEntries(CHANNELS.map((c) => [c, "idle"])) as BlendReport["states"],
      presence: Object.fromEntries(CHANNELS.map((c) => [c, 0])) as BlendReport["presence"],
      peakStep: 0,
    };
  }

  reset(): void {
    this.warm = false;
    for (const spring of this.springs.values()) spring.reset();
    for (const channel of this.channels.values()) {
      channel.state = "idle";
      channel.presence = 0;
      channel.confidence = 0;
      channel.agreeFrames = 0;
      channel.coastTime = 0;
    }
  }

  get lastReport(): BlendReport {
    return this.report;
  }

  /**
   * Advance one frame.
   *
   * `measured` is the retargeted pose, or null when there is no measurement at
   * all this frame. Null is NOT the same as zero confidence: a null frame means
   * "nothing arrived", and every channel coasts on its own history.
   */
  apply(measured: HumanoidPose | null, dt: number): BlendReport {
    const step = clamp(dt, 1 / 240, 0.1);
    let peakStep = 0;

    /* ----------------------------------------------------- state machine -- */
    for (const name of CHANNELS) {
      const channel = this.channels.get(name)!;
      const confidence = measured ? readConfidence(measured.confidence, name) : 0;
      channel.confidence = confidence;
      advance(channel, confidence, step);
      this.report.states[name] = channel.state;
      this.report.presence[name] = channel.presence;
    }

    /* ------------------------------------------------------------ bones -- */
    for (const bone of DRIVEN_BONES) {
      const spring = this.springs.get(bone)!;
      const current = this.output.rotations.get(bone);
      if (!current) continue;
      this.previous.copy(current);

      const channel = this.channels.get(channelOf(bone))!;
      const idle = this.idle.rotations.get(bone) ?? IDENTITY;
      const source = measured?.rotations.get(bone);

      if (channel.state === "coasting") {
        /* Coast: keep turning the way the joint was turning, decaying to a stop.
           This is what hides a short occlusion. Note the spring is NOT given a
           target — feeding it the last measurement would freeze the joint, and a
           frozen hand mid-sign reads as a HOLD, which in sign language is a
           phoneme rather than an absence. */
        spring.coast(step, 0.015);
        // Then fade what is left toward idle over the coast window, so a channel
        // that never recovers arrives at rest having already slowed down.
        const fade = smoothstep(channel.coastTime / COAST_SECONDS) * 0.35;
        if (fade > 0) current.copy(spring.value).slerp(idle, fade * step * 6);
        else current.copy(spring.value);
      } else {
        /* Blend the measurement toward idle by how present the channel is, then
           spring toward that. Doing it in this order matters: springing toward
           the measurement and separately fading to idle would let the two fight,
           and the joint would settle somewhere neither wanted. */
        if (source && channel.presence > 0.001) {
          this.target.copy(idle).slerp(source, channel.presence);
        } else {
          this.target.copy(idle);
        }
        spring.step(this.target, smoothingFor(channel.state), step);
        current.copy(spring.value);
      }

      /* Two guarantees applied to the blended result, in this order.

         CLAMP FIRST, then rate-limit. The opposite order was tried and is wrong:
         the anatomy clamp is itself capable of moving a rotation a long way — it
         is the thing that catches a joint the moment it strays — so limiting the
         rate before it leaves the clamp free to reintroduce the jump. Measured on
         the real library, that ordering still produced 160°/frame. Rate-limiting
         LAST makes the ceiling unconditional: whatever any earlier stage decides,
         nothing reaches the avatar faster than a human joint can move. */
      if (this.rest) {
        const limit = JOINT_LIMITS[bone];
        const boneRest = this.rest.bones.get(bone);
        /* A slerp between two rotations that are each inside the limit box can
           leave it — the geodesic between them is not a straight line in
           swing-space. Measured on the real library, an upper arm blending
           between rest and a raised pose bulged 14° past its own limit. Clamping
           the OUTPUT is the only place that can be caught. */
        if (limit && boneRest) {
          clampInFrame(current, boneRest.frame, limit.flex, limit.spread, limit.twist);
        }
      }
      if (this.warm) this.limitRate(current, step);

      const moved = this.warm ? quatAngle(this.previous, current) : 0;
      if (Number.isFinite(moved) && moved > peakStep) peakStep = moved;
    }

    /* Root offset: small lean and weight shift only, and it follows the body
       channel's presence so a lost body does not leave the figure leaning. */
    const body = this.channels.get("body")!;
    if (measured && body.presence > 0.001) {
      this.output.rootOffset.lerp(measured.rootOffset, 1 - Math.exp(-6 * step));
    } else {
      this.output.rootOffset.lerp(ZERO, 1 - Math.exp(-3 * step));
    }

    // Report presence as the output's confidence, so anything downstream — the
    // renderer's fade, the debug panel — reads how much of this pose is real.
    const c = this.output.confidence;
    c.body = this.channels.get("body")!.presence;
    c.head = this.channels.get("head")!.presence;
    c.arms[0] = this.channels.get("armL")!.presence;
    c.arms[1] = this.channels.get("armR")!.presence;
    c.hands[0] = this.channels.get("handL")!.presence;
    c.hands[1] = this.channels.get("handR")!.presence;

    this.warm = true;
    this.report.peakStep = peakStep;
    return this.report;
  }

  /**
   * Cap how far a bone turned this frame.
   *
   * Works on the rotation VECTOR of the step, so the direction of the motion is
   * preserved exactly and only its magnitude is reduced — the joint still heads
   * where it was going, it just takes more frames to arrive. Scaling the
   * quaternion's components instead would change the axis as well as the angle,
   * and the bone would visibly curve toward its target.
   */
  private limitRate(current: THREE.Quaternion, dt: number): void {
    const maximum = MAX_ANGULAR_SPEED * dt;
    this.delta.copy(this.previous).invert().premultiply(current);
    quatToRotationVector(this.delta, this.axis);
    const angle = this.axis.length();
    if (!(angle > maximum)) return;
    this.axis.multiplyScalar(maximum / angle);
    rotationVectorToQuat(this.axis, this.delta);
    current.copy(this.previous).premultiply(this.delta).normalize();
  }
}

/** One channel's state transition. */
function advance(channel: Channel, confidence: number, dt: number): void {
  switch (channel.state) {
    case "idle":
      if (confidence >= ACQUIRE_THRESHOLD) {
        channel.state = "acquiring";
        channel.agreeFrames = 1;
      }
      break;

    case "acquiring":
      if (confidence < LOSE_THRESHOLD) {
        channel.state = "idle";
        channel.agreeFrames = 0;
      } else if (confidence >= ACQUIRE_THRESHOLD) {
        channel.agreeFrames += 1;
        // Requiring agreement across frames is what stops a single spurious
        // detection — a face mistaken for a hand for one frame — from yanking
        // the avatar into a pose and back out again.
        if (channel.agreeFrames >= ACQUIRE_FRAMES) channel.state = "tracking";
      }
      break;

    case "tracking":
      if (confidence < LOSE_THRESHOLD) {
        channel.state = "coasting";
        channel.coastTime = 0;
      }
      break;

    case "coasting":
      channel.coastTime += dt;
      if (confidence >= ACQUIRE_THRESHOLD) {
        // Straight back to tracking: the hand was only briefly hidden, and
        // making it re-acquire would add a visible hesitation to every occlusion.
        channel.state = "tracking";
      } else if (channel.coastTime >= COAST_SECONDS) {
        channel.state = "idle";
      }
      break;
  }

  /* Presence: how much of this channel's pose is measurement rather than rest.
     Rises quickly and falls slowly — the asymmetry is deliberate. Coming back
     should feel immediate; going away should not be noticed at all. */
  const wanted = channel.state === "tracking" || channel.state === "coasting" ? 1 : 0;
  const rate = wanted > channel.presence ? 9 : 2.2;
  channel.presence += (wanted - channel.presence) * (1 - Math.exp(-rate * dt));
  channel.presence = clamp(channel.presence, 0, 1);
}

function smoothingFor(state: TrackingState): number {
  return SMOOTHING[state];
}

const IDENTITY = new THREE.Quaternion();
const ZERO = new THREE.Vector3();

/**
 * The blender's guarantees, under the exact conditions that broke the old build.
 *
 * The brief is unusually specific about this, and rightly — every item on its
 * list is a thing a user watched happen:
 *
 *     "if confidence decreases: freeze the last reliable pose, gradually reduce
 *      movement, blend into an idle animation, never snap, never teleport, never
 *      rotate 90°, never enter T-pose."
 *
 * Those are testable. Each scenario below drives the blender through a real
 * failure pattern — a clean loss, a flickering hand, a single spurious frame,
 * a total blackout — and asserts that no bone ever moves faster than a human
 * joint can, that a lost hand does not disturb the torso, and that recovery is
 * immediate rather than hesitant.
 *
 *     npm run check -- blend
 */

import * as THREE from "three";
import type { HumanoidPose } from "../../../core/humanoid";
import { blankPose } from "../../../core/humanoid";
import { DEG, quatAngle } from "../../../core/math";
import { PoseBlender } from "../blend";

let failures = 0;
let checks = 0;

function ok(label: string, condition: boolean, detail = ""): void {
  checks += 1;
  if (condition) return;
  failures += 1;
  console.error(`  FAIL  ${label}${detail ? `  — ${detail}` : ""}`);
}

function section(name: string): void {
  console.log(`\n${name}`);
}

const DT = 1 / 60;

/**
 * The fastest a human joint moves, in radians per second.
 *
 * A fingertip in fast fingerspelling peaks around 20 rad/s; a shoulder is far
 * slower. 25 rad/s is above anything real and far below a teleport, so a frame
 * exceeding it is by definition a snap rather than a fast sign.
 */
const MAX_HUMAN_SPEED = 25;
const MAX_STEP = MAX_HUMAN_SPEED * DT;

/** A pose with every bone rotated by `angle` about x, at a given confidence. */
function posed(angle: number, confidence: number): HumanoidPose {
  const pose = blankPose();
  const q = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), angle);
  for (const rotation of pose.rotations.values()) rotation.copy(q);
  pose.confidence.body = confidence;
  pose.confidence.head = confidence;
  pose.confidence.arms[0] = pose.confidence.arms[1] = confidence;
  pose.confidence.hands[0] = pose.confidence.hands[1] = confidence;
  return pose;
}

/** Run frames and return the largest single-frame rotation of any bone. */
function drive(
  blender: PoseBlender,
  frames: (HumanoidPose | null)[],
): { peak: number; peakFrame: number } {
  let peak = 0;
  let peakFrame = -1;
  frames.forEach((frame, i) => {
    const report = blender.apply(frame, DT);
    if (report.peakStep > peak) {
      peak = report.peakStep;
      peakFrame = i;
    }
  });
  return { peak, peakFrame };
}

function fresh(): PoseBlender {
  return new PoseBlender(blankPose(), blankPose());
}

/* ------------------------------------------------------- clean tracking -- */

section("acquisition");
{
  const blender = fresh();
  const target = posed(0.9, 1);
  const { peak } = drive(blender, Array.from({ length: 120 }, () => target));

  ok("no frame exceeds human joint speed while acquiring", peak < MAX_STEP,
    `${(peak / DEG).toFixed(2)}° in one frame`);

  const report = blender.lastReport;
  ok("reaches TRACKING", report.states.body === "tracking", report.states.body);
  ok("presence saturates", report.presence.body > 0.98,
    report.presence.body.toFixed(3));

  const wrist = blender.output.rotations.get("leftHand")!;
  ok("converges on the measured pose",
    quatAngle(wrist, target.rotations.get("leftHand")!) < 2 * DEG,
    `${(quatAngle(wrist, target.rotations.get("leftHand")!) / DEG).toFixed(2)}° off`);
}

/* --------------------------------------------------------- losing track -- */

section("tracking loss");
{
  const blender = fresh();
  const target = posed(1.1, 1);
  drive(blender, Array.from({ length: 120 }, () => target));

  // Total loss: the input simply stops arriving.
  const { peak, peakFrame } = drive(blender, Array.from({ length: 240 }, () => null));
  ok("no snap when tracking is lost", peak < MAX_STEP,
    `${(peak / DEG).toFixed(2)}° at frame ${peakFrame}`);

  const report = blender.lastReport;
  ok("settles to IDLE", report.states.body === "idle", report.states.body);
  ok("presence falls to zero", report.presence.body < 0.02,
    report.presence.body.toFixed(4));

  // And it must actually arrive at rest, not hang somewhere arbitrary — the
  // "never enter T-pose" requirement is really "end up in the idle pose you were
  // designed to end up in, smoothly".
  const rest = new THREE.Quaternion();
  const arrived = quatAngle(blender.output.rotations.get("leftUpperArm")!, rest);
  ok("relaxes into the idle pose", arrived < 3 * DEG, `${(arrived / DEG).toFixed(2)}° off rest`);
}

section("coasting hides a short occlusion");
{
  const blender = fresh();
  // A hand moving steadily, then vanishing mid-movement.
  for (let i = 0; i < 120; i += 1) {
    blender.apply(posed(0.4 + i * 0.004, 1), DT);
  }
  const beforeLoss = blender.output.rotations.get("leftIndexProximal")!.clone();

  // Six frames blind — a typical MediaPipe dropout as a hand crosses the face.
  const lost = posed(0, 0);
  const { peak } = drive(blender, Array.from({ length: 6 }, () => lost));
  const afterCoast = blender.output.rotations.get("leftIndexProximal")!.clone();

  ok("no snap during the dropout", peak < MAX_STEP, `${(peak / DEG).toFixed(2)}°`);
  // The joint must KEEP MOVING, not freeze. A hand stopping dead mid-sign reads
  // as a hold, and a hold is a phoneme.
  const travelled = quatAngle(beforeLoss, afterCoast);
  ok("the joint keeps moving while blind", travelled > 0.2 * DEG,
    `moved only ${(travelled / DEG).toFixed(3)}°`);
  ok("but not far", travelled < 12 * DEG, `moved ${(travelled / DEG).toFixed(2)}°`);
  ok("channel is COASTING, not idle",
    blender.lastReport.states.handL === "coasting",
    blender.lastReport.states.handL);
}

section("recovery is immediate");
{
  const blender = fresh();
  drive(blender, Array.from({ length: 120 }, () => posed(0.8, 1)));
  drive(blender, Array.from({ length: 5 }, () => posed(0, 0)));
  // One good frame after a short dropout must go straight back to tracking —
  // making it re-acquire adds a visible hesitation to every occlusion.
  blender.apply(posed(0.8, 1), DT);
  ok("returns to TRACKING on the first good frame",
    blender.lastReport.states.handL === "tracking",
    blender.lastReport.states.handL);
}

/* ------------------------------------------------------- independence --- */

section("channels fail independently");
{
  const blender = fresh();
  drive(blender, Array.from({ length: 120 }, () => posed(0.7, 1)));
  const torsoBefore = blender.output.rotations.get("spine")!.clone();

  // Lose ONLY the left hand, for a long time.
  for (let i = 0; i < 200; i += 1) {
    const frame = posed(0.7, 1);
    frame.confidence.hands[0] = 0;
    blender.apply(frame, DT);
  }

  const torsoAfter = blender.output.rotations.get("spine")!;
  const drift = quatAngle(torsoBefore, torsoAfter);
  ok("losing a hand does not move the torso", drift < 1 * DEG,
    `spine drifted ${(drift / DEG).toFixed(2)}°`);
  ok("the lost hand goes idle", blender.lastReport.states.handL === "idle",
    blender.lastReport.states.handL);
  ok("the body stays tracking", blender.lastReport.states.body === "tracking",
    blender.lastReport.states.body);
  ok("the other hand stays tracking", blender.lastReport.states.handR === "tracking",
    blender.lastReport.states.handR);
}

/* ------------------------------------------------------------ hostility -- */

section("hostile input");
{
  // A single spurious high-confidence frame 180° away — a face briefly mistaken
  // for a hand. The avatar must not lunge at it.
  const blender = fresh();
  drive(blender, Array.from({ length: 120 }, () => posed(0.3, 1)));
  const before = blender.output.rotations.get("leftUpperArm")!.clone();
  blender.apply(posed(Math.PI, 1), DT);
  const lunge = quatAngle(before, blender.output.rotations.get("leftUpperArm")!);
  ok("one bad frame does not teleport the arm", lunge < MAX_STEP,
    `${(lunge / DEG).toFixed(2)}° in one frame`);

  // Confidence flickering across the thresholds every frame, which is what a
  // marginal detection actually does. This is the case that made the old build
  // vibrate.
  const flicker = fresh();
  drive(flicker, Array.from({ length: 60 }, () => posed(0.6, 1)));
  let peak = 0;
  for (let i = 0; i < 300; i += 1) {
    const confidence = i % 2 === 0 ? 0.9 : 0.1;
    const report = flicker.apply(posed(0.6, confidence), DT);
    peak = Math.max(peak, report.peakStep);
  }
  ok("flickering confidence does not shake the avatar", peak < MAX_STEP,
    `${(peak / DEG).toFixed(2)}° in one frame`);

  // Wildly varying frame times, including a long stall.
  const stalled = fresh();
  let stallPeak = 0;
  const target = posed(1.2, 1);
  for (const dt of [1 / 60, 1 / 30, 0.25, 1 / 90, 0.1, 1 / 60, 0.4, 1 / 60]) {
    const report = stalled.apply(target, dt);
    stallPeak = Math.max(stallPeak, report.peakStep / dt);
  }
  ok("stable across irregular frame times", stallPeak < MAX_HUMAN_SPEED * 1.5,
    `${stallPeak.toFixed(1)} rad/s`);

  // Nothing may ever produce a non-finite rotation.
  let finite = true;
  for (const rotation of stalled.output.rotations.values()) {
    if (!Number.isFinite(rotation.x + rotation.y + rotation.z + rotation.w)) finite = false;
  }
  ok("no non-finite rotations", finite);
}

console.log(`\n${failures === 0 ? "PASS" : "FAIL"}  ${checks - failures}/${checks} checks`);
if (failures > 0) process.exit(1);

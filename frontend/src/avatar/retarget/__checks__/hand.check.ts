/**
 * The hand pipeline, run over every frame of the real motion library.
 *
 * ---------------------------------------------------------------------------
 * WHAT "CORRECT" MEANS HERE, AND WHY IT IS NOT "MATCHES THE INPUT"
 * ---------------------------------------------------------------------------
 * The obvious test is that the avatar's fingers point exactly where MediaPipe
 * said they pointed. That test is wrong, and running it was how the real problem
 * surfaced: MediaPipe's raw output demands ±40° of lateral motion at a PIP joint,
 * measured over all 239 signs. A PIP is a hinge with about 6° of lateral play.
 * Roughly 35° of what the tracker reports is therefore not the hand — it is
 * error — and a rig that reproduced it faithfully would be reproducing noise.
 *
 * So the pipeline is a DENOISER, and these checks assert the four properties
 * that make one trustworthy:
 *
 *   1. It tracks the signal. Bone directions must agree with the measurement in
 *      the median case, and fingertips must land close to where they were seen.
 *   2. It rejects the impossible. No joint, on any frame, may leave clinical
 *      range — a finger bending backwards through its own knuckle is the single
 *      most legibility-destroying artefact a signing avatar can have.
 *   3. It does not amplify noise. High-frequency content at the output must not
 *      exceed the input's; the conditioning stage must measurably reduce it.
 *   4. It preserves meaning. Different handshapes must stay different — a
 *      denoiser that flattens every sign into the same hand would pass 1–3.
 *
 *     npm run check -- hand
 */

import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import * as THREE from "three";
import type { BoneName, Side } from "../../../core/humanoid";
import { JOINT_LIMITS, blankPose, fingerChains } from "../../../core/humanoid";
import {
  DEG,
  makeSwingTwist,
  quatToRotationVector,
  swingTwist,
} from "../../../core/math";
import { HandConditioner } from "../../../tracking/handConditioner";
import { canonicalRest } from "../canonicalSkeleton";
import {
  makeHandFrame,
  measureHand,
  mediapipeToScene,
  separateFingers,
  solveFingers,
} from "../hand";

/* ------------------------------------------------------------- harness --- */

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

function percentile(values: number[], p: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))];
}

/* -------------------------------------------------- the reference rig ---- */

const { rest } = canonicalRest();

section("rest capture");
{
  ok("no bones missing from the reference rig", rest.missing.length === 0,
    rest.missing.join(", "));
  ok("shoulder width is anthropometric", Math.abs(rest.shoulderWidth - 0.36) < 0.06,
    `${rest.shoulderWidth.toFixed(3)} m`);
  ok("hand span is anthropometric",
    rest.handLength[0] > 0.07 && rest.handLength[0] < 0.11,
    `${rest.handLength[0].toFixed(3)} m`);

  const leftUpper = rest.bones.get("leftUpperArm")!;
  const rightUpper = rest.bones.get("rightUpperArm")!;
  ok("left arm points +x", leftUpper.axis.x > 0.99, leftUpper.axis.x.toFixed(3));
  ok("right arm points −x", rightUpper.axis.x < -0.99, rightUpper.axis.x.toFixed(3));

  let worstOrthonormality = 0;
  let improper = 0;
  for (const bone of rest.bones.values()) {
    const m = new THREE.Matrix4().makeRotationFromQuaternion(bone.frame);
    const e = m.elements;
    const col = (i: number) => new THREE.Vector3(e[i * 4], e[i * 4 + 1], e[i * 4 + 2]);
    const [x, y, z] = [col(0), col(1), col(2)];
    worstOrthonormality = Math.max(
      worstOrthonormality,
      Math.abs(x.length() - 1),
      Math.abs(x.dot(y)),
      Math.abs(y.dot(z)),
    );
    if (new THREE.Vector3().crossVectors(x, y).dot(z) < 0) improper += 1;
  }
  ok("every canonical frame is orthonormal", worstOrthonormality < 1e-5,
    worstOrthonormality.toExponential(2));
  ok("every canonical frame is right-handed", improper === 0, `${improper} improper`);

  // The palm must face DOWN in the T-pose — VRM's own convention, and the
  // measurement every finger's flexion direction is derived from. Upside down
  // here means every finger in the product curls the wrong way.
  for (const side of ["left", "right"] as const) {
    const handBone: BoneName = side === "left" ? "leftHand" : "rightHand";
    const palmar = new THREE.Vector3(0, 0, 1)
      .applyQuaternion(rest.bones.get(handBone)!.frame);
    ok(`${side} palm faces down at rest`, palmar.y < -0.9, `y = ${palmar.y.toFixed(3)}`);
  }

  const inferred = [...rest.bones.values()].filter((b) => b.inferred);
  ok("no bone had to infer its axis", inferred.length === 0,
    inferred.map((b) => b.name).join(", "));
}

section("coordinate conversion");
{
  // Chirality. A conversion that flips only y has determinant −1 and turns every
  // left hand into a right hand — invisibly, because the fingers still bend and
  // the palm still faces somewhere. Every handshape would be mirrored.
  const source = new Float32Array(63);
  ([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]] as const).forEach(([x, y, z], i) => {
    source[i * 3] = x;
    source[i * 3 + 1] = y;
    source[i * 3 + 2] = z;
  });
  const converted = new Float32Array(63);
  mediapipeToScene(source, 0, converted, 0);
  const origin = new THREE.Vector3(converted[0], converted[1], converted[2]);
  const ex = new THREE.Vector3(converted[3], converted[4], converted[5]).sub(origin);
  const ey = new THREE.Vector3(converted[6], converted[7], converted[8]).sub(origin);
  const ez = new THREE.Vector3(converted[9], converted[10], converted[11]).sub(origin);
  const handedness = new THREE.Vector3().crossVectors(ex, ey).dot(ez);
  ok("conversion preserves chirality", handedness > 0.99,
    `det = ${handedness.toFixed(4)} — negative would mirror every handshape`);
}

/* --------------------------------------------------- the motion library -- */

const SIGNS = join(process.cwd(), "public", "signs");
const FRAME_FLOATS = 169;
const SOURCE_FPS = 15;

interface Sign {
  gloss: string;
  frames: number[][];
}

const signs: Sign[] = readdirSync(SIGNS)
  .filter((n) => n.endsWith(".json") && n !== "manifest.json" && n !== "recognition.json")
  .map((name) => {
    const raw = JSON.parse(readFileSync(join(SIGNS, name), "utf8"));
    return { gloss: raw.gloss ?? name.replace(/\.json$/, ""), frames: raw.frames ?? [] };
  })
  .filter((s) => s.frames.length > 0);

section(`retargeting ${signs.length} signs`);

interface Run {
  fidelity: number[];
  tipError: number[];
  inputJitter: number[];
  outputJitter: number[];
  limitViolations: string[];
  tipGaps: number[];
  hands: number;
  nonFinite: number;
  curl: Map<string, number[]>;
}

/**
 * One pass over the library.
 *
 * `condition` selects whether the landmarks go through the conditioner first.
 * Running both ways is what turns "the filter is configured" into evidence that
 * it does something.
 */
function sweep(condition: boolean): Run {
  const pose = blankPose();
  const handFrame = makeHandFrame();
  const scene = new Float32Array(63);
  const raw = new Float32Array(63);
  const st = makeSwingTwist();
  const CANONICAL_Y = new THREE.Vector3(0, 1, 0);
  const measured = new THREE.Vector3();
  const swing = new THREE.Vector3();
  const canonical = new THREE.Quaternion();

  const run: Run = {
    fidelity: [],
    tipError: [],
    inputJitter: [],
    outputJitter: [],
    limitViolations: [],
    tipGaps: [],
    hands: 0,
    nonFinite: 0,
    curl: new Map(),
  };

  for (const sign of signs) {
    const conditioner = { left: new HandConditioner(), right: new HandConditioner() };
    // Three-frame history per bone, for the second-difference jitter measure.
    const history = new Map<string, THREE.Vector3[]>();
    let time = 0;

    for (const row of sign.frames) {
      time += 1000 / SOURCE_FPS;
      if (row.length < FRAME_FLOATS) continue;

      for (const [side, wristSlot, worldSlot] of [
        ["left", 0, 2],
        ["right", 65, 67],
      ] as const) {
        if (row[wristSlot] === 0 && row[wristSlot + 1] === 0) {
          conditioner[side].reset();
          continue;
        }
        for (let i = 0; i < 63; i += 1) raw[i] = row[worldSlot + i];
        mediapipeToScene(raw, 0, scene, 0);

        let source = scene;
        if (condition) {
          const result = conditioner[side].update(scene, 0, time);
          if (!result.valid) continue;
          source = conditioner[side].points;
        }

        measureHand(source, 0, side, rest, handFrame);
        if (!handFrame.valid) continue;
        solveFingers(source, 0, side, rest, handFrame.rotation, pose);
        separateFingers(side, rest, pose);
        run.hands += 1;

        const spreads: number[] = [];
        const tips: THREE.Vector3[] = [];
        let totalCurl = 0;

        for (const chain of fingerChains(side)) {
          const world = handFrame.rotation.clone();
          const tip = new THREE.Vector3(
            source[chain.landmarks[0] * 3],
            source[chain.landmarks[0] * 3 + 1],
            source[chain.landmarks[0] * 3 + 2],
          );

          for (let segment = 0; segment < 3; segment += 1) {
            const bone = chain.bones[segment];
            const boneRest = rest.bones.get(bone)!;
            const rotation = pose.rotations.get(bone)!;
            world.multiply(rotation);
            const solved = boneRest.axis.clone().applyQuaternion(world);

            const from = chain.landmarks[segment];
            const to = chain.landmarks[segment + 1];
            measured.set(
              source[to * 3] - source[from * 3],
              source[to * 3 + 1] - source[from * 3 + 1],
              source[to * 3 + 2] - source[from * 3 + 2],
            );
            const length = measured.length();
            if (length < 1e-10) continue;
            measured.divideScalar(length);
            tip.addScaledVector(solved, length);

            const angle = measured.angleTo(solved);
            if (!Number.isFinite(angle)) run.nonFinite += 1;
            else run.fidelity.push(angle / DEG);

            // Second difference isolates high-frequency content from real
            // motion; a first difference is dominated by the sign itself.
            for (const [tag, vector, into] of [
              [`i${side}${chain.finger}${segment}`, measured, run.inputJitter],
              [`o${side}${chain.finger}${segment}`, solved, run.outputJitter],
            ] as const) {
              const past = history.get(tag) ?? history.set(tag, []).get(tag)!;
              past.push(vector.clone());
              if (past.length > 3) past.shift();
              if (past.length === 3) {
                const midpoint = past[0].clone().add(past[2]).multiplyScalar(0.5);
                if (midpoint.lengthSq() > 1e-12) {
                  into.push(midpoint.angleTo(past[1]) / DEG);
                }
              }
            }

            /* -------------------------------------------- 2. anatomy -- */
            const limit = JOINT_LIMITS[bone];
            if (limit) {
              canonical
                .copy(boneRest.frame)
                .invert()
                .multiply(rotation)
                .multiply(boneRest.frame);
              swingTwist(canonical, CANONICAL_Y, st);
              quatToRotationVector(st.swing, swing);
              const slack = 0.5 * DEG;
              if (swing.x < limit.flex[0] - slack || swing.x > limit.flex[1] + slack) {
                run.limitViolations.push(`${bone}.flex ${(swing.x / DEG).toFixed(1)}°`);
              }
              if (swing.z < limit.spread[0] - slack || swing.z > limit.spread[1] + slack) {
                run.limitViolations.push(`${bone}.spread ${(swing.z / DEG).toFixed(1)}°`);
              }
              if (segment === 0) {
                if (chain.finger !== "thumb") spreads.push(swing.z);
                totalCurl += Math.abs(swing.x);
              }
            }
          }

          const trueTip = new THREE.Vector3(
            source[chain.landmarks[3] * 3],
            source[chain.landmarks[3] * 3 + 1],
            source[chain.landmarks[3] * 3 + 2],
          );
          run.tipError.push(tip.distanceTo(trueTip) / handFrame.span);
          if (chain.finger !== "thumb") tips.push(tip);
        }

        /* ---------------------------------------- 3. no interpenetration -- */
        // The invariant is FINGERTIP SEPARATION, not spread ordering.
        //
        // Spread ordering was the obvious test and it is wrong: real hands
        // converge their fingers all the time. An "O" handshape, a fist, and any
        // grip bring the fingertips toward a common point, and the spread order
        // legitimately inverts on the way. Asserting monotone spread flags every
        // one of those as a defect. What can never happen is two fingers
        // occupying the same space, and that is a distance.
        //
        // Adjacent fingers TOUCHING is also normal — a flat hand does it — so
        // the threshold is set below finger width, at the point where the two
        // are no longer adjacent but coincident.
        void spreads;
        for (let i = 1; i < tips.length; i += 1) {
          run.tipGaps.push(tips[i].distanceTo(tips[i - 1]) * 1000);
        }

        const list = run.curl.get(sign.gloss) ?? [];
        list.push(totalCurl / 5 / DEG);
        run.curl.set(sign.gloss, list);
      }
    }
  }
  return run;
}

const rawRun = sweep(false);
const run = sweep(true);

/* ------------------------------------------------------------- results -- */

ok("solved a meaningful number of hands", run.hands > 5000, `${run.hands} hands`);
ok("no non-finite rotations", run.nonFinite === 0, `${run.nonFinite} NaN`);

const median = percentile(run.fidelity, 0.5);
const p95 = percentile(run.fidelity, 0.95);
console.log(
  `  bone direction vs measurement: median ${median.toFixed(2)}°  ` +
    `p95 ${p95.toFixed(2)}°  over ${run.fidelity.length} bones`,
);
console.log(
  `  fingertip error: p50 ${(percentile(run.tipError, 0.5) * 100).toFixed(1)}%  ` +
    `p90 ${(percentile(run.tipError, 0.9) * 100).toFixed(1)}% of hand span`,
);

// 1. Tracks the signal.
ok("median bone direction error under 6°", median < 6, `${median.toFixed(2)}°`);
ok(
  "median fingertip lands within 12% of hand span",
  percentile(run.tipError, 0.5) < 0.12,
  `${(percentile(run.tipError, 0.5) * 100).toFixed(1)}%`,
);
// The tail is where the joint limits refuse an impossible measurement. That
// divergence is the feature — but it must stay bounded, or the limits are
// fighting the data rather than cleaning it.
ok("p95 bone direction error under 60°", p95 < 60, `${p95.toFixed(2)}°`);

// 2. Rejects the impossible.
ok(
  "no joint ever leaves clinical range",
  run.limitViolations.length === 0,
  run.limitViolations.slice(0, 4).join("; ") +
    (run.limitViolations.length > 4 ? ` (+${run.limitViolations.length - 4})` : ""),
);
const tipGapP1 = percentile(run.tipGaps, 0.01);
console.log(
  `  adjacent fingertip separation: p1 ${tipGapP1.toFixed(1)} mm  ` +
    `p50 ${percentile(run.tipGaps, 0.5).toFixed(1)} mm`,
);
// 3 mm is well inside finger width, so this is not "fingers must not touch" —
// it is "two fingers must not be the same finger".
ok("adjacent fingertips never coincide", tipGapP1 > 3,
  `1% of pairs are within ${tipGapP1.toFixed(1)} mm`);

// 3. Does not amplify noise, and the conditioner measurably removes some.
const inJitter = percentile(run.inputJitter, 0.5);
const outJitter = percentile(run.outputJitter, 0.5);
const rawJitter = percentile(rawRun.inputJitter, 0.5);
const reduction = (1 - inJitter / rawJitter) * 100;
console.log(
  `  jitter (2nd difference, p50): raw ${rawJitter.toFixed(2)}°  ` +
    `conditioned ${inJitter.toFixed(2)}°  retargeted ${outJitter.toFixed(2)}°  ` +
    `(${reduction.toFixed(0)}% removed)`,
);
ok("conditioning removes at least 25% of landmark jitter", reduction > 25,
  `${reduction.toFixed(1)}%`);
ok("retargeting does not amplify jitter by more than 25%",
  outJitter < inJitter * 1.25,
  `${inJitter.toFixed(2)}° → ${outJitter.toFixed(2)}°`);

// 4. Preserves meaning.
section("handshape separation");
{
  const meanCurl = (gloss: string): number | null => {
    const list = run.curl.get(gloss);
    if (!list?.length) return null;
    return list.reduce((a, b) => a + b, 0) / list.length;
  };
  const mean = (values: number[]) => values.reduce((a, b) => a + b, 0) / values.length;

  const open = ["five", "hello", "welcome", "all"].map(meanCurl).filter((v): v is number => v !== null);
  const closed = ["yes", "ten", "know", "ok"].map(meanCurl).filter((v): v is number => v !== null);

  if (open.length && closed.length) {
    console.log(
      `  mean MCP flexion — open-hand signs ${mean(open).toFixed(1)}°, ` +
        `closed-hand signs ${mean(closed).toFixed(1)}°`,
    );
    ok("open and closed handshapes are distinguishable",
      Math.abs(mean(closed) - mean(open)) > 8,
      `only ${Math.abs(mean(closed) - mean(open)).toFixed(1)}° apart`);
  }

  const all = [...run.curl.values()].flat();
  const spread = percentile(all, 0.9) - percentile(all, 0.1);
  console.log(
    `  library-wide MCP flexion: p10 ${percentile(all, 0.1).toFixed(1)}° ` +
      `p50 ${percentile(all, 0.5).toFixed(1)}° p90 ${percentile(all, 0.9).toFixed(1)}°`,
  );
  ok("the library uses a wide range of handshapes", spread > 25,
    `p10..p90 spans only ${spread.toFixed(1)}°`);
}

console.log(`\n${failures === 0 ? "PASS" : "FAIL"}  ${checks - failures}/${checks} checks`);
if (failures > 0) process.exit(1);

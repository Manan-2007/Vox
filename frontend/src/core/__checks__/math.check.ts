/**
 * Numerical checks for the rotation maths.
 *
 * These are not decoration. Every bone in the rig is driven through
 * `DampedQuaternion` and clamped through `clampSwingTwist`, so an error here is
 * an error on all sixty of them at once — and rotation bugs are exactly the kind
 * that look "sort of right" on screen while quietly making a handshape
 * unreadable. Each check below corresponds to a specific way the previous build
 * failed:
 *
 *   * the shorter-arc check is the 360° flip
 *   * the swing/twist round trip is "wrists rotate incorrectly"
 *   * the clamp checks are "fingers intersect" and "elbows bend unnaturally"
 *   * the spring checks are "never snap" and "never teleport"
 *
 *     npm run check
 */

import * as THREE from "three";
import {
  AXIS_X,
  AXIS_Y,
  AXIS_Z,
  DEG,
  DampedQuaternion,
  DampedVector3,
  basisRotation,
  clampSwingTwist,
  makeSwingTwist,
  quatAngle,
  quatToRotationVector,
  rotationBetween,
  rotationVectorToQuat,
  swingTwist,
  twistAngle,
} from "../math";

let failures = 0;
let checks = 0;

function ok(label: string, condition: boolean, detail = ""): void {
  checks += 1;
  if (condition) return;
  failures += 1;
  console.error(`  FAIL  ${label}${detail ? `  — ${detail}` : ""}`);
}

function close(label: string, a: number, b: number, tolerance: number): void {
  ok(label, Math.abs(a - b) <= tolerance, `${a.toFixed(6)} vs ${b.toFixed(6)}`);
}

function section(name: string): void {
  console.log(`\n${name}`);
}

/* --------------------------------------------------- log / exp round trip -- */

section("rotation vector round trip");
{
  const v = new THREE.Vector3();
  const q = new THREE.Quaternion();
  const back = new THREE.Vector3();
  let worst = 0;

  // Sweep the whole usable range, including near-identity and near-180°, which
  // are the two places the formulation can go singular.
  for (let i = 0; i < 400; i += 1) {
    const angle = (i / 400) * Math.PI * 1.98 + 1e-6;
    const axis = new THREE.Vector3(
      Math.sin(i * 1.7),
      Math.cos(i * 0.9),
      Math.sin(i * 2.3 + 1),
    ).normalize();
    q.setFromAxisAngle(axis, angle);
    quatToRotationVector(q, v);
    rotationVectorToQuat(v, new THREE.Quaternion()); // exercise allocation-free path
    rotationVectorToQuat(v, q.clone());
    const round = rotationVectorToQuat(v, new THREE.Quaternion());
    const original = new THREE.Quaternion().setFromAxisAngle(axis, angle);
    worst = Math.max(worst, quatAngle(round, original));
    back.copy(v);
  }
  ok("exp(log(q)) == q over 0..2π", worst < 1e-6, `worst ${worst.toExponential(2)}`);
}

section("shorter arc");
{
  // q and -q are the same rotation. Their logs must agree, or a spring driven by
  // the error will occasionally take the long way round — a full-speed flip.
  const q = new THREE.Quaternion().setFromAxisAngle(AXIS_Y, 0.4);
  const negated = new THREE.Quaternion(-q.x, -q.y, -q.z, -q.w);
  const a = quatToRotationVector(q, new THREE.Vector3());
  const b = quatToRotationVector(negated, new THREE.Vector3());
  close("log(q) == log(-q).x", a.x, b.x, 1e-9);
  close("log(q) == log(-q).y", a.y, b.y, 1e-9);
  close("log(q) == log(-q).z", a.z, b.z, 1e-9);
  ok("log magnitude is the short arc", a.length() <= Math.PI + 1e-9);
}

/* ------------------------------------------------------------ swing/twist -- */

section("swing / twist decomposition");
{
  const st = makeSwingTwist();
  const composed = new THREE.Quaternion();
  let worstRecompose = 0;
  let worstTwist = 0;

  for (let i = 0; i < 300; i += 1) {
    const twistAmount = ((i / 300) * 2 - 1) * 2.5;
    const swingAmount = ((i * 7) % 100) / 100 * 1.4;
    const swingAxis = new THREE.Vector3(Math.cos(i), 0, Math.sin(i)).normalize();

    const twist = new THREE.Quaternion().setFromAxisAngle(AXIS_Y, twistAmount);
    const swing = new THREE.Quaternion().setFromAxisAngle(swingAxis, swingAmount);
    const q = swing.clone().multiply(twist);

    swingTwist(q, AXIS_Y, st);
    composed.copy(st.swing).multiply(st.twist);
    worstRecompose = Math.max(worstRecompose, quatAngle(composed, q));

    // The recovered twist must be about the axis and nothing else.
    const recovered = twistAngle(st.twist, AXIS_Y);
    const expected = Math.atan2(
      Math.sin(twistAmount / 2),
      Math.cos(twistAmount / 2),
    ) * 2;
    worstTwist = Math.max(worstTwist, Math.abs(recovered - expected));
  }
  ok("q == swing ⋅ twist", worstRecompose < 1e-6, `worst ${worstRecompose.toExponential(2)}`);
  ok("twist angle recovered", worstTwist < 1e-6, `worst ${worstTwist.toExponential(2)}`);

  // A pure swing must produce zero twist, and vice versa. This is the property
  // that makes a palm-facing change independent of a finger bend.
  const pureSwing = new THREE.Quaternion().setFromAxisAngle(AXIS_X, 0.8);
  swingTwist(pureSwing, AXIS_Y, st);
  close("pure swing has no twist", twistAngle(st.twist, AXIS_Y), 0, 1e-6);

  const pureTwist = new THREE.Quaternion().setFromAxisAngle(AXIS_Y, 0.8);
  swingTwist(pureTwist, AXIS_Y, st);
  close("pure twist has no swing", quatAngle(st.swing, new THREE.Quaternion()), 0, 1e-6);
}

/* ---------------------------------------------------------------- clamps -- */

section("joint limits");
{
  // A PIP joint: hinge only, 0..110° of flexion. Everything else is noise.
  const hingeX: [number, number] = [0, 110 * DEG];
  const tiny: [number, number] = [-3 * DEG, 3 * DEG];
  const st = makeSwingTwist();

  // Hyperextension must be refused. This is the check that stops a finger
  // bending backwards through its own knuckle.
  const back = new THREE.Quaternion().setFromAxisAngle(AXIS_X, -50 * DEG);
  clampSwingTwist(back, AXIS_Y, hingeX, tiny, tiny);
  swingTwist(back, AXIS_Y, st);
  const backVector = quatToRotationVector(st.swing, new THREE.Vector3());
  ok("PIP refuses hyperextension", backVector.x >= -1e-6, `x = ${(backVector.x / DEG).toFixed(2)}°`);

  // Over-flexion clamps to the limit, not past it.
  const over = new THREE.Quaternion().setFromAxisAngle(AXIS_X, 150 * DEG);
  clampSwingTwist(over, AXIS_Y, hingeX, tiny, tiny);
  swingTwist(over, AXIS_Y, st);
  const overVector = quatToRotationVector(st.swing, new THREE.Vector3());
  close("PIP clamps to 110°", overVector.x / DEG, 110, 0.01);

  // Sideways splay on a hinge is squashed to the play allowance.
  const splay = new THREE.Quaternion().setFromAxisAngle(AXIS_Z, 40 * DEG);
  clampSwingTwist(splay, AXIS_Y, hingeX, tiny, tiny);
  swingTwist(splay, AXIS_Y, st);
  const splayVector = quatToRotationVector(st.swing, new THREE.Vector3());
  ok(
    "PIP refuses sideways splay",
    Math.abs(splayVector.z) <= 3 * DEG + 1e-6,
    `z = ${(splayVector.z / DEG).toFixed(2)}°`,
  );

  // Inside the limits, nothing moves. A clamp that perturbs a legal pose would
  // add a permanent bias to every finger.
  const legal = new THREE.Quaternion().setFromAxisAngle(AXIS_X, 45 * DEG);
  const before = legal.clone();
  clampSwingTwist(legal, AXIS_Y, hingeX, tiny, tiny);
  close("legal poses pass through untouched", quatAngle(legal, before), 0, 1e-6);

  // Idempotence: clamping twice equals clamping once. Without this the rig
  // would drift a little further into its limits every frame it sat on one.
  const twice = new THREE.Quaternion().setFromAxisAngle(AXIS_X, 150 * DEG);
  clampSwingTwist(twice, AXIS_Y, hingeX, tiny, tiny);
  const once = twice.clone();
  clampSwingTwist(twice, AXIS_Y, hingeX, tiny, tiny);
  close("clamp is idempotent", quatAngle(twice, once), 0, 1e-9);
}

/* --------------------------------------------------------------- springs -- */

section("damped quaternion");
{
  const spring = new DampedQuaternion();
  const target = new THREE.Quaternion().setFromAxisAngle(AXIS_Y, 1.2);
  spring.snap(new THREE.Quaternion());

  // Converges.
  for (let i = 0; i < 400; i += 1) spring.step(target, 0.08, 1 / 60);
  close("converges to the target", quatAngle(spring.value, target), 0, 1e-3);

  // Never overshoots. Critically damped means monotone approach; an overshoot
  // here would be a joint that visibly bounces past its pose and comes back.
  const spring2 = new DampedQuaternion();
  spring2.snap(new THREE.Quaternion());
  let previous = quatAngle(spring2.value, target);
  let overshoots = 0;
  for (let i = 0; i < 240; i += 1) {
    spring2.step(target, 0.1, 1 / 60);
    const now = quatAngle(spring2.value, target);
    if (now > previous + 1e-9) overshoots += 1;
    previous = now;
  }
  ok("critically damped — no overshoot", overshoots === 0, `${overshoots} frames grew`);

  // Continuity under a step input — the formal version of "never snap".
  //
  // The meaningful property is NOT a fixed per-frame ceiling: a 172° step with
  // a 90 ms smooth time is *supposed* to move fast on the first frame, and
  // capping that would just be a slower spring. What must hold is that the
  // motion is smooth — each frame's step is no larger than the last, so the
  // joint decelerates into its pose and never has a discontinuity in velocity —
  // and that the move is spread over enough frames to be seen as a move.
  const spring3 = new DampedQuaternion();
  spring3.snap(new THREE.Quaternion());
  const far = new THREE.Quaternion().setFromAxisAngle(AXIS_Z, 3.0);
  const steps: number[] = [];
  let last = spring3.value.clone();
  for (let i = 0; i < 240; i += 1) {
    spring3.step(far, 0.09, 1 / 60);
    steps.push(quatAngle(spring3.value, last));
    last = spring3.value.clone();
  }
  // The step profile must be UNIMODAL: accelerate once, decelerate once, and
  // never speed up again. A spring starting from rest genuinely does accelerate
  // for the first few frames — that ease-in is precisely what stops the motion
  // reading as robotic — so the property to assert is that there is exactly one
  // peak, not that the sequence only ever falls. A second peak would mean the
  // joint surged, slowed, and surged again: visible as a stutter.
  let peak = 0;
  for (let i = 1; i < steps.length; i += 1) if (steps[i] > steps[peak]) peak = i;
  let risesAfterPeak = 0;
  for (let i = peak + 1; i < steps.length; i += 1) {
    if (steps[i] > steps[i - 1] + 1e-9) risesAfterPeak += 1;
  }
  ok("step profile is unimodal", risesAfterPeak === 0, `${risesAfterPeak} late surges`);
  ok("acceleration phase is brief", peak <= 8, `peak at frame ${peak}`);

  const framesToArrive = steps.findIndex((_, i) =>
    quatAngle(new THREE.Quaternion(), far) * 0.02 > steps[i],
  );
  ok(
    "a 172° step is spread over several frames",
    framesToArrive > 6,
    `settled after ${framesToArrive} frames`,
  );

  // And under a REALISTIC input — a target moving at a fast signing speed — the
  // per-frame step must stay small, because that is the regime the rig actually
  // runs in. 6 rad/s is roughly a brisk wrist flick.
  const spring5 = new DampedQuaternion();
  spring5.snap(new THREE.Quaternion());
  const moving = new THREE.Quaternion();
  let worstLive = 0;
  let previousLive = spring5.value.clone();
  for (let i = 0; i < 180; i += 1) {
    moving.setFromAxisAngle(AXIS_Z, Math.sin((i / 60) * 6) * 1.2);
    spring5.step(moving, 0.06, 1 / 60);
    worstLive = Math.max(worstLive, quatAngle(spring5.value, previousLive));
    previousLive = spring5.value.clone();
  }
  ok(
    "tracks fast signing without jumping",
    worstLive < 12 * DEG,
    `worst ${(worstLive / DEG).toFixed(2)}° in one frame`,
  );

  // Stability at a large timestep. A dropped frame must not blow the spring up.
  const spring4 = new DampedQuaternion();
  spring4.snap(new THREE.Quaternion());
  for (let i = 0; i < 50; i += 1) spring4.step(far, 0.05, 0.25);
  ok(
    "stable at dt = 250 ms",
    Number.isFinite(spring4.value.x) && quatAngle(spring4.value, far) < 0.05,
    `error ${quatAngle(spring4.value, far).toFixed(4)}`,
  );
}

section("coasting");
{
  // A vector coasting under its own velocity must keep going and then stop —
  // this is what hides a short occlusion instead of freezing the hand.
  const v = new DampedVector3();
  v.snap(new THREE.Vector3(0, 0, 0));
  const moving = new THREE.Vector3();
  for (let i = 0; i < 30; i += 1) {
    moving.x += 0.02;
    v.step(moving, 0.06, 1 / 60);
  }
  const atLoss = v.value.clone();
  for (let i = 0; i < 6; i += 1) v.coast(1 / 60, 0.02);
  ok("coasting keeps moving", v.value.x > atLoss.x, `${atLoss.x.toFixed(4)} → ${v.value.x.toFixed(4)}`);

  // It must also come to a stop, and the total distance travelled while blind
  // must be bounded. An unbounded coast is worse than a freeze: the hand sails
  // off across the signing space and then snaps back when tracking returns.
  const start = v.value.clone();
  let previousStep = Infinity;
  let accelerated = 0;
  for (let i = 0; i < 120; i += 1) {
    const before = v.value.clone();
    v.coast(1 / 60, 0.02);
    const step = v.value.distanceTo(before);
    if (step > previousStep + 1e-12) accelerated += 1;
    previousStep = step;
  }
  ok("coasting decelerates monotonically", accelerated === 0);
  ok(
    "coasting comes to rest",
    previousStep < 1e-4,
    `final step ${previousStep.toExponential(2)} per frame`,
  );
  // The signer's whole reach is about 2 shoulder widths; a blind coast must
  // never travel a meaningful fraction of that.
  ok(
    "total coast travel is bounded",
    v.value.distanceTo(start) < 0.35,
    `travelled ${v.value.distanceTo(start).toFixed(4)} units`,
  );
}

/* ----------------------------------------------------------------- bases -- */

section("basis and rotation-between");
{
  const q = new THREE.Quaternion();
  const forward = new THREE.Vector3(0, 0, 1);
  const up = new THREE.Vector3(0, 1, 0);
  basisRotation(forward, up, q);
  close("identity basis", quatAngle(q, new THREE.Quaternion()), 0, 1e-6);

  // A non-orthogonal `up` must still produce an orthonormal frame — the palm
  // normal and the wrist→knuckle direction are never exactly perpendicular in
  // real landmark data.
  basisRotation(new THREE.Vector3(0, 0, 1), new THREE.Vector3(0.3, 1, 0.7), q);
  const m = new THREE.Matrix4().makeRotationFromQuaternion(q);
  const e = m.elements;
  const col = (i: number) => new THREE.Vector3(e[i * 4], e[i * 4 + 1], e[i * 4 + 2]);
  close("basis x is unit", col(0).length(), 1, 1e-6);
  close("basis y is unit", col(1).length(), 1, 1e-6);
  close("basis x ⟂ y", col(0).dot(col(1)), 0, 1e-6);
  close("basis y ⟂ z", col(1).dot(col(2)), 0, 1e-6);

  // Antiparallel vectors: the singular case. Must be stable and must respect the
  // hint, or a hand pointing straight down its own bone axis flickers.
  const from = new THREE.Vector3(0, 1, 0);
  const to = new THREE.Vector3(0, -1, 0);
  rotationBetween(from, to, new THREE.Vector3(1, 0, 0), q);
  const moved = from.clone().applyQuaternion(q);
  close("antiparallel rotation lands on target", moved.distanceTo(to), 0, 1e-6);

  const q2 = new THREE.Quaternion();
  rotationBetween(from, to, new THREE.Vector3(1, 0, 0), q2);
  close("antiparallel is deterministic", quatAngle(q, q2), 0, 1e-12);
}

/* ----------------------------------------------------------------- report -- */

console.log(
  `\n${failures === 0 ? "PASS" : "FAIL"}  ${checks - failures}/${checks} checks`,
);
if (failures > 0) process.exit(1);

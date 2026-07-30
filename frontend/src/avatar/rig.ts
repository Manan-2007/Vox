/**
 * The signing figure: two anatomically-proportioned hands on an upper body,
 * built in three.js and posed from the motion format in signMotion.ts.
 *
 * ---------------------------------------------------------------------------
 * HOW A HAND IS BUILT
 * ---------------------------------------------------------------------------
 * Not from image coordinates. MediaPipe's normalized landmarks have a z that is
 * a weak per-point guess, so a hand assembled from them is a flat constellation
 * no amount of lighting will rescue. The shape here comes from
 * `hand_world_landmarks` — 21 points in metres with real depth and real
 * proportions — and only the *placement* of the wrist comes from image space.
 *
 * Geometry per hand: a palm solid oriented to the palm plane, fifteen tapered
 * finger bones, rounded joints so bends read as knuckles rather than hinges,
 * fingernails, and a wrist that continues into the forearm. Every part is
 * re-posed each frame; nothing is baked.
 *
 * ---------------------------------------------------------------------------
 * SCALE AND SPACE
 * ---------------------------------------------------------------------------
 * One scene unit = the signer's shoulder width. That is the same anchor
 * ml/normalize.py uses, so the avatar and the recogniser agree about where a
 * sign happens. Converting the hands' metric coordinates into that space needs
 * one anatomical assumption — SHOULDER_WIDTH_M below — because MediaPipe gives
 * no absolute body scale. Getting it wrong makes the hands too big or too small
 * relative to the body, and nothing else.
 */

import * as THREE from "three";
import {
  LEFT_WORLD,
  LEFT_WRIST,
  POSE_BLOCK,
  POSE_L_ELBOW,
  POSE_L_SHOULDER,
  POSE_NOSE,
  POSE_R_ELBOW,
  POSE_R_SHOULDER,
  RIGHT_WORLD,
  RIGHT_WRIST,
} from "./signMotion";

/** Adult shoulder width, used to convert metric hand coordinates to scene units. */
const SHOULDER_WIDTH_M = 0.40;
const METRES_TO_UNITS = 1 / SHOULDER_WIDTH_M;

/** MediaPipe hand topology: [from, to, radius at from, radius at to]. */
const FINGER_BONES: [number, number, number, number][] = [
  [1, 2, 0.105, 0.092], [2, 3, 0.092, 0.082], [3, 4, 0.082, 0.066],   // thumb
  [5, 6, 0.084, 0.074], [6, 7, 0.074, 0.064], [7, 8, 0.064, 0.052],   // index
  [9, 10, 0.086, 0.076], [10, 11, 0.076, 0.066], [11, 12, 0.066, 0.053], // middle
  [13, 14, 0.081, 0.071], [14, 15, 0.071, 0.062], [15, 16, 0.062, 0.050], // ring
  [17, 18, 0.071, 0.062], [18, 19, 0.062, 0.055], [19, 20, 0.055, 0.045], // pinky
];
const FINGER_TIPS = [4, 8, 12, 16, 20];
/** Landmarks that get a rounded cap. The tips are handled with nails instead. */
const KNUCKLES = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19];

const WRIST = 0;
const INDEX_MCP = 5;
const MIDDLE_MCP = 9;
const PINKY_MCP = 17;

/* Palette. Warm porcelain skin, deep sage garment — the figure should look like
   a considered object, not a debug view. */
const SKIN = 0xe3bfa2;
const SKIN_DEEP = 0xd0a88c;
const NAIL = 0xf2dcc9;
const GARMENT = 0x2f4a44;
const GARMENT_LIGHT = 0x3b5b54;

/** Radii as a fraction of the hand's own wrist-to-middle-knuckle length. */
const PALM_THICKNESS = 0.30;
const JOINT_SCALE = 1.02;

export interface RigOptions {
  /** Show the torso and head. Off for a hands-only close-up. */
  body?: boolean;
}

/**
 * A posable signing figure. Construct once, call `apply` per frame, `dispose`
 * when the canvas goes away.
 */
export class SignRig {
  readonly root: THREE.Group;

  private readonly hands: HandParts[];
  private readonly torso: THREE.Mesh;
  private readonly neck: THREE.Mesh;
  private readonly head: THREE.Mesh;
  private readonly shoulderCaps: THREE.Mesh[];
  private readonly elbowCaps: THREE.Mesh[];
  private readonly upperArms: THREE.Mesh[];
  private readonly foreArms: THREE.Mesh[];
  private readonly showBody: boolean;
  private readonly disposables: { dispose(): void }[] = [];

  /* Scratch vectors — allocating inside the render loop would churn the GC. */
  private readonly a = new THREE.Vector3();
  private readonly b = new THREE.Vector3();
  private readonly c = new THREE.Vector3();
  private readonly mid = new THREE.Vector3();
  private readonly dir = new THREE.Vector3();
  private readonly up = new THREE.Vector3(0, 1, 0);
  private readonly quat = new THREE.Quaternion();
  private readonly points: THREE.Vector3[] = Array.from(
    { length: 21 },
    () => new THREE.Vector3(),
  );
  private readonly wrists: (THREE.Vector3 | null)[] = [null, null];

  constructor(options: RigOptions = {}) {
    this.showBody = options.body !== false;
    this.root = new THREE.Group();

    // Bones are cylinders, not capsules. A capsule's hemispherical caps are part
    // of its geometry, so scaling y by the bone length and x/z by the radius
    // stretches the caps into ellipsoids — visible as a pinched, melted look at
    // every joint. A cylinder scales exactly, and the rounded joint spheres
    // placed at every landmark supply the caps anyway.
    const shared = {
      bone: new THREE.CylinderGeometry(1, 1, 1, 14, 1, false),
      joint: new THREE.SphereGeometry(1, 16, 12),
      nail: new THREE.SphereGeometry(1, 10, 8),
      palm: new THREE.BoxGeometry(1, 1, 1, 2, 2, 2),
      limb: new THREE.CylinderGeometry(1, 1, 1, 16, 1, false),
    };
    Object.values(shared).forEach((g) => this.disposables.push(g));

    const skin = this.material(SKIN, 0.52, 0.22);
    const skinDeep = this.material(SKIN_DEEP, 0.58, 0.12);
    const nail = this.material(NAIL, 0.28, 0.45);
    const garment = this.material(GARMENT, 0.78, 0.02);
    const garmentLight = this.material(GARMENT_LIGHT, 0.72, 0.04);

    this.hands = [
      this.buildHand(shared, skin, skinDeep, nail),
      this.buildHand(shared, skin, skinDeep, nail),
    ];

    /* --- body. Proportions are in shoulder-width units. --------------------- */
    // The torso is a revolved profile, not a box. A box reads as a slab behind
    // the hands and flattens the whole figure; a lathe gives shoulders that
    // round off, a waist that narrows, and a silhouette that catches the key
    // light the way a body does. Squashed in z afterwards, because a chest is
    // much wider than it is deep.
    // The profile is written directly in scene units (1 unit = shoulder width),
    // measured down from the shoulder line at y = 0, so the torso needs no
    // vertical scaling and its proportions cannot drift. It continues well below
    // the framed area on purpose: the camera frames chest-up, the way sign
    // reference footage is framed, and a torso that stopped at the frame edge
    // would read as a bust floating in the dark.
    const torsoProfile = [
      new THREE.Vector2(0.06, 0.22),   // base of the neck
      new THREE.Vector2(0.44, 0.16),
      new THREE.Vector2(0.52, 0.04),
      new THREE.Vector2(0.51, -0.30),
      new THREE.Vector2(0.44, -0.75),  // waist
      new THREE.Vector2(0.41, -1.10),
      new THREE.Vector2(0.45, -1.45),  // hips
      new THREE.Vector2(0.43, -1.72),
      new THREE.Vector2(0.05, -1.78),
    ];
    const torsoGeo = new THREE.LatheGeometry(torsoProfile, 28);
    this.disposables.push(torsoGeo);
    this.torso = new THREE.Mesh(torsoGeo, garment);
    this.torso.castShadow = true;
    this.torso.receiveShadow = true;
    this.neck = new THREE.Mesh(shared.limb, skinDeep);
    this.head = new THREE.Mesh(shared.joint, skin);
    this.head.castShadow = true;
    this.shoulderCaps = [0, 1].map(() => {
      const mesh = new THREE.Mesh(shared.joint, garmentLight);
      mesh.castShadow = true;
      return mesh;
    });
    this.elbowCaps = [0, 1].map(() => {
      const mesh = new THREE.Mesh(shared.joint, skinDeep);
      mesh.castShadow = true;
      return mesh;
    });
    this.upperArms = [0, 1].map(() => {
      const mesh = new THREE.Mesh(shared.limb, garmentLight);
      mesh.castShadow = true;
      return mesh;
    });
    this.foreArms = [0, 1].map(() => {
      const mesh = new THREE.Mesh(shared.limb, skinDeep);
      mesh.castShadow = true;
      return mesh;
    });

    for (const mesh of [
      this.torso, this.neck, this.head,
      ...this.shoulderCaps, ...this.elbowCaps,
      ...this.upperArms, ...this.foreArms,
    ]) {
      mesh.visible = false;
      this.root.add(mesh);
    }
  }

  private material(colour: number, roughness: number, sheen: number) {
    const material = new THREE.MeshPhysicalMaterial({
      color: colour,
      roughness,
      metalness: 0,
      // A little sheen and clearcoat is what stops skin reading as plastic.
      sheen,
      sheenColor: new THREE.Color(0xffd9c0),
      sheenRoughness: 0.6,
      clearcoat: 0.12,
      clearcoatRoughness: 0.5,
    });
    this.disposables.push(material);
    return material;
  }

  private buildHand(
    shared: Record<string, THREE.BufferGeometry>,
    skin: THREE.Material,
    skinDeep: THREE.Material,
    nailMaterial: THREE.Material,
  ): HandParts {
    const group = new THREE.Group();
    group.visible = false;
    this.root.add(group);

    const bones = FINGER_BONES.map(([from, to, r0, r1]) => {
      const mesh = new THREE.Mesh(shared.bone, skin);
      mesh.castShadow = true;
      group.add(mesh);
      return { mesh, from, to, radius: (r0 + r1) / 2 };
    });

    const joints = [
      ...KNUCKLES.map((index) => ({
        index,
        radius: FINGER_BONES.find((bone) => bone[0] === index)?.[2] ?? 0.07,
      })),
      { index: WRIST, radius: 0.135 },
    ].map(({ index, radius }) => {
      const mesh = new THREE.Mesh(shared.joint, skin);
      mesh.castShadow = true;
      group.add(mesh);
      return { mesh, index, radius: radius * JOINT_SCALE };
    });

    const tips = FINGER_TIPS.map((index) => {
      const mesh = new THREE.Mesh(shared.joint, skin);
      mesh.castShadow = true;
      group.add(mesh);
      const nail = new THREE.Mesh(shared.nail, nailMaterial);
      group.add(nail);
      const radius =
        FINGER_BONES.find((bone) => bone[1] === index)?.[3] ?? 0.05;
      return { mesh, nail, index, radius };
    });

    const palm = new THREE.Mesh(shared.palm, skin);
    palm.castShadow = true;
    palm.receiveShadow = true;
    group.add(palm);

    const cuff = new THREE.Mesh(shared.limb, skinDeep);
    cuff.castShadow = true;
    group.add(cuff);

    return { group, bones, joints, tips, palm, cuff };
  }

  /**
   * Pose the figure from one 140-float frame. Returns false when the frame has
   * no usable body anchor, in which case nothing is drawn.
   */
  apply(frame: Float32Array): boolean {
    const pose = POSE_BLOCK;
    const lsx = frame[pose + POSE_L_SHOULDER * 2];
    const lsy = frame[pose + POSE_L_SHOULDER * 2 + 1];
    const rsx = frame[pose + POSE_R_SHOULDER * 2];
    const rsy = frame[pose + POSE_R_SHOULDER * 2 + 1];
    const width = Math.hypot(lsx - rsx, lsy - rsy);

    if (width < 1e-4) {
      this.hide();
      return false;
    }

    // Image space -> scene: shoulder midpoint at the origin, shoulder width = 1,
    // y flipped because image y grows downward.
    const cx = (lsx + rsx) / 2;
    const cy = (lsy + rsy) / 2;
    const k = 1 / width;
    const place = (target: THREE.Vector3, x: number, y: number) =>
      target.set((x - cx) * k, -(y - cy) * k, 0);

    /* ------------------------------------------------------------- hands --- */
    for (let block = 0; block < 2; block += 1) {
      const wristSlot = block === 0 ? LEFT_WRIST : RIGHT_WRIST;
      const worldSlot = block === 0 ? LEFT_WORLD : RIGHT_WORLD;
      const parts = this.hands[block];
      const present = frame[wristSlot] !== 0 || frame[wristSlot + 1] !== 0;
      parts.group.visible = present;
      if (!present) {
        this.wrists[block] = null;
        continue;
      }

      place(this.a, frame[wristSlot], frame[wristSlot + 1]);
      this.wrists[block] = this.points[WRIST].copy(this.a);

      // World landmarks are relative to the hand's own centre; re-root them at
      // the wrist and drop them into the scene at the wrist's placed position.
      // MediaPipe's world axes are x right, y down, z toward the camera, so y
      // and z are negated to match three.js.
      const wx = frame[worldSlot];
      const wy = frame[worldSlot + 1];
      const wz = frame[worldSlot + 2];
      for (let i = 0; i < 21; i += 1) {
        const base = worldSlot + i * 3;
        this.points[i].set(
          this.a.x + (frame[base] - wx) * METRES_TO_UNITS,
          this.a.y - (frame[base + 1] - wy) * METRES_TO_UNITS,
          this.a.z - (frame[base + 2] - wz) * METRES_TO_UNITS,
        );
      }

      // The hand's own ruler, so thickness stays proportional at any distance.
      const unit = Math.max(
        1e-4,
        this.points[WRIST].distanceTo(this.points[MIDDLE_MCP]),
      );

      for (const bone of parts.bones) {
        this.orient(
          bone.mesh,
          this.points[bone.from],
          this.points[bone.to],
          unit * bone.radius,
        );
      }
      for (const joint of parts.joints) {
        joint.mesh.position.copy(this.points[joint.index]);
        joint.mesh.scale.setScalar(unit * joint.radius);
      }
      for (const tip of parts.tips) {
        const point = this.points[tip.index];
        tip.mesh.position.copy(point);
        tip.mesh.scale.setScalar(unit * tip.radius);
        // The nail sits on the back of the last phalanx, offset along the bone.
        const previous = this.points[tip.index - 1];
        this.dir.subVectors(point, previous).normalize();
        tip.nail.position
          .copy(point)
          .addScaledVector(this.dir, unit * tip.radius * 0.25);
        tip.nail.scale.set(
          unit * tip.radius * 0.66,
          unit * tip.radius * 0.28,
          unit * tip.radius * 0.86,
        );
        tip.nail.quaternion.setFromUnitVectors(this.up, this.dir);
      }

      this.posePalm(parts.palm, unit);
      // Wrist continues toward the forearm: away from the middle knuckle.
      this.dir.subVectors(this.points[WRIST], this.points[MIDDLE_MCP]).normalize();
      this.b.copy(this.points[WRIST]).addScaledVector(this.dir, unit * 0.5);
      this.orient(parts.cuff, this.points[WRIST], this.b, unit * 0.145);
    }

    /* -------------------------------------------------------------- body --- */
    if (!this.showBody) {
      for (const mesh of [
        this.torso, this.neck, this.head,
        ...this.shoulderCaps, ...this.elbowCaps,
      ...this.upperArms, ...this.foreArms,
      ]) {
        mesh.visible = false;
      }
      return true;
    }

    place(this.a, lsx, lsy);
    place(this.b, rsx, rsy);
    const shoulderL = this.a.clone();
    const shoulderR = this.b.clone();
    const shoulderMid = this.mid.copy(shoulderL).lerp(shoulderR, 0.5).clone();

    // Torso: the lathe profile spans y = -0.38..1.0 in its own units, so it is
    // scaled to sit with its shoulder line on the tracked shoulders and run down
    // to roughly the waist. The hips are not tracked, so the length below the
    // shoulders is anatomical rather than measured, and it stops at the waist
    // instead of guessing at legs.
    this.torso.visible = true;
    this.torso.position.set(shoulderMid.x, shoulderMid.y, -0.04);
    // Only z is scaled: a chest is much wider than it is deep, and the profile
    // already carries the true width and height.
    this.torso.scale.set(1, 1, 0.54);
    this.torso.quaternion.identity();

    this.shoulderCaps[0].visible = true;
    this.shoulderCaps[0].position.copy(shoulderL);
    this.shoulderCaps[0].scale.setScalar(0.14);
    this.shoulderCaps[1].visible = true;
    this.shoulderCaps[1].position.copy(shoulderR);
    this.shoulderCaps[1].scale.setScalar(0.14);

    // Head first, then a neck long enough to actually reach it.
    //
    // The head is placed from the tracked nose, so it turns and tilts with the
    // real signer — which matters, because head position is part of how a sign
    // is aimed. But the nose moves and the shoulders move independently, so a
    // fixed-length neck leaves a gap between them whenever the signer leans.
    // The neck is therefore drawn to wherever the head actually ended up.
    const noseX = frame[pose + POSE_NOSE * 2];
    const noseY = frame[pose + POSE_NOSE * 2 + 1];
    const hasNose = noseX !== 0 || noseY !== 0;

    // Head radii in shoulder widths: a head is a little over half a shoulder
    // span tall and slightly narrower than it is tall.
    const headRadiusY = 0.30;
    this.head.visible = true;
    if (hasNose) {
      place(this.c, noseX, noseY);
      // The nose is on the front of the face; the skull's centre is behind it
      // and a little above.
      this.head.position.set(this.c.x, this.c.y + 0.09, this.c.z - 0.10);
    } else {
      this.head.position.set(shoulderMid.x, shoulderMid.y + 0.62, -0.05);
    }
    this.head.scale.set(0.25, headRadiusY, 0.27);

    this.neck.visible = true;
    // Run from just inside the torso's collar to just inside the skull, so
    // neither end shows a seam.
    this.a.copy(shoulderMid).setY(shoulderMid.y + 0.04);
    this.b
      .copy(this.head.position)
      .setY(this.head.position.y - headRadiusY * 0.72);
    this.orient(this.neck, this.a, this.b, 0.115);

    /* Arms: shoulder -> elbow -> wrist. Only the shoulders and elbows are
       tracked, so the arm is straight between them; the forearm ends at the
       hand's own wrist when that hand is visible, and at the elbow otherwise. */
    const elbows: [number, THREE.Vector3][] = [
      [POSE_L_ELBOW, shoulderL],
      [POSE_R_ELBOW, shoulderR],
    ];
    elbows.forEach(([elbowIndex, shoulder], side) => {
      const ex = frame[pose + elbowIndex * 2];
      const ey = frame[pose + elbowIndex * 2 + 1];
      if (ex === 0 && ey === 0) {
        this.upperArms[side].visible = false;
        this.foreArms[side].visible = false;
        this.elbowCaps[side].visible = false;
        return;
      }
      place(this.c, ex, ey);
      this.orient(this.upperArms[side], shoulder, this.c, 0.115);
      this.elbowCaps[side].visible = true;
      this.elbowCaps[side].position.copy(this.c);
      this.elbowCaps[side].scale.setScalar(0.105);

      const wrist = this.wrists[side];
      if (wrist) {
        this.orient(this.foreArms[side], this.c, wrist, 0.095);
      } else {
        this.foreArms[side].visible = false;
      }
    });

    return true;
  }

  /** Orient the palm box to the plane through wrist, index MCP and pinky MCP. */
  private posePalm(palm: THREE.Mesh, unit: number): void {
    const wrist = this.points[WRIST];
    const index = this.points[INDEX_MCP];
    const pinky = this.points[PINKY_MCP];
    const middle = this.points[MIDDLE_MCP];

    // Local frame: x across the knuckles, y from wrist to knuckles, z the normal.
    const across = this.a.subVectors(index, pinky);
    const along = this.b.subVectors(middle, wrist);
    const normal = this.c.copy(across).cross(along);
    if (normal.lengthSq() < 1e-12) {
      palm.visible = false;
      return;
    }
    normal.normalize();
    const yAxis = along.clone().normalize();
    const xAxis = new THREE.Vector3().crossVectors(yAxis, normal).normalize();

    const basis = new THREE.Matrix4().makeBasis(xAxis, yAxis, normal);
    palm.visible = true;
    palm.quaternion.setFromRotationMatrix(basis);
    palm.position
      .copy(wrist)
      .addScaledVector(yAxis, along.length() * 0.52)
      // Nudge the solid behind the knuckle line so fingers emerge from it.
      .addScaledVector(normal, -unit * PALM_THICKNESS * 0.12);
    palm.scale.set(
      Math.max(unit * 0.6, across.length() * 1.02),
      along.length() * 1.04,
      unit * PALM_THICKNESS,
    );
  }

  /** Place a capsule spanning a -> b with the given radius. */
  private orient(
    mesh: THREE.Mesh,
    from: THREE.Vector3,
    to: THREE.Vector3,
    radius: number,
  ): void {
    const length = from.distanceTo(to);
    if (length < 1e-6) {
      mesh.visible = false;
      return;
    }
    this.mid.addVectors(from, to).multiplyScalar(0.5);
    this.dir.subVectors(to, from).normalize();
    this.quat.setFromUnitVectors(this.up, this.dir);
    mesh.position.copy(this.mid);
    mesh.quaternion.copy(this.quat);
    // CapsuleGeometry(1, 1) is 1 radius + 1 length along Y, i.e. 3 tall. Scaling
    // x and z by the radius and y by length/(1 + 2*radius) would distort the
    // caps, so the capsule is built unit-radius and scaled uniformly in x/z with
    // the cylindrical section absorbing the length.
    // A unit cylinder is 1 tall and 1 in radius, centred at the origin, so this
    // scaling is exact: no distortion of any cap, because there is no cap.
    mesh.scale.set(radius, Math.max(1e-4, length), radius);
    mesh.visible = true;
  }

  hide(): void {
    for (const hand of this.hands) hand.group.visible = false;
    for (const mesh of [
      this.torso, this.neck, this.head,
      ...this.shoulderCaps, ...this.elbowCaps,
      ...this.upperArms, ...this.foreArms,
    ]) {
      mesh.visible = false;
    }
    this.wrists[0] = null;
    this.wrists[1] = null;
  }

  dispose(): void {
    for (const item of this.disposables) item.dispose();
  }
}

interface HandParts {
  group: THREE.Group;
  bones: { mesh: THREE.Mesh; from: number; to: number; radius: number }[];
  joints: { mesh: THREE.Mesh; index: number; radius: number }[];
  tips: { mesh: THREE.Mesh; nail: THREE.Mesh; index: number; radius: number }[];
  palm: THREE.Mesh;
  cuff: THREE.Mesh;
}

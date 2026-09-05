/**
 * The built-in signer: a real skinned mesh, generated at load.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS IS NOT THE OLD RIG WEARING A NEW NAME
 * ---------------------------------------------------------------------------
 * The build this replaced also drew a figure from code, and it was replaced for
 * good reasons. The difference is structural, not cosmetic:
 *
 *   OLD  ~120 unparented meshes, each repositioned every frame from landmark
 *        POSITIONS. A finger was three cylinders and three spheres, so a bend
 *        gapped on the outside of the curve and interpenetrated on the inside.
 *        Nothing could be limited, because there were no rotations to limit.
 *
 *   THIS one `THREE.SkinnedMesh` over one `THREE.Skeleton`, driven by bone
 *        ROTATIONS. The surface is continuous across every joint because it is
 *        a single mesh whose vertices are weighted to two bones and blended.
 *        A bent finger deforms; it does not come apart.
 *
 * It is the same skeleton, the same names, the same `captureRest`, and the same
 * retargeter as the VRM path — see `SigningAvatar`. So it is not a second
 * implementation of the rig; it is a second BODY for the one rig.
 *
 * ---------------------------------------------------------------------------
 * WHAT IT IS AND IS NOT FOR
 * ---------------------------------------------------------------------------
 * A modelled human will beat this on faces, hair, clothing and silhouette, and
 * that is fine — those are not what a sign is read from. What this has to get
 * right is the hands, and it spends its geometry accordingly: a hand here is
 * about a third of the figure's triangles, with a real palm mass, a thenar
 * eminence, tapered phalanges and rounded tips.
 *
 * ---------------------------------------------------------------------------
 * HOW SKINNING IS ASSIGNED
 * ---------------------------------------------------------------------------
 * Every vertex is generated as part of a specific bone's tube, at a known
 * parameter `t` from 0 (the joint at its start) to 1 (the joint at its end).
 * Weights come from that parameter rather than from distance:
 *
 *     t < BLEND      blend toward the PARENT bone
 *     t > 1 − BLEND  blend toward the CHILD bone
 *     otherwise      fully this bone
 *
 * Distance-based weighting is the more common trick and it is wrong here: the
 * fingers pass within a few millimetres of each other, so a distance falloff
 * bleeds the ring finger's weights onto the middle finger and the two deform
 * together. Parametric weighting cannot bleed across a gap, because it never
 * consults geometry at all.
 */

import * as THREE from "three";
import type { BoneName, HumanoidPose, Side } from "../core/humanoid";
import { fingerChains } from "../core/humanoid";
import { captureRest, type RestPose } from "./retarget/rest";
import {
  buildCanonicalSkeleton,
  type CanonicalSkeleton,
} from "./retarget/canonicalSkeleton";
import type { Presentation, SigningAvatar } from "./SigningAvatar";

/** How much of each bone's length is spent blending into its neighbour. */
const BLEND = 0.30;

/* ------------------------------------------------------------- palette --- */

const SKIN = 0xe7b48f;
const GARMENT = 0x2f4a52;
const HAIR = 0x2a211f;
const EYE = 0x1b1512;

/* -------------------------------------------------------------- builder --- */

interface Vertex {
  position: THREE.Vector3;
  normal: THREE.Vector3;
  bones: [number, number];
  weights: [number, number];
}

/**
 * Accumulates vertices and triangles, then bakes one BufferGeometry.
 *
 * Positions are generated in the skeleton's REST world space, which is also the
 * bind space — `THREE.Skeleton` inverts the bones' world matrices at
 * construction, so nothing here has to be expressed bone-locally.
 */
class MeshBuilder {
  private readonly vertices: Vertex[] = [];
  private readonly indices: number[] = [];
  private readonly boneIndex: Map<BoneName, number>;

  constructor(boneIndex: Map<BoneName, number>) {
    this.boneIndex = boneIndex;
  }

  get triangleCount(): number {
    return this.indices.length / 3;
  }

  private index(name: BoneName | null): number {
    if (!name) return 0;
    return this.boneIndex.get(name) ?? 0;
  }

  /**
   * A tapered tube along one bone, with its ends weighted into the neighbours.
   *
   * `sections` are [t, radius] pairs, so a caller can shape a limb — a forearm
   * that swells at the elbow and narrows at the wrist — rather than getting a
   * plain cone.
   */
  tube(
    from: THREE.Vector3,
    to: THREE.Vector3,
    sections: readonly (readonly [number, number])[],
    bone: BoneName,
    parent: BoneName | null,
    child: BoneName | null,
    radial = 10,
    options: { capStart?: boolean; capEnd?: boolean; squash?: number } = {},
  ): void {
    const axis = new THREE.Vector3().subVectors(to, from);
    const length = axis.length();
    if (length < 1e-6) return;
    axis.divideScalar(length);

    // A stable perpendicular frame. Choosing the reference axis by the smallest
    // component avoids the degenerate case where the bone is parallel to it.
    const reference =
      Math.abs(axis.y) < 0.9 ? UP : new THREE.Vector3(1, 0, 0);
    const side = new THREE.Vector3().crossVectors(reference, axis).normalize();
    const up = new THREE.Vector3().crossVectors(axis, side).normalize();

    const squash = options.squash ?? 1;
    const self = this.index(bone);
    const parentIndex = this.index(parent ?? bone);
    const childIndex = this.index(child ?? bone);

    const ringStart = this.vertices.length;

    for (const [t, radius] of sections) {
      const centre = new THREE.Vector3().copy(from).addScaledVector(axis, length * t);

      /* Weights from the parameter, never from distance — see the header. */
      let other = self;
      let mix = 0;
      if (t < BLEND && parent) {
        other = parentIndex;
        mix = 0.5 * (1 - t / BLEND);
      } else if (t > 1 - BLEND && child) {
        other = childIndex;
        mix = 0.5 * (1 - (1 - t) / BLEND);
      }

      for (let i = 0; i < radial; i += 1) {
        const theta = (i / radial) * Math.PI * 2;
        const c = Math.cos(theta);
        const s = Math.sin(theta);
        const offset = new THREE.Vector3()
          .addScaledVector(side, c * radius * squash)
          .addScaledVector(up, s * radius);
        this.vertices.push({
          position: centre.clone().add(offset),
          normal: offset.clone().normalize(),
          bones: [self, other],
          weights: [1 - mix, mix],
        });
      }
    }

    for (let s = 0; s < sections.length - 1; s += 1) {
      for (let i = 0; i < radial; i += 1) {
        const next = (i + 1) % radial;
        const a = ringStart + s * radial + i;
        const b = ringStart + s * radial + next;
        const c = ringStart + (s + 1) * radial + next;
        const d = ringStart + (s + 1) * radial + i;
        this.indices.push(a, b, c, a, c, d);
      }
    }

    if (options.capStart) this.cap(ringStart, radial, from, axis, self, true);
    if (options.capEnd) {
      const lastRing = ringStart + (sections.length - 1) * radial;
      const tip = new THREE.Vector3().copy(from).addScaledVector(axis, length);
      this.cap(lastRing, radial, tip, axis, child ? childIndex : self, false);
    }
  }

  /** Close a tube end with a fan to a centre vertex. */
  private cap(
    ringStart: number,
    radial: number,
    centre: THREE.Vector3,
    axis: THREE.Vector3,
    bone: number,
    reversed: boolean,
  ): void {
    const hub = this.vertices.length;
    this.vertices.push({
      position: centre.clone(),
      normal: axis.clone().multiplyScalar(reversed ? -1 : 1),
      bones: [bone, bone],
      weights: [1, 0],
    });
    for (let i = 0; i < radial; i += 1) {
      const next = (i + 1) % radial;
      if (reversed) this.indices.push(hub, ringStart + next, ringStart + i);
      else this.indices.push(hub, ringStart + i, ringStart + next);
    }
  }

  /** An ellipsoid rigidly bound to one bone — head, palm mass, eyes. */
  blob(
    centre: THREE.Vector3,
    radii: THREE.Vector3,
    bone: BoneName,
    segments = 14,
    rings = 10,
    shape?: (v: THREE.Vector3, u: number, phi: number) => void,
  ): void {
    const index = this.index(bone);
    const start = this.vertices.length;

    for (let r = 0; r <= rings; r += 1) {
      const phi = (r / rings) * Math.PI;
      for (let s = 0; s <= segments; s += 1) {
        const theta = (s / segments) * Math.PI * 2;
        const unit = new THREE.Vector3(
          Math.sin(phi) * Math.cos(theta),
          Math.cos(phi),
          Math.sin(phi) * Math.sin(theta),
        );
        const local = new THREE.Vector3(
          unit.x * radii.x,
          unit.y * radii.y,
          unit.z * radii.z,
        );
        shape?.(local, r / rings, theta);
        this.vertices.push({
          position: local.add(centre),
          normal: unit.clone(),
          bones: [index, index],
          weights: [1, 0],
        });
      }
    }

    const stride = segments + 1;
    for (let r = 0; r < rings; r += 1) {
      for (let s = 0; s < segments; s += 1) {
        const a = start + r * stride + s;
        const b = a + 1;
        const c = a + stride + 1;
        const d = a + stride;
        this.indices.push(a, b, c, a, c, d);
      }
    }
  }

  build(): THREE.BufferGeometry {
    const count = this.vertices.length;
    const positions = new Float32Array(count * 3);
    const normals = new Float32Array(count * 3);
    const skinIndices = new Uint16Array(count * 4);
    const skinWeights = new Float32Array(count * 4);

    this.vertices.forEach((vertex, i) => {
      positions[i * 3] = vertex.position.x;
      positions[i * 3 + 1] = vertex.position.y;
      positions[i * 3 + 2] = vertex.position.z;
      normals[i * 3] = vertex.normal.x;
      normals[i * 3 + 1] = vertex.normal.y;
      normals[i * 3 + 2] = vertex.normal.z;
      skinIndices[i * 4] = vertex.bones[0];
      skinIndices[i * 4 + 1] = vertex.bones[1];
      skinWeights[i * 4] = vertex.weights[0];
      skinWeights[i * 4 + 1] = vertex.weights[1];
    });

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
    geometry.setAttribute("skinIndex", new THREE.BufferAttribute(skinIndices, 4));
    geometry.setAttribute("skinWeight", new THREE.BufferAttribute(skinWeights, 4));
    geometry.setIndex(this.indices);
    // Recomputed rather than kept: the generated normals are per-primitive and
    // do not agree where two pieces meet, which shows as a hard seam at every
    // knuckle exactly where the eye is looking.
    geometry.computeVertexNormals();
    return geometry;
  }
}

const UP = new THREE.Vector3(0, 1, 0);

/* --------------------------------------------------------------- avatar --- */

export class ProceduralAvatar implements SigningAvatar {
  readonly root = new THREE.Group();
  readonly rest: RestPose;
  readonly kind = "Built-in";
  triangles = 0;

  private readonly skeleton: CanonicalSkeleton;
  private readonly meshes: THREE.SkinnedMesh[] = [];
  /** Always drawn — in both modes the hands are the message. */
  private figure: THREE.SkinnedMesh[] = [];
  private ghost!: THREE.SkinnedMesh;
  private presentation: Presentation = "figure";
  private readonly disposables: { dispose(): void }[] = [];
  private breathPhase = 0;

  constructor() {
    this.skeleton = buildCanonicalSkeleton();
    this.rest = captureRest(this.skeleton.lookup);

    const order: BoneName[] = [...this.skeleton.bones.keys()];
    const boneIndex = new Map<BoneName, number>();
    const bones: THREE.Bone[] = [];
    order.forEach((name, i) => {
      boneIndex.set(name, i);
      bones.push(this.skeleton.bones.get(name)!);
    });

    // The skeleton must be in its rest pose with world matrices current before
    // `THREE.Skeleton` captures the inverse bind matrices.
    this.skeleton.root.updateMatrixWorld(true);
    const armature = new THREE.Skeleton(bones);

    const skin = this.material(SKIN, 0.62);
    const garment = this.material(GARMENT, 0.85);
    const hair = this.material(HAIR, 0.72);
    const eye = this.material(EYE, 0.25);

    /* Built as separate meshes so the presentation can be switched without
       rebuilding anything: HANDS mode simply stops drawing the rest. */
    this.add(this.buildHands(boneIndex), skin, armature);
    this.figure = [
      this.add(this.buildBody(boneIndex), skin, armature),
      this.add(this.buildGarment(boneIndex), garment, armature),
      this.add(this.buildHair(boneIndex), hair, armature),
      this.add(this.buildEyes(boneIndex), eye, armature),
    ];

    /* The spatial reference for hands mode.
     *
     * Location is PHONEMIC in sign language: the same handshape at the forehead,
     * the chin and the chest are three different signs. Hands floating in an
     * empty void delete that distinction entirely, so hands-only cannot mean
     * nothing-but-hands.
     *
     * The answer is a ghost — head and shoulders at very low opacity, enough to
     * anchor where a sign is being made and not enough to be a character with a
     * face. It reads as a diagram rather than as a person, which is exactly the
     * clean look hands mode is for. */
    this.ghost = this.add(this.buildGhost(boneIndex), this.ghostMaterial(), armature);
    this.ghost.visible = false;
    this.ghost.castShadow = false;
    this.ghost.receiveShadow = false;

    this.root.add(this.skeleton.root);
  }

  private material(colour: number, roughness: number): THREE.MeshStandardMaterial {
    /* MeshStandardMaterial, not MeshPhysicalMaterial.
       The stage supplies a real environment map, so standard PBR already reads
       as skin. Physical's sheen and clearcoat are roughly twice the fragment
       cost and, at the size a hand occupies here, buy nothing a viewer can see. */
    const material = new THREE.MeshStandardMaterial({
      color: colour,
      roughness,
      metalness: 0,
    });
    this.disposables.push(material);
    return material;
  }

  private add(
    geometry: THREE.BufferGeometry,
    material: THREE.Material,
    armature: THREE.Skeleton,
  ): THREE.SkinnedMesh {
    const mesh = new THREE.SkinnedMesh(geometry, material);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    // The bind volume is computed from the rest pose, and a raised arm reaches
    // well outside it — with culling on, hands vanish at the most extreme point
    // of a sign, which is the moment they matter most.
    mesh.frustumCulled = false;
    mesh.add(this.skeleton.root);
    mesh.bind(armature);
    this.meshes.push(mesh);
    this.disposables.push(geometry);
    this.root.add(mesh);
    this.triangles += geometry.getIndex()!.count / 3;
    return mesh;
  }

  /**
   * Which parts of the body to draw.
   *
   *   "figure"  the whole signer
   *   "hands"   hands and forearms, over a faint head-and-shoulders ghost
   *
   * Hands mode is not a reduced version of the product — for a signing tool it
   * is arguably the better one. It removes the uncanny-valley problem completely
   * (there is no face to be wrong), it puts every pixel of the frame on the part
   * that carries the meaning, and it is what several sign-language teaching
   * systems already do. What it costs is non-manual grammar, which lives on a
   * face this mode does not draw — so a brow-raise question and a plain
   * statement become indistinguishable. That is a real loss, not a cosmetic one,
   * which is why "figure" remains available rather than being replaced.
   */
  setPresentation(mode: Presentation): void {
    if (mode === this.presentation) return;
    this.presentation = mode;
    const wholeFigure = mode === "figure";
    for (const mesh of this.figure) mesh.visible = wholeFigure;
    this.ghost.visible = !wholeFigure;
  }

  /* ------------------------------------------------------------ pieces -- */

  private world(name: BoneName): THREE.Vector3 {
    return this.skeleton.bones.get(name)!.getWorldPosition(new THREE.Vector3());
  }

  /** Every hand bone, as its own mesh so hands mode can draw it alone. */
  private buildHands(boneIndex: Map<BoneName, number>): THREE.BufferGeometry {
    const b = new MeshBuilder(boneIndex);
    for (const side of ["left", "right"] as const) {
      const S = side === "left" ? "left" : "right";
      const elbow = this.world(`${S}LowerArm` as BoneName);
      const wrist = this.world(`${S}Hand` as BoneName);
      /* The forearm comes with the hand. A wrist ending in nothing reads as a
         severed hand, and — more usefully — forearm ROTATION is how palm
         orientation is produced, so showing it makes the rotation legible
         instead of leaving the hand to appear to spin by itself. */
      b.tube(
        elbow, wrist,
        [[0.45, 0.032], [0.7, 0.029], [1, 0.025]],
        `${S}LowerArm` as BoneName, `${S}UpperArm` as BoneName, `${S}Hand` as BoneName,
        14,
        { capStart: true },
      );
      this.buildHand(b, side);
    }
    return b.build();
  }

  /** Head, neck and arms — everything skin-coloured that is not a hand. */
  private buildBody(boneIndex: Map<BoneName, number>): THREE.BufferGeometry {
    const b = new MeshBuilder(boneIndex);

    /* ---------------------------------------------------------- head --- */
    const head = this.world("head");
    const neck = this.world("neck");
    b.tube(neck, head, [[0, 0.043], [0.5, 0.041], [1, 0.045]], "neck", "upperChest", "head", 12);

    /* A skull rather than a ball: narrowed to a jaw below the cheekbones,
       flattened at the temples and the occiput. Four cheap deformations, and
       without them a head reads as a mannequin's. */
    b.blob(
      head.clone().add(new THREE.Vector3(0, 0.055, 0.004)),
      new THREE.Vector3(0.074, 0.098, 0.086),
      "head",
      20,
      16,
      (v) => {
        const h = v.y / 0.098;
        if (h < -0.1) {
          const t = Math.min(1, (-h - 0.1) / 0.9);
          v.x *= 1 - 0.30 * t * t;
          v.z *= 1 - 0.16 * t * t;
          if (v.z > 0) v.z += 0.010 * t * t;
        }
        v.x *= 0.93;
        if (v.z < 0) v.z *= 0.94;
      },
    );

    // A nose, so a head turn is legible in silhouette. Without one the head is
    // a sphere and its yaw — which carries negation — cannot be read at all.
    b.blob(
      head.clone().add(new THREE.Vector3(0, 0.040, 0.082)),
      new THREE.Vector3(0.012, 0.019, 0.017),
      "head",
      10,
      8,
    );

    /* ---------------------------------------------------------- arms --- */
    for (const side of ["left", "right"] as const) {
      const S = side === "left" ? "left" : "right";
      const shoulder = this.world(`${S}UpperArm` as BoneName);
      const elbow = this.world(`${S}LowerArm` as BoneName);
      const wrist = this.world(`${S}Hand` as BoneName);

      b.tube(
        shoulder, elbow,
        [[0, 0.050], [0.25, 0.046], [0.6, 0.040], [1, 0.035]],
        `${S}UpperArm` as BoneName, `${S}Shoulder` as BoneName, `${S}LowerArm` as BoneName,
        12,
      );
      b.tube(
        elbow, wrist,
        [[0, 0.036], [0.2, 0.040], [0.55, 0.034], [1, 0.026]],
        `${S}LowerArm` as BoneName, `${S}UpperArm` as BoneName, `${S}Hand` as BoneName,
        12,
      );
    }

    return b.build();
  }

  /**
   * One hand. This is where the geometry budget goes.
   *
   * A palm is not a box. The three features a reader actually uses to judge
   * palm orientation are the THENAR eminence (the pad at the base of the
   * thumb), the HYPOTHENAR pad along the little-finger edge, and the arch
   * across the knuckles. All three are here, and all three are why palm-up and
   * palm-down look different rather than merely lit differently — which, on the
   * old box-shaped palm, was the entire distinction.
   */
  private buildHand(b: MeshBuilder, side: Side): void {
    const S = side === "left" ? "left" : "right";
    const hand: BoneName = `${S}Hand` as BoneName;
    const wrist = this.world(hand);
    const chains = fingerChains(side);

    const indexMcp = this.world(chains[1].bones[0]);
    const littleMcp = this.world(chains[4].bones[0]);
    const middleMcp = this.world(chains[2].bones[0]);

    const across = new THREE.Vector3().subVectors(indexMcp, littleMcp);
    const along = new THREE.Vector3().subVectors(middleMcp, wrist).normalize();
    const palmNormal = new THREE.Vector3().crossVectors(across, along).normalize();
    const width = across.length();

    /* The palm slab: a flattened tube from wrist to knuckles, wider at the
       knuckle line than at the wrist, which is the shape a real palm is. */
    const knuckleMid = new THREE.Vector3()
      .addVectors(indexMcp, littleMcp)
      .multiplyScalar(0.5);
    b.tube(
      wrist, knuckleMid,
      [[0, width * 0.44], [0.35, width * 0.52], [0.75, width * 0.60], [1, width * 0.58]],
      hand, `${S}LowerArm` as BoneName, null,
      14,
      // Squashed across the palm normal: a hand is about two and a half times
      // wider than it is thick, and a round palm reads as a paw.
      { squash: 0.40, capStart: true, capEnd: true },
    );

    // Thenar eminence — the muscular pad at the base of the thumb.
    const thumbBase = this.world(chains[0].bones[0]);
    b.blob(
      thumbBase.clone().lerp(wrist, 0.3).addScaledVector(palmNormal, -width * 0.10),
      new THREE.Vector3(width * 0.26, width * 0.20, width * 0.22),
      hand,
      10,
      8,
    );
    // Hypothenar pad, along the little-finger edge.
    b.blob(
      littleMcp.clone().lerp(wrist, 0.45).addScaledVector(palmNormal, -width * 0.07),
      new THREE.Vector3(width * 0.20, width * 0.16, width * 0.20),
      hand,
      10,
      8,
    );

    /* Fingers. Each phalanx is its own tube weighted to its own bone, so a
       curl deforms continuously instead of coming apart at the knuckles. */
    for (const chain of chains) {
      const isThumb = chain.finger === "thumb";
      const base = isThumb ? 0.0175 : 0.0145;
      const scale = chain.finger === "little" ? 0.86 : chain.finger === "index" ? 0.97 : 1;

      for (let segment = 0; segment < 3; segment += 1) {
        const bone = chain.bones[segment];
        const from = this.world(bone);
        const to =
          segment < 2
            ? this.world(chain.bones[segment + 1])
            : this.fingerTip(chain.bones[2]);

        const r0 = base * scale * (1 - segment * 0.13);
        const r1 = base * scale * (1 - (segment + 1) * 0.13);
        const parent: BoneName = segment === 0 ? hand : chain.bones[segment - 1];
        const child: BoneName | null = segment < 2 ? chain.bones[segment + 1] : null;

        b.tube(
          from, to,
          segment === 2
            ? // The last phalanx rounds off rather than ending in a disc — a
              // flat fingertip is the single most obviously synthetic thing a
              // hand can have, and tips are what a reader looks at.
              [[0, r0], [0.45, r1 * 1.02], [0.78, r1 * 0.92], [0.93, r1 * 0.66], [1, r1 * 0.22]]
            : [[0, r0], [0.5, (r0 + r1) / 2], [1, r1]],
          bone, parent, child,
          9,
          { capEnd: segment === 2, squash: isThumb ? 1.12 : 0.94 },
        );
      }
    }
  }

  /** Where a distal phalanx ends, from the leaf node the skeleton carries. */
  private fingerTip(distal: BoneName): THREE.Vector3 {
    const bone = this.skeleton.bones.get(distal)!;
    const leaf = bone.children[0];
    if (leaf) return leaf.getWorldPosition(new THREE.Vector3());
    // No leaf: continue straight on for a plausible phalanx length.
    const here = bone.getWorldPosition(new THREE.Vector3());
    const parent = bone.parent as THREE.Object3D | null;
    if (!parent) return here.clone().add(new THREE.Vector3(0, 0.02, 0));
    const back = parent.getWorldPosition(new THREE.Vector3());
    return here.clone().add(here.clone().sub(back).setLength(0.019));
  }

  /** Torso and sleeves. */
  private buildGarment(boneIndex: Map<BoneName, number>): THREE.BufferGeometry {
    const b = new MeshBuilder(boneIndex);

    const hips = this.world("hips");
    const spine = this.world("spine");
    const chest = this.world("chest");
    const upperChest = this.world("upperChest");
    const neck = this.world("neck");

    /* A torso is much wider than it is deep, and that difference is most of
       what makes a body read as a body from three-quarters. `squash` supplies
       it; a solid of revolution here reads as a bottle. */
    b.tube(
      hips, spine,
      [[0, 0.142], [0.35, 0.140], [0.7, 0.136], [1, 0.134]],
      "hips", null, "spine", 24, { squash: 1.26, capStart: true },
    );
    // A waist. Straight sides from hip to chest is a barrel, and a barrel is
    // most of why a generated torso reads as furniture.
    b.tube(
      spine, chest,
      [[0, 0.134], [0.4, 0.133], [0.75, 0.141], [1, 0.148]],
      "spine", "hips", "chest", 24, { squash: 1.30 },
    );
    b.tube(
      chest, upperChest,
      [[0, 0.148], [0.4, 0.155], [0.75, 0.159], [1, 0.158]],
      "chest", "spine", "upperChest", 24, { squash: 1.34 },
    );
    /* The trapezius: the rings above the shoulder line. Without them the torso
       stops flat at the collar and the neck stands clear of it — which reads,
       unmistakably, as a giraffe. Real shoulders climb toward the neck. */
    b.tube(
      upperChest, neck,
      [[0, 0.158], [0.3, 0.152], [0.58, 0.126], [0.82, 0.083], [1, 0.052]],
      "upperChest", "chest", "neck", 24,
      { squash: 1.36, capEnd: true },
    );

    for (const side of ["left", "right"] as const) {
      const S = side === "left" ? "left" : "right";
      const shoulder = this.world(`${S}UpperArm` as BoneName);
      const elbow = this.world(`${S}LowerArm` as BoneName);
      // A deltoid cap plus a short sleeve, so the arm reads as growing out of
      // the body rather than being pushed into it.
      /* The deltoid caps the joint and no more. Sized larger it becomes a
         shoulder PAD — the shape a padded jacket makes, not a shoulder — which
         is the single most common way a generated figure looks inflated. It is
         only a little wider than the arm beneath it. */
      b.blob(
        shoulder.clone().lerp(elbow, 0.06),
        new THREE.Vector3(0.052, 0.050, 0.052),
        `${S}UpperArm` as BoneName,
        12,
        10,
      );
      b.tube(
        shoulder, elbow,
        [[0, 0.053], [0.3, 0.052], [0.62, 0.048], [0.68, 0.045]],
        `${S}UpperArm` as BoneName, `${S}Shoulder` as BoneName, `${S}LowerArm` as BoneName,
        14,
        { capEnd: true },
      );
    }

    return b.build();
  }

  private buildHair(boneIndex: Map<BoneName, number>): THREE.BufferGeometry {
    const b = new MeshBuilder(boneIndex);
    const head = this.world("head");
    /* The hairline is the whole job. Cut it too low and it covers the brow —
       and since ISL carries clause type on the brows, that is not a styling
       mistake, it is a loss of meaning. This sits well above them. */
    b.blob(
      head.clone().add(new THREE.Vector3(0, 0.062, -0.004)),
      new THREE.Vector3(0.079, 0.101, 0.090),
      "head",
      20,
      14,
      (v) => {
        const h = v.y / 0.101;
        const front = v.z > 0 ? v.z / 0.090 : 0;
        // Tuck everything below the hairline inside the skull rather than
        // flattening it, which would leave a brim standing off the forehead.
        const floor = -0.15 + front * 0.62;
        if (h < floor) {
          const depth = Math.min(1, (floor - h) / 0.9);
          v.y = floor * 0.101 - depth * 0.004;
          v.x *= 1 - 0.34 * depth;
          v.z *= 1 - 0.34 * depth;
        }
      },
    );
    return b.build();
  }

  /**
   * The ghost: head and shoulders, faint, for hands mode.
   *
   * Deliberately featureless — no eyes, no mouth, no hair. The moment it has a
   * face it is a character again, and being a *barely visible* character is
   * worse than being a clear one. This is a landmark, not a person: it exists so
   * "at the forehead" and "at the chest" remain different places.
   */
  private buildGhost(boneIndex: Map<BoneName, number>): THREE.BufferGeometry {
    const b = new MeshBuilder(boneIndex);
    const head = this.world("head");
    const neck = this.world("neck");
    const upperChest = this.world("upperChest");

    b.blob(
      head.clone().add(new THREE.Vector3(0, 0.055, 0.004)),
      new THREE.Vector3(0.072, 0.096, 0.084),
      "head", 16, 12,
      (v) => {
        const h = v.y / 0.096;
        if (h < -0.1) {
          const t = Math.min(1, (-h - 0.1) / 0.9);
          v.x *= 1 - 0.30 * t * t;
          v.z *= 1 - 0.16 * t * t;
        }
        v.x *= 0.93;
      },
    );
    b.tube(neck, head, [[0, 0.040], [1, 0.042]], "neck", "upperChest", "head", 12);
    // Shoulder line only — enough to say where the midline and the shoulders
    // are, which is what most sign locations are specified against.
    b.tube(
      upperChest, neck,
      [[0, 0.150], [0.45, 0.128], [0.8, 0.080], [1, 0.052]],
      "upperChest", "chest", "neck", 20, { squash: 1.34 },
    );
    for (const side of ["left", "right"] as const) {
      const S = side === "left" ? "left" : "right";
      const shoulder = this.world(`${S}UpperArm` as BoneName);
      b.blob(shoulder.clone(), new THREE.Vector3(0.048, 0.046, 0.048),
        `${S}UpperArm` as BoneName, 10, 8);
    }
    return b.build();
  }

  /**
   * The ghost's material.
   *
   * Depth-write OFF and a low opacity, so hands passing in FRONT of the face —
   * which they do constantly, and which is itself a sign location — are never
   * occluded by it. A ghost that hides the hands would defeat its own purpose.
   */
  private ghostMaterial(): THREE.MeshStandardMaterial {
    const material = new THREE.MeshStandardMaterial({
      color: 0x9fb0b8,
      roughness: 0.95,
      metalness: 0,
      transparent: true,
      opacity: 0.14,
      depthWrite: false,
    });
    this.disposables.push(material);
    return material;
  }

  private buildEyes(boneIndex: Map<BoneName, number>): THREE.BufferGeometry {
    const b = new MeshBuilder(boneIndex);
    const head = this.world("head");
    /* Small, and level with the middle of the head — where real pupils sit.
       The old build put them above centre and 24% too far apart, which is most
       of why that face read as a mask. */
    for (const x of [-0.029, 0.029]) {
      b.blob(
        head.clone().add(new THREE.Vector3(x, 0.055, 0.070)),
        new THREE.Vector3(0.0092, 0.0112, 0.006),
        "head",
        10,
        8,
      );
    }
    return b.build();
  }

  /* ------------------------------------------------------------- apply -- */

  boneNode(name: BoneName): THREE.Object3D | null {
    return this.skeleton.bones.get(name) ?? null;
  }

  apply(pose: HumanoidPose, dt: number): void {
    for (const [name, bone] of this.skeleton.bones) {
      const rotation = pose.rotations.get(name);
      if (rotation) bone.quaternion.copy(rotation);
    }

    /* A shallow breath, and only on the chest.
       A perfectly motionless avatar reads as a frozen frame rather than as a
       person waiting — but it must never touch the arms, because during signing
       the real motion IS the message and a loop on top of it fights the sign. */
    this.breathPhase += dt;
    const chest = this.skeleton.bones.get("chest");
    if (chest) {
      const breath = Math.sin(this.breathPhase * 1.5) * 0.008;
      chest.quaternion.multiply(
        BREATH.setFromAxisAngle(BREATH_AXIS, breath),
      );
    }

    this.skeleton.root.updateMatrixWorld(true);
  }

  dispose(): void {
    for (const item of this.disposables) item.dispose();
    for (const mesh of this.meshes) mesh.skeleton?.dispose?.();
  }
}

const BREATH = new THREE.Quaternion();
const BREATH_AXIS = new THREE.Vector3(1, 0, 0);

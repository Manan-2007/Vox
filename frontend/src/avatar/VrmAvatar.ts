/**
 * The avatar: a VRM humanoid, bound to the rig by bone name.
 *
 * ---------------------------------------------------------------------------
 * WHY VRM AND NOT A PROCEDURAL FIGURE
 * ---------------------------------------------------------------------------
 * The previous build assembled its figure from about 120 unparented three.js
 * primitives — cylinders between landmarks, spheres at joints, a box for each
 * palm — repositioned every frame. There was no `THREE.Bone`, no `Skeleton` and
 * no `SkinnedMesh` anywhere in the project. Everything the brief lists as broken
 * follows from that:
 *
 *   * A finger was three cylinders and three spheres, so a bend gapped on the
 *     outside of the curve and interpenetrated on the inside.
 *   * A palm was a box, so palm-up and palm-down differed only by which flat
 *     face caught the light — and palm orientation is one of the five parameters
 *     that distinguish one sign from another.
 *   * Nothing could be limited, because a limit is a statement about a rotation
 *     and there were no rotations.
 *
 * A VRM fixes all of it at once, and not because it looks better. The VRM
 * humanoid specification defines fifteen finger bones per hand, which is exactly
 * the fifteen phalanx segments MediaPipe reports. The correspondence is
 * one-to-one, so a tracked hand becomes fifteen joint rotations with no fitting
 * — and a rotation can be limited, blended and skinned.
 *
 * ---------------------------------------------------------------------------
 * NORMALIZED BONES
 * ---------------------------------------------------------------------------
 * Everything is driven through three-vrm's NORMALIZED humanoid, never the raw
 * one. In the normalized rig every bone's rest rotation is identity, so a
 * rotation solved for one model is valid for every other. Driving the raw rig
 * would bake in one artist's bind pose and require re-deriving every constant in
 * `core/humanoid.ts` for each new .vrm.
 *
 * ---------------------------------------------------------------------------
 * WHY LOADING IS STRICT
 * ---------------------------------------------------------------------------
 * A .vrm missing its finger bones loads perfectly and signs nothing — the hands
 * simply stay flat while everything else moves. That is the worst possible
 * failure for this product: it looks like a subtle animation bug and it is
 * actually total loss of meaning. So the required bones are checked at load and
 * a model that lacks them is REFUSED, by name.
 */

import { VRMLoaderPlugin, VRMUtils, type VRM } from "@pixiv/three-vrm";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import type { BoneName, HumanoidPose } from "../core/humanoid";
import { DRIVEN_BONES, REQUIRED_BONES } from "../core/humanoid";
import { captureRest, type RestPose } from "./retarget/rest";
import type { SigningAvatar } from "./SigningAvatar";

export interface AvatarLoadResult {
  avatar: VrmAvatar;
  /** Optional bones the model does not have. Not fatal; noted for the panel. */
  missing: BoneName[];
  /** Bytes transferred, for the debug panel. */
  bytes: number;
}

export class AvatarError extends Error {
  /** What the user can do about it, in one sentence. */
  readonly remedy: string;

  constructor(message: string, remedy: string) {
    super(message);
    this.name = "AvatarError";
    this.remedy = remedy;
  }
}

/**
 * A loaded, posable VRM.
 *
 * Construct via `loadVrmAvatar`. `apply` per frame, `dispose` when the canvas
 * goes away.
 */
export class VrmAvatar implements SigningAvatar {
  readonly vrm: VRM;
  readonly rest: RestPose;
  readonly root: THREE.Group;
  readonly kind = "VRM";

  private readonly bones = new Map<BoneName, THREE.Object3D>();
  private readonly hipsRest = new THREE.Vector3();
  private disposed = false;

  constructor(vrm: VRM) {
    this.vrm = vrm;
    this.root = new THREE.Group();
    this.root.add(vrm.scene);

    const humanoid = vrm.humanoid;
    if (!humanoid) {
      throw new AvatarError(
        "This file has no VRM humanoid definition.",
        "Export it from VRoid Studio, or convert the GLB to VRM with UniVRM.",
      );
    }

    for (const bone of DRIVEN_BONES) {
      const node = humanoid.getNormalizedBoneNode(bone);
      if (node) this.bones.set(bone, node);
    }

    const missing = REQUIRED_BONES.filter((bone) => !this.bones.has(bone));
    if (missing.length > 0) {
      const fingers = missing.filter((b) => /Thumb|Index|Middle|Ring|Little/.test(b));
      throw new AvatarError(
        `This model is missing ${missing.length} bone${missing.length > 1 ? "s" : ""} the rig needs: ${missing.join(", ")}.`,
        fingers.length > 0
          ? "It has no finger bones, so it cannot form handshapes. Choose a model exported with full hand rigging."
          : "Choose a model with a complete VRM humanoid skeleton.",
      );
    }

    /* Rest capture must happen with the model in its bind pose and BEFORE any
       rotation is applied, since every canonical frame is measured from it. */
    humanoid.resetNormalizedPose();
    vrm.scene.updateMatrixWorld(true);
    /* Measured in the NORMALIZED ROOT's space, not the world's. A VRM 0.x model
       has been turned 180° by `rotateVRM0`, and measuring through that rotation
       puts every axis half a turn away from the space the rig's local rotations
       actually compose in. */
    this.rest = captureRest(
      (name) => this.bones.get(name) ?? null,
      humanoid.normalizedHumanBonesRoot,
    );

    const hips = this.bones.get("hips");
    if (hips) this.hipsRest.copy(hips.position);
  }

  boneNode(name: BoneName): THREE.Object3D | null {
    return this.bones.get(name) ?? null;
  }

  /** Optional bones this model does not have. */
  get missingBones(): BoneName[] {
    return DRIVEN_BONES.filter((bone) => !this.bones.has(bone));
  }

  /** Triangle count, for the debug panel. */
  get triangles(): number {
    let total = 0;
    this.vrm.scene.traverse((object) => {
      const mesh = object as THREE.Mesh;
      const geometry = mesh.geometry as THREE.BufferGeometry | undefined;
      if (!geometry) return;
      const index = geometry.getIndex();
      const position = geometry.getAttribute("position");
      if (index) total += index.count / 3;
      else if (position) total += position.count / 3;
    });
    return Math.round(total);
  }

  /**
   * Wear one pose.
   *
   * `dt` drives three-vrm's own update — spring bones (hair and clothing follow
   * the body), look-at, and expressions. Skipping it leaves the model rigid in
   * exactly the places that sell it as a body rather than a mannequin.
   */
  apply(pose: HumanoidPose, dt: number): void {
    if (this.disposed) return;

    for (const [bone, node] of this.bones) {
      const rotation = pose.rotations.get(bone);
      if (rotation) node.quaternion.copy(rotation);
    }

    const hips = this.bones.get("hips");
    if (hips) hips.position.copy(this.hipsRest).add(pose.rootOffset);

    // Push the normalized pose onto the raw bones the mesh is skinned to, then
    // let three-vrm run its own systems.
    this.vrm.humanoid?.update();
    this.vrm.update(dt);
  }

  /** Aim the model's gaze, if it has a look-at rig. */
  lookAt(target: THREE.Object3D | null): void {
    if (this.vrm.lookAt) this.vrm.lookAt.target = target ?? undefined;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.root.remove(this.vrm.scene);
    VRMUtils.deepDispose(this.vrm.scene);
  }
}

/**
 * Load a .vrm (or a VRM-extended .glb) from a URL.
 *
 * `onProgress` reports 0-1 where the server sends a content length, and −1 where
 * it does not — a distinction worth keeping, because a progress bar that sits at
 * zero is worse than one that says it cannot tell.
 */
export async function loadVrmAvatar(
  url: string,
  onProgress?: (fraction: number, bytes: number) => void,
): Promise<AvatarLoadResult> {
  const loader = new GLTFLoader();
  loader.register((parser) => new VRMLoaderPlugin(parser));

  let bytes = 0;

  const gltf = await new Promise<Awaited<ReturnType<GLTFLoader["loadAsync"]>>>(
    (resolve, reject) => {
      loader.load(
        url,
        resolve,
        (event) => {
          bytes = event.loaded;
          onProgress?.(event.total > 0 ? event.loaded / event.total : -1, event.loaded);
        },
        (error) => {
          const detail = error instanceof Error ? error.message : String(error);
          reject(
            new AvatarError(
              `Could not load the avatar from ${url} — ${detail}`,
              "Check the file exists and is a valid .vrm.",
            ),
          );
        },
      );
    },
  );

  const vrm = gltf.userData.vrm as VRM | undefined;
  if (!vrm) {
    throw new AvatarError(
      "That file loaded, but it is not a VRM.",
      "It looks like a plain glTF. Convert it with UniVRM, or export a .vrm from VRoid Studio.",
    );
  }

  /* Housekeeping three-vrm recommends and the docs bury.

     `removeUnnecessaryVertices` and `combineSkeletons` cut draw calls
     substantially on VRoid exports, which routinely ship dozens of separate
     skinned meshes. `rotateVRM0` is the one that matters for correctness: VRM 0.x
     models face −Z, VRM 1.0 face +Z, and without this a 0.x model signs with its
     back to the viewer. */
  VRMUtils.removeUnnecessaryVertices(gltf.scene);
  VRMUtils.combineSkeletons(gltf.scene);
  VRMUtils.rotateVRM0(vrm);

  // Frustum culling off: the mesh's bounding volume is computed from the bind
  // pose, and a raised arm reaches well outside it. With culling on, hands
  // disappear at exactly the moment a sign is at its most extreme.
  vrm.scene.traverse((object) => {
    object.frustumCulled = false;
    const mesh = object as THREE.Mesh;
    if (mesh.isMesh) {
      mesh.castShadow = true;
      mesh.receiveShadow = true;
    }
  });

  const avatar = new VrmAvatar(vrm);
  return { avatar, missing: avatar.missingBones, bytes };
}

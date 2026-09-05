/**
 * What the stage needs from a body, and nothing more.
 *
 * Two implementations:
 *
 *   VrmAvatar         a real .vrm — the production path, best quality
 *   ProceduralAvatar  a generated SkinnedMesh — the built-in fallback
 *
 * They are interchangeable because they share the ONE thing that matters: a
 * humanoid bone hierarchy with VRM's names, measured into a `RestPose` by the
 * same `captureRest`. Everything upstream — the retargeter, the joint limits,
 * the blender, the debug view — is written against bone names and knows nothing
 * about which body it is driving.
 *
 * That is the whole point of the boundary. Swapping the signer changes what the
 * viewer sees and changes no logic at all.
 */

import type * as THREE from "three";
import type { BoneName, HumanoidPose } from "../core/humanoid";
import type { RestPose } from "./retarget/rest";

/**
 * How much of the body to draw.
 *
 *   "figure"  the whole signer — carries non-manual grammar on the face
 *   "hands"   hands and forearms over a faint head-and-shoulders reference
 */
export type Presentation = "figure" | "hands";

export interface SigningAvatar {
  /** Scene node to add to the stage. */
  readonly root: THREE.Object3D;
  /** Measured rest geometry — the retargeter's reference for this body. */
  readonly rest: RestPose;
  /** For the debug panel. */
  readonly triangles: number;
  /** Human-readable, for the panel: "VRM" or "Built-in". */
  readonly kind: string;

  /**
   * A driven bone's node, for framing and inspection.
   *
   * Must return the NORMALIZED bone — the one whose rest rotation is identity —
   * because that is the space every rotation in the pipeline is expressed in.
   */
  boneNode(name: BoneName): THREE.Object3D | null;

  /** Wear one pose. `dt` drives any secondary motion the body has. */
  apply(pose: HumanoidPose, dt: number): void;

  /**
   * Draw the whole figure, or hands only.
   *
   * Optional: a body that cannot hide its own torso simply ignores it, and the
   * stage falls back to framing tightly on the hands instead — which gets most
   * of the benefit from any avatar, including a .vrm.
   */
  setPresentation?(mode: Presentation): void;

  dispose(): void;
}

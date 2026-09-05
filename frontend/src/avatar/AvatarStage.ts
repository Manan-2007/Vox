/**
 * The stage: renderer, lighting, camera framing, and the loop that ties the
 * pipeline together.
 *
 * Deliberately NOT a React component. The render loop runs sixty times a second
 * and touches nothing React owns; making it a component meant every tracking
 * frame became a state update and a re-render of the whole page, which is most
 * of where the old build's main-thread budget went.
 *
 * ---------------------------------------------------------------------------
 * FRAMING — REQUIREMENT 9, AND WHY IT NEEDED SOLVING
 * ---------------------------------------------------------------------------
 * The old camera fitted a FIXED box: `{ centreY: -0.20, height: 3.15 }`, with a
 * comment accepting that resting hands leave the frame. For a sign-language
 * product that is backwards — the hands are the message, and a sign made above
 * the head or out to the side was simply cropped.
 *
 * Here the camera fits the CONTENT: head, both shoulders, both forearms and both
 * hands, every frame, with the extent smoothed so the view breathes rather than
 * snapping. A sign that reaches high pulls the camera back before the hand
 * arrives, because the smoothing is applied to the extent rather than to the
 * camera.
 *
 * ---------------------------------------------------------------------------
 * LIGHTING
 * ---------------------------------------------------------------------------
 * An environment map does the work. The old build used four analytic lights and
 * `MeshPhysicalMaterial` with sheen and clearcoat enabled — reflectance models
 * that need something to reflect — and never set `scene.environment`, so it paid
 * for the most expensive material in three.js and got the look of the cheapest.
 * A generated room environment gives real indirect light for one render at
 * startup and nothing per frame.
 */

import * as THREE from "three";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import type { HumanoidPose } from "../core/humanoid";
import { blankPose } from "../core/humanoid";
import { clamp } from "../core/math";
import { PoseBlender, type BlendReport } from "./pose/blend";
import { buildIdlePose } from "./pose/idle";
import type { Presentation, SigningAvatar } from "./SigningAvatar";

export interface StageStats {
  fps: number;
  frameMs: number;
  drawCalls: number;
  triangles: number;
  /** Largest single-frame bone rotation, degrees — the snap detector. */
  peakStep: number;
}

export interface StageOptions {
  /** Mirror the figure, so a learner can copy it directly. */
  mirror?: boolean;
  /** Show the debug skeleton overlay. */
  skeleton?: boolean;
  /** Draw the whole signer, or hands only. */
  presentation?: Presentation;
}

/**
 * Points that must stay in shot, by bone. Forearms are included as well as
 * hands: a sign is read from the whole arm's movement, and a frame that clips
 * the elbow loses the movement path even when the hand is visible.
 */
const FRAMED_BONES = [
  "head",
  "leftUpperArm",
  "rightUpperArm",
  "leftLowerArm",
  "rightLowerArm",
  "leftHand",
  "rightHand",
  "leftMiddleDistal",
  "rightMiddleDistal",
  "leftThumbDistal",
  "rightThumbDistal",
] as const;

/**
 * Margin around the fitted content, as a fraction of its extent.
 *
 * Generous on purpose. The framed set ends at the finger JOINTS, so the tips
 * reach a little past it; and a sign is read from where a hand is going as much
 * as where it is, so a hand pressed against the edge of frame is already too
 * late. The brief's requirement is "never crop the hands", and the cost of extra
 * margin is a slightly smaller figure — much the cheaper error.
 */
const FRAMING_MARGIN = 1.38;

/**
 * The subset that defines the frame in HANDS mode: wrists, fingertips and the
 * elbows the forearms hang from. The signing space is then sized by the hands
 * rather than by a body the viewer cannot see.
 */
const HAND_FRAMED = new Set<string>([
  "leftLowerArm", "rightLowerArm",
  "leftHand", "rightHand",
  "leftMiddleDistal", "rightMiddleDistal",
  "leftThumbDistal", "rightThumbDistal",
]);
/** Seconds for the framing to follow a change in extent. */
const FRAMING_SMOOTH = 0.35;

export class AvatarStage {
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  blender: PoseBlender;

  private readonly renderer: THREE.WebGLRenderer;
  private readonly mount: HTMLElement;
  private readonly pivot = new THREE.Group();
  private avatar: SigningAvatar | null = null;
  private skeletonHelper: THREE.SkeletonHelper | null = null;

  private readonly framedNodes: THREE.Object3D[] = [];
  private readonly box = new THREE.Box3();
  private readonly point = new THREE.Vector3();
  private readonly centre = new THREE.Vector3(0, 1.4, 0);
  private readonly smoothCentre = new THREE.Vector3(0, 1.4, 0);
  private smoothHeight = 0.9;
  private smoothWidth = 0.9;

  private raf = 0;
  private previous = 0;
  private readonly started = performance.now();
  private frameCount = 0;
  private fpsWindow = 0;
  private disposed = false;

  readonly stats: StageStats = {
    fps: 0,
    frameMs: 0,
    drawCalls: 0,
    triangles: 0,
    peakStep: 0,
  };

  private options: Required<StageOptions> = {
    mirror: true,
    skeleton: false,
    presentation: "figure",
  };
  private lastReport: BlendReport | null = null;

  /**
   * Supplies the pose to wear this frame, or null when there is no source.
   * Set by the owner; called once per rendered frame.
   */
  source: ((dt: number, time: number) => HumanoidPose | null) | null = null;

  /** Called after each frame with the blend report, for the debug panel. */
  onFrame: ((report: BlendReport, stats: StageStats) => void) | null = null;

  constructor(mount: HTMLElement, options: StageOptions = {}) {
    this.mount = mount;
    Object.assign(this.options, options);

    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: "high-performance",
      // Depth precision matters here: fingers are millimetres apart at the scale
      // the camera frames them, and the default 16-bit depth buffer z-fights
      // between adjacent phalanges on some integrated GPUs.
      logarithmicDepthBuffer: false,
    });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    mount.appendChild(this.renderer.domElement);
    this.renderer.domElement.style.touchAction = "none";

    this.camera = new THREE.PerspectiveCamera(30, 1, 0.05, 40);

    /* Environment. One render at startup, then nothing per frame — and it is
       what makes skin read as skin rather than as painted plastic. */
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    const environment = pmrem.fromScene(new RoomEnvironment(), 0.04);
    this.scene.environment = environment.texture;
    this.scene.environmentIntensity = 0.85;
    pmrem.dispose();

    /* One shadow-casting key. The environment supplies fill and bounce, so a
       second and third analytic light would only wash out the form the
       environment is already describing. */
    const key = new THREE.DirectionalLight(0xfff4e8, 1.6);
    key.position.set(1.4, 3.0, 2.2);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.near = 0.5;
    key.shadow.camera.far = 8;
    key.shadow.camera.left = -1.2;
    key.shadow.camera.right = 1.2;
    key.shadow.camera.top = 1.2;
    key.shadow.camera.bottom = -1.2;
    // normalBias offsets along the surface normal, which is what actually cures
    // acne on smooth curved geometry. A large depth bias alone trades acne for
    // peter-panning and, on a face, for a muddy render.
    key.shadow.bias = -0.0002;
    key.shadow.normalBias = 0.018;
    key.shadow.radius = 2.5;
    this.scene.add(key);
    this.scene.add(key.target);

    // A cool rim, to separate the silhouette from a light page.
    const rim = new THREE.DirectionalLight(0xcfe6ff, 0.55);
    rim.position.set(-1.8, 1.6, -2.4);
    this.scene.add(rim);

    this.scene.add(this.pivot);

    // Without a model there is no rest pose to clamp against; the blender is
    // rebuilt with one as soon as an avatar loads.
    this.blender = new PoseBlender(blankPose(), blankPose());

    this.bindPointer();
    this.resize();
  }

  /* ------------------------------------------------------------- avatar -- */

  setAvatar(avatar: SigningAvatar | null): void {
    if (this.avatar) {
      this.pivot.remove(this.avatar.root);
      this.detachSkeleton();
    }
    this.avatar = avatar;
    this.framedNodes.length = 0;
    if (!avatar) return;

    /* Rebuild with this body's own rest pose, so the blender can hold every bone
       inside its anatomical range while slerping — and, critically, so the pose
       it relaxes INTO is a relaxed signer rather than a T-pose. */
    this.blender = new PoseBlender(
      blankPose(),
      buildIdlePose(avatar.rest),
      avatar.rest,
    );
    avatar.setPresentation?.(this.options.presentation);
    this.pivot.add(avatar.root);
    for (const name of FRAMED_BONES) {
      const node = avatar.boneNode(name);
      if (node) this.framedNodes.push(node);
    }
    this.stats.triangles = avatar.triangles;
    if (this.options.skeleton) this.attachSkeleton();
  }

  setOptions(options: StageOptions): void {
    const wantsSkeleton = options.skeleton ?? this.options.skeleton;
    if (wantsSkeleton !== this.options.skeleton) {
      if (wantsSkeleton) this.attachSkeleton();
      else this.detachSkeleton();
    }
    const wantsMode = options.presentation ?? this.options.presentation;
    if (wantsMode !== this.options.presentation) {
      this.avatar?.setPresentation?.(wantsMode);
    }
    Object.assign(this.options, options);
  }

  private attachSkeleton(): void {
    if (this.skeletonHelper || !this.avatar) return;
    // Drawn against the avatar's own root, so it works for either body.
    this.skeletonHelper = new THREE.SkeletonHelper(this.avatar.root);
    (this.skeletonHelper.material as THREE.LineBasicMaterial).depthTest = false;
    this.skeletonHelper.renderOrder = 999;
    this.scene.add(this.skeletonHelper);
  }

  private detachSkeleton(): void {
    if (!this.skeletonHelper) return;
    this.scene.remove(this.skeletonHelper);
    this.skeletonHelper.dispose();
    this.skeletonHelper = null;
  }

  /* -------------------------------------------------------------- input -- */

  private dragging = false;
  private dragX = 0;
  private dragY = 0;
  private yaw = 0;
  private pitch = 0;
  private userMoved = false;
  private readonly listeners: [string, (event: PointerEvent) => void][] = [];

  private bindPointer(): void {
    const element = this.renderer.domElement;
    const down = (event: PointerEvent) => {
      this.dragging = true;
      this.userMoved = true;
      this.dragX = event.clientX;
      this.dragY = event.clientY;
      element.setPointerCapture(event.pointerId);
    };
    const move = (event: PointerEvent) => {
      if (!this.dragging) return;
      this.yaw += (event.clientX - this.dragX) * 0.008;
      this.pitch = clamp(this.pitch + (event.clientY - this.dragY) * 0.004, -0.35, 0.35);
      this.dragX = event.clientX;
      this.dragY = event.clientY;
    };
    const up = (event: PointerEvent) => {
      this.dragging = false;
      if (element.hasPointerCapture(event.pointerId)) {
        element.releasePointerCapture(event.pointerId);
      }
    };
    this.listeners.push(["pointerdown", down], ["pointermove", move], ["pointerup", up], ["pointercancel", up]);
    for (const [type, handler] of this.listeners) {
      element.addEventListener(type, handler as EventListener);
    }
    element.style.cursor = "grab";
  }

  /* --------------------------------------------------------------- loop -- */

  start(): void {
    if (this.raf) return;
    this.previous = performance.now();
    const tick = () => {
      this.raf = requestAnimationFrame(tick);
      this.frame();
    };
    this.raf = requestAnimationFrame(tick);
  }

  stop(): void {
    if (!this.raf) return;
    cancelAnimationFrame(this.raf);
    this.raf = 0;
  }

  private frame(): void {
    if (this.disposed) return;
    const now = performance.now();
    const dt = Math.min(0.1, (now - this.previous) / 1000);
    this.previous = now;
    const time = (now - this.started) / 1000;

    /* One blender, whatever the source. A source that stops — the camera lost,
       the queue finished — relaxes the figure through the state machine instead
       of leaving it frozen or hiding it. */
    const measured = this.source ? this.source(dt, time) : null;
    const report = this.blender.apply(measured, dt);
    this.lastReport = report;

    if (this.avatar) {
      this.avatar.apply(this.blender.output, dt);
      this.updateFraming(dt);
    }

    this.pivot.scale.x = this.options.mirror ? -1 : 1;
    if (this.userMoved) {
      this.pivot.rotation.y = this.yaw;
      this.pivot.rotation.x = this.pitch;
    }

    this.renderer.render(this.scene, this.camera);

    /* ---------------------------------------------------------- stats -- */
    this.frameCount += 1;
    this.fpsWindow += dt;
    if (this.fpsWindow >= 0.5) {
      this.stats.fps = Math.round(this.frameCount / this.fpsWindow);
      this.frameCount = 0;
      this.fpsWindow = 0;
    }
    this.stats.frameMs = Math.round((performance.now() - now) * 100) / 100;
    this.stats.drawCalls = this.renderer.info.render.calls;
    this.stats.peakStep = Math.round((report.peakStep * 180) / Math.PI * 100) / 100;
    this.onFrame?.(report, this.stats);
  }

  /**
   * Fit the camera to what must stay visible.
   *
   * The EXTENT is smoothed, not the camera position, and the difference is worth
   * spelling out: smoothing the camera makes it chase the content and always
   * arrive late, so a fast sign is cropped for the few frames that matter most.
   * Smoothing the extent means the framing widens as soon as the content starts
   * growing, and the camera is derived from it exactly.
   */
  private updateFraming(dt: number): void {
    if (this.framedNodes.length === 0) return;

    /* In HANDS mode the head is deliberately excluded from the fit.
       Including it would force the camera back far enough to hold a whole
       upper body, which throws away the entire point of the mode — the hands
       want the frame. The ghost head can leave the top of shot; it is a
       landmark, and a landmark half in frame still tells you where the chin is. */
    const handsOnly = this.options.presentation === "hands";
    this.box.makeEmpty();
    for (const node of this.framedNodes) {
      // Hands mode fits the HANDS and the forearms that carry them, and nothing
      // else. Leaving the shoulders in the fit was enough on its own to push the
      // hands to the bottom edge of the panel: shoulder-to-hand is most of an
      // arm's length, so the camera was still framing a torso that is not drawn.
      if (handsOnly && !HAND_FRAMED.has(node.name)) continue;
      node.getWorldPosition(this.point);
      this.box.expandByPoint(this.point);
    }
    // Pad by roughly a hand's length, so a fingertip at the very edge of the set
    // is not on the very edge of the frame.
    this.box.expandByScalar(handsOnly ? 0.09 : 0.11);

    this.box.getCenter(this.centre);
    const size = this.box.getSize(this.point);

    const blend = 1 - Math.exp(-dt / FRAMING_SMOOTH);
    this.smoothCentre.lerp(this.centre, blend);
    this.smoothHeight += (size.y - this.smoothHeight) * blend;
    this.smoothWidth += (Math.max(size.x, size.z) - this.smoothWidth) * blend;

    const height = Math.max(0.45, this.smoothHeight) * FRAMING_MARGIN;
    const width = Math.max(0.45, this.smoothWidth) * FRAMING_MARGIN;

    const vFov = (this.camera.fov * Math.PI) / 180;
    const forHeight = height / 2 / Math.tan(vFov / 2);
    const hFov = 2 * Math.atan(Math.tan(vFov / 2) * this.camera.aspect);
    const forWidth = width / 2 / Math.tan(hFov / 2);
    const distance = Math.max(forHeight, forWidth);

    this.camera.position.set(
      this.smoothCentre.x,
      this.smoothCentre.y,
      this.smoothCentre.z + distance,
    );
    this.camera.lookAt(this.smoothCentre);
  }

  /* -------------------------------------------------------------- misc -- */

  resize(): void {
    const width = this.mount.clientWidth || 480;
    const height = this.mount.clientHeight || 360;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    /* `updateStyle` left at its default of TRUE, deliberately.
       Passing false sets the drawing buffer without setting the canvas's CSS
       size, so on a 2× display the canvas lays out at twice its container and
       the viewer sees one quadrant of the render — which looks exactly like a
       broken camera, and sent me hunting through the framing solver for it. */
    this.renderer.setSize(width, height);
  }

  get report(): BlendReport | null {
    return this.lastReport;
  }

  dispose(): void {
    this.disposed = true;
    this.stop();
    const element = this.renderer.domElement;
    for (const [type, handler] of this.listeners) {
      element.removeEventListener(type, handler as EventListener);
    }
    this.detachSkeleton();
    this.avatar?.dispose();
    this.scene.environment?.dispose();
    this.renderer.dispose();
    if (element.parentNode === this.mount) this.mount.removeChild(element);
  }
}

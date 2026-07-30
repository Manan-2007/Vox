/**
 * The signing avatar — the product's primary output for speech → sign.
 *
 * This replaced a video player. Video was the wrong answer for three reasons
 * that are not about looks: a clip can only be watched from the angle it was
 * filmed at, it cannot be slowed down without going to mush, and every clip is
 * someone's likeness, which is a licensing and consent problem the moment the
 * app leaves one laptop. The avatar is driven by landmark motion extracted from
 * the reference recordings, so no video ships, the sign can be orbited and
 * slowed, and the geometry is the same in every sign.
 *
 * Two sources drive the same rig:
 *   replay — a queue of signs from the motion library (what to sign)
 *   live   — frames from the camera worker (what you are signing)
 * Using one rig for both is deliberate: a learner comparing their own hands to
 * the reference should be looking at the same object, not at a polished avatar
 * next to a wireframe of themselves.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { SignRig } from "../avatar/rig";
import {
  FRAME_FLOATS,
  SignPlayer,
  type PlayerState,
  type QueueItem,
} from "../avatar/signMotion";

interface Props {
  /** Replay queue. When empty, `liveFrame` drives the rig instead. */
  queue?: readonly QueueItem[];
  /** A 140-float avatar-format frame from the camera, for the live view. */
  liveFrame?: Float32Array | null;
  /** Show the torso and head, or crop tight to the hands. */
  body?: boolean;
  /** Mirror the figure, so a learner can copy it directly. */
  mirror?: boolean;
  playbackSpeed?: number;
  loop?: boolean;
  onState?: (state: PlayerState) => void;
  /** Called once with the player instance, so a parent can render controls. */
  onPlayer?: (player: SignPlayer) => void;
  className?: string;
}

/**
 * Framing, in shoulder-width units (one unit = the signer's shoulder span).
 *
 * The camera distance is COMPUTED from these and the viewport aspect rather
 * than hard-coded, because the stage is a different shape on every screen and a
 * fixed distance crops the hands off exactly when the panel is narrow. `height`
 * is the vertical extent that must stay in shot: from a little below the waist
 * to above the head, which is where signs actually happen.
 */
const FRAMING = {
  // Chest-up, which is how sign-language reference footage is framed. Measured
  // from the motion library: the nose sits about 0.67 units above the shoulder
  // line and signs reach roughly 0.95 below it, while an idle hand hangs down at
  // about 1.35. Framing to include that idle hand would shrink the sign itself,
  // so the resting arm is allowed to leave the bottom of frame — exactly as it
  // does in the source recordings.
  body: { centreY: -0.1, height: 2.7, width: 2.5 },
  hands: { centreY: -0.1, height: 1.5, width: 1.5 },
};

export function SignAvatar({
  queue,
  liveFrame,
  body = true,
  mirror = true,
  playbackSpeed = 1,
  loop = false,
  onState,
  onPlayer,
  className = "",
}: Props) {
  const mountRef = useRef<HTMLDivElement>(null);
  const playerRef = useRef<SignPlayer>(null as unknown as SignPlayer);
  if (!playerRef.current) playerRef.current = new SignPlayer();

  // Refs, so the render loop reads current values without being rebuilt.
  const liveRef = useRef<Float32Array | null>(null);
  liveRef.current = liveFrame ?? null;
  const onStateRef = useRef(onState);
  onStateRef.current = onState;
  const mirrorRef = useRef(mirror);
  mirrorRef.current = mirror;

  const [webglFailed, setWebglFailed] = useState(false);

  const player = playerRef.current;
  player.speed = playbackSpeed;
  player.loop = loop;

  const onPlayerRef = useRef(onPlayer);
  onPlayerRef.current = onPlayer;
  useEffect(() => {
    onPlayerRef.current?.(player);
  }, [player]);

  /* ------------------------------------------------------ queue changes -- */
  useEffect(() => {
    if (queue && queue.length) player.setQueue([...queue]);
    else player.clear();
  }, [queue, player]);

  /* --------------------------------------------------------- the scene -- */
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true,
        powerPreference: "high-performance",
      });
    } catch {
      setWebglFailed(true);
      return;
    }

    const size = () => ({
      width: mount.clientWidth || 480,
      height: mount.clientHeight || 360,
    });
    const initial = size();

    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.setSize(initial.width, initial.height);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    mount.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      34,
      initial.width / initial.height,
      0.1,
      100,
    );

    /* Lighting: a soft sky/ground hemisphere for overall form, one warm key with
       a shadow, a cool fill to keep shadow sides from going dead, and a rim to
       separate the figure from the background. */
    scene.add(new THREE.HemisphereLight(0xfdf3e7, 0x2b3a3f, 0.85));
    const key = new THREE.DirectionalLight(0xfff2e0, 2.1);
    key.position.set(2.2, 3.4, 3.0);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    key.shadow.camera.near = 0.5;
    key.shadow.camera.far = 14;
    key.shadow.camera.left = -3;
    key.shadow.camera.right = 3;
    key.shadow.camera.top = 3;
    key.shadow.camera.bottom = -3;
    key.shadow.bias = -0.0012;
    key.shadow.radius = 3;
    scene.add(key);
    const fill = new THREE.DirectionalLight(0x8fd7d0, 0.55);
    fill.position.set(-3, 0.6, 2);
    scene.add(fill);
    const rim = new THREE.DirectionalLight(0xffd7a8, 0.9);
    rim.position.set(-1.4, 1.6, -3);
    scene.add(rim);

    /* Shadow catcher: an invisible plane that only receives the shadow, so the
       figure sits on something without a visible floor. */
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.ShadowMaterial({ color: 0x0b1a1c, opacity: 0.24 }),
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -1.55;
    floor.receiveShadow = true;
    scene.add(floor);

    const rig = new SignRig({ body });
    // Mirroring is applied as a scale on the pivot. Materials are double-sided
    // so the flipped winding order does not punch holes in the figure.
    rig.root.traverse((object) => {
      const mesh = object as THREE.Mesh;
      const material = mesh.material as THREE.Material | undefined;
      if (material && "side" in material) material.side = THREE.DoubleSide;
    });
    const pivot = new THREE.Group();
    pivot.add(rig.root);
    scene.add(pivot);

    /* ------------------------------------------------------------ input -- */
    let dragging = false;
    let dragX = 0;
    let dragY = 0;
    let yaw = 0;
    let pitch = 0;
    let userMoved = false;

    const onPointerDown = (event: PointerEvent) => {
      dragging = true;
      userMoved = true;
      dragX = event.clientX;
      dragY = event.clientY;
      renderer.domElement.setPointerCapture(event.pointerId);
    };
    const onPointerMove = (event: PointerEvent) => {
      if (!dragging) return;
      yaw += (event.clientX - dragX) * 0.008;
      pitch = Math.max(-0.5, Math.min(0.5, pitch + (event.clientY - dragY) * 0.005));
      dragX = event.clientX;
      dragY = event.clientY;
    };
    const onPointerUp = (event: PointerEvent) => {
      dragging = false;
      if (renderer.domElement.hasPointerCapture(event.pointerId)) {
        renderer.domElement.releasePointerCapture(event.pointerId);
      }
    };
    renderer.domElement.addEventListener("pointerdown", onPointerDown);
    renderer.domElement.addEventListener("pointermove", onPointerMove);
    renderer.domElement.addEventListener("pointerup", onPointerUp);
    renderer.domElement.addEventListener("pointercancel", onPointerUp);
    renderer.domElement.style.touchAction = "none";
    renderer.domElement.style.cursor = "grab";

    /* ----------------------------------------------------- render loop -- */
    const frame = new Float32Array(FRAME_FLOATS);
    const framing = body ? FRAMING.body : FRAMING.hands;
    const lookAt = new THREE.Vector3(0, framing.centreY, 0);

    /**
     * Distance at which both the required height and width are in shot.
     * Rotating the figure swings the hands out sideways, so a little headroom is
     * added rather than framing exactly to the bounds.
     */
    const fitDistance = (): number => {
      const vFov = (camera.fov * Math.PI) / 180;
      const forHeight = framing.height / 2 / Math.tan(vFov / 2);
      const hFov = 2 * Math.atan(Math.tan(vFov / 2) * camera.aspect);
      const forWidth = framing.width / 2 / Math.tan(hFov / 2);
      return Math.max(forHeight, forWidth) * 1.06;
    };

    let raf = 0;
    let previous = performance.now();
    let drift = 0;
    let lastReported = -1;

    const render = () => {
      raf = requestAnimationFrame(render);
      const now = performance.now();
      const dt = Math.min(0.1, (now - previous) / 1000);
      previous = now;

      const live = liveRef.current;
      let posed = false;
      if (playerRef.current.items.length > 0) {
        if (playerRef.current.advance(dt, frame)) posed = rig.apply(frame);
        else rig.hide();
      } else if (live) {
        posed = rig.apply(live);
      } else {
        rig.hide();
      }
      if (!posed) rig.hide();

      pivot.scale.x = mirrorRef.current ? -1 : 1;

      // An idle drift sells the third dimension; the moment the user drags, it
      // stops for good, because fighting a user's camera is infuriating.
      if (!userMoved) {
        drift += dt * 0.32;
        pivot.rotation.y = Math.sin(drift) * 0.30;
        pivot.rotation.x = 0;
      } else {
        pivot.rotation.y = yaw;
        pivot.rotation.x = pitch;
      }

      camera.position.set(0, framing.centreY + 0.12, fitDistance());
      camera.lookAt(lookAt);
      renderer.render(scene, camera);

      const state = playerRef.current.state;
      // Report only on change; this runs 60 times a second.
      const fingerprint = state.index * 1000 + Math.round(state.progress * 100);
      if (onStateRef.current && fingerprint !== lastReported) {
        lastReported = fingerprint;
        onStateRef.current(state);
      }
    };
    render();

    const onResize = () => {
      const { width, height } = size();
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };
    const observer = new ResizeObserver(onResize);
    observer.observe(mount);

    return () => {
      cancelAnimationFrame(raf);
      observer.disconnect();
      renderer.domElement.removeEventListener("pointerdown", onPointerDown);
      renderer.domElement.removeEventListener("pointermove", onPointerMove);
      renderer.domElement.removeEventListener("pointerup", onPointerUp);
      renderer.domElement.removeEventListener("pointercancel", onPointerUp);
      rig.dispose();
      floor.geometry.dispose();
      (floor.material as THREE.Material).dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === mount) {
        mount.removeChild(renderer.domElement);
      }
    };
  }, [body]);

  if (webglFailed) {
    return (
      <div className={`avatar avatar--failed ${className}`}>
        <p className="avatar__fallback">
          This browser could not start WebGL, so the 3D signer cannot be shown.
          Recognition and speech still work.
        </p>
      </div>
    );
  }

  return <div ref={mountRef} className={`avatar ${className}`} />;
}

/** Shared playback bar, used by the panels that host an avatar. */
export function AvatarControls({
  player,
  state,
  speed,
  onSpeed,
  loop,
  onLoop,
  mirror,
  onMirror,
}: {
  player: SignPlayer;
  state: PlayerState;
  speed: number;
  onSpeed: (value: number) => void;
  loop: boolean;
  onLoop: (value: boolean) => void;
  mirror: boolean;
  onMirror: (value: boolean) => void;
}) {
  const [, force] = useState(0);
  const rerender = useCallback(() => force((n) => n + 1), []);
  const items = player.items;

  const speeds = useMemo(() => [0.5, 1, 1.5], []);

  return (
    <div className="playback">
      <div className="playback__row">
        <button
          type="button"
          className="iconbtn iconbtn--primary"
          onClick={() => {
            player.toggle();
            rerender();
          }}
          disabled={items.length === 0}
          aria-label={state.playing ? "Pause" : "Play"}
        >
          {state.playing ? "❙❙" : "▶"}
        </button>
        <button
          type="button"
          className="iconbtn"
          onClick={() => {
            player.restart();
            rerender();
          }}
          disabled={items.length === 0}
          aria-label="Play from the beginning"
        >
          ↻
        </button>

        <div className="playback__track" aria-hidden>
          <div
            className="playback__fill"
            style={{ width: `${Math.round(state.progress * 100)}%` }}
          />
        </div>

        <div className="segmented" role="group" aria-label="Playback speed">
          {speeds.map((value) => (
            <button
              key={value}
              type="button"
              className={`segmented__option ${speed === value ? "is-active" : ""}`}
              onClick={() => onSpeed(value)}
            >
              {value}×
            </button>
          ))}
        </div>

        <button
          type="button"
          className={`iconbtn ${loop ? "is-active" : ""}`}
          onClick={() => onLoop(!loop)}
          aria-pressed={loop}
          title="Loop the phrase"
        >
          ⟳
        </button>
        <button
          type="button"
          className={`iconbtn ${mirror ? "is-active" : ""}`}
          onClick={() => onMirror(!mirror)}
          aria-pressed={mirror}
          title={
            mirror
              ? "Mirrored — copy the hand you see on your own side"
              : "Facing you — as another signer would appear"
          }
        >
          ⇋
        </button>
      </div>

      {items.length > 0 && (
        <ol className="playback__words">
          {items.map((item, index) => (
            <li key={`${item.gloss}-${index}`}>
              <button
                type="button"
                className={`wordchip ${index === state.index ? "is-active" : ""} ${
                  item.missing ? "wordchip--missing" : ""
                }`}
                onClick={() => {
                  player.step(index);
                  player.play();
                  rerender();
                }}
                title={
                  item.missing
                    ? `No sign in the library for "${item.label}"`
                    : `Jump to "${item.label}"`
                }
              >
                {item.label}
              </button>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

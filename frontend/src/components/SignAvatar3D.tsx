/**
 * 3D holistic sign avatar — an upper-body skeleton (arms + both hands) rendered
 * in three.js and driven by the SAME 141-float frames the recognizer uses.
 *
 * Two sources feed it:
 *   live   — frames streaming from the camera worker (what you are signing)
 *   replay — frames extracted from an ISL clip (what the sign should look like)
 * so the panel can show the reference motion in 3D while the clip plays, and
 * mirror your own motion the rest of the time.
 *
 * Why this is not SignAvatars: that dataset provides SMPL-X *mesh* parameters
 * (a full rigged body). We only have MediaPipe landmarks, so this is a joint
 * skeleton, not a skinned mesh — see docs/3D-AVATAR.md for what adopting
 * SignAvatars would actually require.
 *
 * Depth: MediaPipe's z is a weak relative estimate, so the avatar reads mostly
 * as 2D-in-3D. It is smoothed and scaled down deliberately rather than
 * presented as true depth.
 */
import { useEffect, useRef } from "react";
import * as THREE from "three";
import {
  FEATURES_PER_HAND,
  HANDS_DIM,
  LANDMARKS_PER_HAND,
  POSE_POINTS,
} from "../landmarks";

/** MediaPipe hand topology (21 points). */
const HAND_BONES: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];
/** Pose block order: nose, L shoulder, R shoulder, L elbow, R elbow. */
const POSE_BONES: [number, number][] = [[1, 2], [1, 3], [2, 4]];

const HAND_COLOUR = [0x38a1d8, 0xd6ae82]; // left block, right block
const POSE_COLOUR = 0x789b7b;
const SMOOTHING = 0.35; // exponential smoothing on incoming frames

interface Props {
  /** Latest raw 141-float frame, or null when nothing is being tracked. */
  frame: Float32Array | null;
  height?: number;
}

export function SignAvatar3D({ frame, height = 200 }: Props) {
  const mountRef = useRef<HTMLDivElement>(null);
  const frameRef = useRef<Float32Array | null>(null);
  const smoothedRef = useRef<Float32Array | null>(null);
  frameRef.current = frame;

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const width = mount.clientWidth || 260;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(42, width / height, 0.1, 100);
    camera.position.set(0, 0, 3.1);

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    } catch {
      return; // no WebGL — the panel simply shows nothing rather than crashing
    }
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.setSize(width, height);
    mount.appendChild(renderer.domElement);

    scene.add(new THREE.AmbientLight(0xffffff, 1.7));
    const key = new THREE.DirectionalLight(0xffffff, 1.1);
    key.position.set(1, 2, 3);
    scene.add(key);

    const root = new THREE.Group();
    // MediaPipe y grows downward and the preview is mirrored, so flip both.
    root.scale.set(-1, -1, 1);
    scene.add(root);

    /* joints: 21 per hand + 5 pose points */
    const jointGeo = new THREE.SphereGeometry(0.035, 10, 10);
    const joints: THREE.Mesh[] = [];
    const makeJoints = (count: number, colour: number, size: number) => {
      const material = new THREE.MeshStandardMaterial({
        color: colour, roughness: 0.45, metalness: 0.05,
      });
      for (let i = 0; i < count; i += 1) {
        const mesh = new THREE.Mesh(jointGeo, material);
        mesh.scale.setScalar(size);
        mesh.visible = false;
        root.add(mesh);
        joints.push(mesh);
      }
    };
    makeJoints(LANDMARKS_PER_HAND, HAND_COLOUR[0], 1);
    makeJoints(LANDMARKS_PER_HAND, HAND_COLOUR[1], 1);
    makeJoints(POSE_POINTS, POSE_COLOUR, 1.5);

    /* bones as line segments, one buffer per group */
    const makeBones = (bones: [number, number][], colour: number, offset: number) => {
      const positions = new Float32Array(bones.length * 6);
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      const lines = new THREE.LineSegments(
        geometry,
        new THREE.LineBasicMaterial({ color: colour, transparent: true, opacity: 0.85 }),
      );
      lines.visible = false;
      root.add(lines);
      return { lines, positions, bones, offset, geometry };
    };
    const groups = [
      makeBones(HAND_BONES, HAND_COLOUR[0], 0),
      makeBones(HAND_BONES, HAND_COLOUR[1], LANDMARKS_PER_HAND),
      makeBones(POSE_BONES, POSE_COLOUR, LANDMARKS_PER_HAND * 2),
    ];

    let raf = 0;
    let spin = 0;

    const render = () => {
      raf = requestAnimationFrame(render);
      const incoming = frameRef.current;

      if (!incoming) {
        joints.forEach((j) => (j.visible = false));
        groups.forEach((g) => (g.lines.visible = false));
        smoothedRef.current = null;
        renderer.render(scene, camera);
        return;
      }

      // smooth: landmark jitter is very visible once rendered as a skeleton
      let smoothed = smoothedRef.current;
      if (!smoothed || smoothed.length !== incoming.length) {
        smoothed = Float32Array.from(incoming);
      } else {
        for (let i = 0; i < incoming.length; i += 1) {
          // never smooth toward zero — that would drag an absent hand across
          // the scene instead of hiding it
          smoothed[i] =
            incoming[i] === 0 ? 0 : smoothed[i] * (1 - SMOOTHING) + incoming[i] * SMOOTHING;
        }
      }
      smoothedRef.current = smoothed;

      // centre on the shoulders so the figure sits still while hands move
      const poseBase = HANDS_DIM;
      const hasPose = smoothed.subarray(poseBase).some((v) => v !== 0);
      const cx = hasPose ? (smoothed[poseBase + 3] + smoothed[poseBase + 6]) / 2 : 0.5;
      const cy = hasPose ? (smoothed[poseBase + 4] + smoothed[poseBase + 7]) / 2 : 0.5;
      const dx = smoothed[poseBase + 3] - smoothed[poseBase + 6];
      const dy = smoothed[poseBase + 4] - smoothed[poseBase + 7];
      const span = hasPose ? Math.max(0.08, Math.hypot(dx, dy)) : 0.25;
      const k = 1 / (span * 3.4); // shoulders ≈ constant on-screen width

      const place = (mesh: THREE.Mesh, x: number, y: number, z: number) => {
        mesh.position.set((x - cx) * k, (y - cy) * k, z * k * 0.35);
      };

      for (let hand = 0; hand < 2; hand += 1) {
        const base = hand * FEATURES_PER_HAND;
        const present = smoothed.subarray(base, base + FEATURES_PER_HAND).some((v) => v !== 0);
        for (let i = 0; i < LANDMARKS_PER_HAND; i += 1) {
          const mesh = joints[hand * LANDMARKS_PER_HAND + i];
          mesh.visible = present;
          if (present) {
            const o = base + i * 3;
            place(mesh, smoothed[o], smoothed[o + 1], smoothed[o + 2]);
          }
        }
      }
      for (let i = 0; i < POSE_POINTS; i += 1) {
        const mesh = joints[LANDMARKS_PER_HAND * 2 + i];
        const o = poseBase + i * 3;
        const present = hasPose && (smoothed[o] !== 0 || smoothed[o + 1] !== 0);
        mesh.visible = present;
        if (present) place(mesh, smoothed[o], smoothed[o + 1], smoothed[o + 2]);
      }

      for (const group of groups) {
        let visible = true;
        for (let b = 0; b < group.bones.length; b += 1) {
          const [a, c] = group.bones[b];
          const ja = joints[group.offset + a];
          const jc = joints[group.offset + c];
          if (!ja?.visible || !jc?.visible) {
            visible = visible && false;
            group.positions.set([0, 0, 0, 0, 0, 0], b * 6);
            continue;
          }
          group.positions.set(
            [ja.position.x, ja.position.y, ja.position.z,
             jc.position.x, jc.position.y, jc.position.z],
            b * 6,
          );
        }
        group.lines.visible = visible;
        group.geometry.attributes.position.needsUpdate = true;
      }

      // a slow orbit sells the third dimension without disorienting
      spin += 0.0045;
      root.rotation.y = Math.sin(spin) * 0.36;

      renderer.render(scene, camera);
    };
    render();

    const onResize = () => {
      const w = mount.clientWidth || width;
      camera.aspect = w / height;
      camera.updateProjectionMatrix();
      renderer.setSize(w, height);
    };
    const observer = new ResizeObserver(onResize);
    observer.observe(mount);

    return () => {
      cancelAnimationFrame(raf);
      observer.disconnect();
      renderer.dispose();
      jointGeo.dispose();
      groups.forEach((g) => g.geometry.dispose());
      if (renderer.domElement.parentNode === mount) {
        mount.removeChild(renderer.domElement);
      }
    };
  }, [height]);

  return <div ref={mountRef} className="avatar3d" style={{ height }} />;
}

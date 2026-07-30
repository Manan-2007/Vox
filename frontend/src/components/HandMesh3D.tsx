/**
 * 3D hand model — actual hands, built as mesh geometry, not a stick skeleton.
 *
 * Each hand is assembled from real volume:
 *   - a palm built as a filled quad-strip across the five MCP knuckles + wrist
 *   - every finger bone as a TAPERED capsule (wider at the knuckle, narrower
 *     at the tip), oriented along the bone and re-oriented each frame
 *   - rounded joint caps so bends read as knuckles instead of hinges
 *
 * Driven by the same 141-float frames the recognizer consumes, so what you see
 * is exactly what the model sees. Two sources:
 *   live   — the camera worker (your own hands)
 *   replay — frames from ml/clip_motion.py (the reference sign)
 *
 * Honest note on depth: MediaPipe's z is a weak per-landmark estimate, not
 * metric depth. Fingers are therefore given real thickness (which reads as 3D)
 * while z displacement is damped, rather than pretending the depth is accurate.
 */
import { useEffect, useRef } from "react";
import * as THREE from "three";
import {
  FEATURES_PER_HAND,
  HANDS_DIM,
  LANDMARKS_PER_HAND,
  POSE_POINTS,
} from "../landmarks";

/** MediaPipe hand topology: [from, to, thickness at from, thickness at to]. */
const FINGER_BONES: [number, number, number, number][] = [
  // thumb — thickest
  [1, 2, 1.00, 0.88], [2, 3, 0.88, 0.78], [3, 4, 0.78, 0.62],
  // index
  [5, 6, 0.92, 0.80], [6, 7, 0.80, 0.70], [7, 8, 0.70, 0.55],
  // middle
  [9, 10, 0.95, 0.83], [10, 11, 0.83, 0.72], [11, 12, 0.72, 0.56],
  // ring
  [13, 14, 0.88, 0.77], [14, 15, 0.77, 0.67], [15, 16, 0.67, 0.53],
  // pinky — thinnest
  [17, 18, 0.78, 0.68], [18, 19, 0.68, 0.60], [19, 20, 0.60, 0.48],
];
/** Palm surface: triangles across wrist + the knuckle line. */
const PALM_TRIANGLES: [number, number, number][] = [
  [0, 1, 5], [0, 5, 9], [0, 9, 13], [0, 13, 17],
  [5, 9, 6], [9, 13, 10], [13, 17, 14],
];
const JOINTS = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19];
const TIPS = [4, 8, 12, 16, 20];

/** Pose block order: nose, L shoulder, R shoulder, L elbow, R elbow. */
const ARM_BONES: [number, number][] = [[1, 2], [1, 3], [2, 4]];

/* Vox palette. Hands are warm skin-like tones drawn from the accent/secondary
   ramp so they belong to the product rather than looking like a debug view. */
const SKIN_LEFT = 0xc9d4bd;   // secondary, lightened
const SKIN_RIGHT = 0xe8c9a3;  // accent, lightened
const BODY_COLOUR = 0x789b7b; // primary
const SMOOTHING = 0.4;
// Bone thickness as a fraction of the hand's own world size, so fingers stay
// proportional at any zoom. Scaling a unit cylinder's X and Z separately would
// make an ELLIPSE, not a taper, so each bone gets one uniform radius.
const BONE_THICKNESS = 0.17;
const Z_DAMP = 0.35;          // MediaPipe z is a weak estimate — damp it

interface Props {
  /** Latest raw 141-float frame, or null when nothing is tracked. */
  frame: Float32Array | null;
  height?: number;
  /** Slowly orbit the view to sell the third dimension. */
  autoRotate?: boolean;
  /** Fit tightly to the hands (replay) instead of the whole body (live). */
  handsOnly?: boolean;
}

export function HandMesh3D({
  frame,
  height = 220,
  autoRotate = true,
  handsOnly = false,
}: Props) {
  const mountRef = useRef<HTMLDivElement>(null);
  const frameRef = useRef<Float32Array | null>(null);
  const smoothedRef = useRef<Float32Array | null>(null);
  frameRef.current = frame;

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const width = mount.clientWidth || 280;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(40, width / height, 0.1, 100);
    camera.position.set(0, 0, handsOnly ? 2.3 : 3.2);

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    } catch {
      return; // no WebGL — render nothing rather than crash
    }
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.setSize(width, height);
    renderer.shadowMap.enabled = false;
    mount.appendChild(renderer.domElement);

    scene.add(new THREE.AmbientLight(0xfff4e6, 1.25));
    const key = new THREE.DirectionalLight(0xffffff, 1.5);
    key.position.set(1.5, 2.5, 3);
    scene.add(key);
    const rim = new THREE.DirectionalLight(0x789b7b, 0.75);
    rim.position.set(-2, 0.5, -1.5);
    scene.add(rim);
    const bounce = new THREE.DirectionalLight(0xd6ae82, 0.4);
    bounce.position.set(0, -2, 1);
    scene.add(bounce);

    const root = new THREE.Group();
    // MediaPipe y grows downward; the preview is mirrored. Flip both.
    root.scale.set(-1, -1, 1);
    scene.add(root);

    /* ---------------------------------------------------------- geometry -- */
    // A unit cylinder along +Y, so each bone only needs position/rotation/scale.
    const boneGeo = new THREE.CylinderGeometry(1, 1, 1, 12, 1, true);
    const jointGeo = new THREE.SphereGeometry(1, 14, 12);

    const skin = (colour: number) =>
      new THREE.MeshStandardMaterial({
        color: colour,
        roughness: 0.62,
        metalness: 0.02,
        emissive: new THREE.Color(colour).multiplyScalar(0.10),
      });

    interface HandParts {
      group: THREE.Group;
      bones: { mesh: THREE.Mesh; from: number; to: number; r0: number; r1: number }[];
      joints: { mesh: THREE.Mesh; index: number; scale: number }[];
      palm: THREE.Mesh;
      palmPositions: Float32Array;
    }

    const buildHand = (colour: number): HandParts => {
      const group = new THREE.Group();
      const material = skin(colour);

      const bones = FINGER_BONES.map(([from, to, r0, r1]) => {
        const mesh = new THREE.Mesh(boneGeo, material);
        group.add(mesh);
        return { mesh, from, to, r0, r1 };
      });

      const joints = [
        ...JOINTS.map((index) => ({ index, scale: 1.0 })),
        ...TIPS.map((index) => ({ index, scale: 0.78 })),
        { index: 0, scale: 1.5 }, // wrist
      ].map(({ index, scale }) => {
        const mesh = new THREE.Mesh(jointGeo, material);
        group.add(mesh);
        return { mesh, index, scale };
      });

      const palmPositions = new Float32Array(PALM_TRIANGLES.length * 9);
      const palmGeo = new THREE.BufferGeometry();
      palmGeo.setAttribute("position", new THREE.BufferAttribute(palmPositions, 3));
      const palm = new THREE.Mesh(
        palmGeo,
        new THREE.MeshStandardMaterial({
          color: colour, roughness: 0.66, metalness: 0.02, side: THREE.DoubleSide,
          emissive: new THREE.Color(colour).multiplyScalar(0.08),
        }),
      );
      group.add(palm);

      group.visible = false;
      root.add(group);
      return { group, bones, joints, palm, palmPositions };
    };

    const hands = [buildHand(SKIN_LEFT), buildHand(SKIN_RIGHT)];

    /* arms + shoulders, so the hands are attached to a body, not floating */
    const bodyMat = new THREE.MeshStandardMaterial({
      color: BODY_COLOUR, roughness: 0.5, metalness: 0.05,
      emissive: new THREE.Color(BODY_COLOUR).multiplyScalar(0.12),
    });
    const armMeshes = ARM_BONES.map(() => {
      const mesh = new THREE.Mesh(boneGeo, bodyMat);
      mesh.visible = false;
      root.add(mesh);
      return mesh;
    });
    const poseJoints = Array.from({ length: POSE_POINTS }, () => {
      const mesh = new THREE.Mesh(jointGeo, bodyMat);
      mesh.visible = false;
      root.add(mesh);
      return mesh;
    });

    /* ------------------------------------------------------------ render -- */
    const a = new THREE.Vector3();
    const b = new THREE.Vector3();
    const mid = new THREE.Vector3();
    const dir = new THREE.Vector3();
    const up = new THREE.Vector3(0, 1, 0);
    const quat = new THREE.Quaternion();

    let raf = 0;
    let spin = 0;

    /** Place a bone spanning a -> b with a uniform radius. */
    const orientBone = (mesh: THREE.Mesh, radius: number) => {
      const length = a.distanceTo(b);
      if (length < 1e-6) {
        mesh.visible = false;
        return;
      }
      mid.addVectors(a, b).multiplyScalar(0.5);
      dir.subVectors(b, a).normalize();
      quat.setFromUnitVectors(up, dir);
      mesh.position.copy(mid);
      mesh.quaternion.copy(quat);
      // unit cylinder: x,z = radius (equal, or it distorts), y = length
      mesh.scale.set(radius, length, radius);
      mesh.visible = true;
    };

    const render = () => {
      raf = requestAnimationFrame(render);
      const incoming = frameRef.current;

      if (!incoming) {
        hands.forEach((h) => (h.group.visible = false));
        armMeshes.forEach((m) => (m.visible = false));
        poseJoints.forEach((m) => (m.visible = false));
        smoothedRef.current = null;
        renderer.render(scene, camera);
        return;
      }

      // Smooth: raw landmark jitter is very visible on a solid mesh.
      let smoothed = smoothedRef.current;
      if (!smoothed || smoothed.length !== incoming.length) {
        smoothed = Float32Array.from(incoming);
      } else {
        for (let i = 0; i < incoming.length; i += 1) {
          // never ease toward zero — that drags an absent hand across the scene
          smoothed[i] =
            incoming[i] === 0
              ? 0
              : smoothed[i] === 0
                ? incoming[i]
                : smoothed[i] * (1 - SMOOTHING) + incoming[i] * SMOOTHING;
        }
      }
      smoothedRef.current = smoothed;

      const poseBase = HANDS_DIM;
      const hasPose = smoothed.subarray(poseBase).some((v) => v !== 0);

      // Frame the view. Live: centre on the body. Replay: centre on the hands
      // so a single sign fills the panel.
      let cx = 0.5;
      let cy = 0.5;
      let span = 0.25;
      if (handsOnly) {
        // Zoom off HAND SIZE, not the bounding box of both hands. A bounding
        // box collapses the view whenever the hands are far apart (which is
        // most two-handed signs) — anchoring on the hand itself keeps a hand
        // the same size on screen no matter where the other one is.
        let sumX = 0, sumY = 0, seen = 0, sizeSum = 0, handCount = 0;
        for (let hand = 0; hand < 2; hand += 1) {
          const base = hand * FEATURES_PER_HAND;
          if (!smoothed.subarray(base, base + FEATURES_PER_HAND).some((v) => v !== 0)) continue;
          for (let i = 0; i < LANDMARKS_PER_HAND; i += 1) {
            sumX += smoothed[base + i * 3];
            sumY += smoothed[base + i * 3 + 1];
            seen += 1;
          }
          sizeSum += Math.hypot(
            smoothed[base + 9 * 3] - smoothed[base],
            smoothed[base + 9 * 3 + 1] - smoothed[base + 1],
          );
          handCount += 1;
        }
        if (seen) {
          cx = sumX / seen;
          cy = sumY / seen;
          // wrist -> middle knuckle is roughly a third of a hand's length
          span = Math.max(0.05, sizeSum / handCount);
        }
      } else if (hasPose) {
        cx = (smoothed[poseBase + 3] + smoothed[poseBase + 6]) / 2;
        cy = (smoothed[poseBase + 4] + smoothed[poseBase + 7]) / 2;
        span = Math.max(
          0.08,
          Math.hypot(
            smoothed[poseBase + 3] - smoothed[poseBase + 6],
            smoothed[poseBase + 4] - smoothed[poseBase + 7],
          ),
        );
      }
      // handsOnly: a hand's wrist->knuckle span becomes ~0.62 world units, so
      // one hand fills roughly a third of the frame and two still fit.
      const k = handsOnly ? 0.62 / span : 1 / (span * 3.4);

      const read = (target: THREE.Vector3, offset: number) => {
        target.set(
          (smoothed![offset] - cx) * k,
          (smoothed![offset + 1] - cy) * k,
          smoothed![offset + 2] * k * Z_DAMP,
        );
      };

      for (let hand = 0; hand < 2; hand += 1) {
        const parts = hands[hand];
        const base = hand * FEATURES_PER_HAND;
        const present = smoothed
          .subarray(base, base + FEATURES_PER_HAND)
          .some((v) => v !== 0);
        parts.group.visible = present;
        if (!present) continue;

        // Thickness is derived in WORLD units from the hand's own size, so it
        // is self-consistent at any zoom: wrist -> middle knuckle is the ruler.
        read(a, base + 0 * 3);
        read(b, base + 9 * 3);
        const unit = Math.max(1e-3, a.distanceTo(b)) * BONE_THICKNESS;

        for (const bone of parts.bones) {
          read(a, base + bone.from * 3);
          read(b, base + bone.to * 3);
          // average the table's two entries: one uniform radius per bone, with
          // thicker bones near the palm and thinner ones at the tips
          orientBone(bone.mesh, unit * ((bone.r0 + bone.r1) / 2));
        }
        for (const joint of parts.joints) {
          read(a, base + joint.index * 3);
          joint.mesh.position.copy(a);
          joint.mesh.scale.setScalar(unit * joint.scale * 0.92);
          joint.mesh.visible = true;
        }

        let p = 0;
        for (const [i0, i1, i2] of PALM_TRIANGLES) {
          for (const index of [i0, i1, i2]) {
            read(a, base + index * 3);
            parts.palmPositions[p] = a.x;
            parts.palmPositions[p + 1] = a.y;
            parts.palmPositions[p + 2] = a.z;
            p += 3;
          }
        }
        parts.palm.geometry.attributes.position.needsUpdate = true;
        parts.palm.geometry.computeVertexNormals();
      }

      // arms + shoulders (live view only — the replay is framed on the hands)
      const showBody = hasPose && !handsOnly;
      // Arms are sized off the shoulder span, the body's own ruler.
      read(a, poseBase + 1 * 3);
      read(b, poseBase + 2 * 3);
      const bodyUnit = Math.max(1e-3, a.distanceTo(b)) * 0.09;
      ARM_BONES.forEach(([from, to], i) => {
        if (!showBody) {
          armMeshes[i].visible = false;
          return;
        }
        read(a, poseBase + from * 3);
        read(b, poseBase + to * 3);
        orientBone(armMeshes[i], bodyUnit);
      });
      poseJoints.forEach((mesh, i) => {
        if (!showBody) {
          mesh.visible = false;
          return;
        }
        read(a, poseBase + i * 3);
        mesh.position.copy(a);
        mesh.scale.setScalar(bodyUnit * (i === 0 ? 2.0 : 1.35));
        mesh.visible = true;
      });

      if (autoRotate) {
        spin += 0.004;
        root.rotation.y = Math.sin(spin) * 0.42;
      }
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
      boneGeo.dispose();
      jointGeo.dispose();
      hands.forEach((h) => h.palm.geometry.dispose());
      if (renderer.domElement.parentNode === mount) {
        mount.removeChild(renderer.domElement);
      }
    };
  }, [height, autoRotate, handsOnly]);

  return <div ref={mountRef} className="handmesh" style={{ height }} />;
}

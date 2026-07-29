/**
 * Camera + MediaPipe HandLandmarker, driving a canvas overlay and emitting one
 * 126-float vector per throttled frame.
 *
 * The detection contract is unchanged from the spike and must stay that way:
 *   - detect on the RAW frame; only the preview is mirrored, in CSS
 *   - throttle to ~15 FPS
 *   - build the vector with buildFrameVector (see src/landmarks.ts)
 *
 * MediaPipe still runs on the main thread. Moving it into a Web Worker is
 * tracked separately; it does not change anything below except where `tick`
 * lives.
 */
import { useEffect, useRef, useState } from "react";
import {
  DrawingUtils,
  FilesetResolver,
  HandLandmarker,
  type HandLandmarkerResult,
} from "@mediapipe/tasks-vision";
import { buildFrameVector } from "../landmarks";

export const TARGET_FPS = 15;
const FRAME_INTERVAL_MS = 1000 / TARGET_FPS;

// Match the preview colours in ml/collect.py so both previews look the same.
const HAND_COLOURS: Record<string, string> = { Left: "#38bdf8", Right: "#fbbf24" };

export type TrackingStatus =
  | "starting"
  | "requesting-camera"
  | "loading-model"
  | "running"
  | "stopped";

export interface HandTracking {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
  status: TrackingStatus;
  error: string | null;
  delegate: "GPU" | "CPU" | null;
  handsVisible: number;
}

/** `onVector` is called with the raw 126-float frame; keep it cheap. */
export function useHandTracking(onVector: (vector: Float32Array) => void): HandTracking {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Held in a ref so a changing callback identity never restarts the camera.
  const onVectorRef = useRef(onVector);
  onVectorRef.current = onVector;

  const [status, setStatus] = useState<TrackingStatus>("starting");
  const [error, setError] = useState<string | null>(null);
  const [delegate, setDelegate] = useState<"GPU" | "CPU" | null>(null);
  const [handsVisible, setHandsVisible] = useState(0);

  useEffect(() => {
    // StrictMode mounts effects twice in dev. Every await re-checks this flag so
    // a torn-down mount never leaves a camera or landmarker alive.
    let disposed = false;
    let stream: MediaStream | null = null;
    let landmarker: HandLandmarker | null = null;
    let raf = 0;

    const start = async () => {
      try {
        setStatus("requesting-camera");
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 1280, height: 720 },
          audio: false,
        });
        if (disposed) return;

        const video = videoRef.current;
        if (!video) return;
        video.srcObject = stream;
        await video.play();
        if (disposed) return;

        setStatus("loading-model");
        const fileset = await FilesetResolver.forVisionTasks("/mediapipe/wasm");
        if (disposed) return;

        const options = {
          baseOptions: { modelAssetPath: "/models/hand_landmarker.task" },
          numHands: 2,
          runningMode: "VIDEO" as const,
        };
        try {
          landmarker = await HandLandmarker.createFromOptions(fileset, {
            ...options,
            baseOptions: { ...options.baseOptions, delegate: "GPU" as const },
          });
          if (!disposed) setDelegate("GPU");
        } catch {
          if (disposed) return;
          landmarker = await HandLandmarker.createFromOptions(fileset, {
            ...options,
            baseOptions: { ...options.baseOptions, delegate: "CPU" as const },
          });
          if (!disposed) setDelegate("CPU");
        }
        if (disposed) return;

        setStatus("running");

        const canvas = canvasRef.current!;
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext("2d")!;
        const drawing = new DrawingUtils(ctx);

        let lastFrame = 0;
        let lastTimestamp = -1;
        let lastHandCount = -1;

        const tick = () => {
          raf = requestAnimationFrame(tick);
          if (disposed || !landmarker) return;

          const now = performance.now();
          if (now - lastFrame < FRAME_INTERVAL_MS) return; // throttle to ~15 FPS
          lastFrame = now;

          const v = videoRef.current;
          if (!v || v.readyState < 2) return;

          // detectForVideo requires strictly increasing timestamps.
          let ts = Math.round(now);
          if (ts <= lastTimestamp) ts = lastTimestamp + 1;
          lastTimestamp = ts;

          // Detection runs on the RAW frame; the preview is mirrored in CSS.
          const result: HandLandmarkerResult = landmarker.detectForVideo(v, ts);

          ctx.clearRect(0, 0, canvas.width, canvas.height);
          result.landmarks.forEach((hand, i) => {
            const label = result.handedness[i]?.[0]?.categoryName ?? "";
            const colour = HAND_COLOURS[label] ?? "#94a3b8";
            drawing.drawConnectors(hand, HandLandmarker.HAND_CONNECTIONS, {
              color: colour,
              lineWidth: 3,
            });
            drawing.drawLandmarks(hand, { color: "#f8fafc", radius: 3 });
          });

          if (result.landmarks.length !== lastHandCount) {
            lastHandCount = result.landmarks.length;
            setHandsVisible(lastHandCount);
          }

          onVectorRef.current(buildFrameVector(result));
        };
        tick();
      } catch (err) {
        if (disposed) return;
        const message = err instanceof Error ? err.message : String(err);
        const denied =
          message.includes("Permission") ||
          message.includes("denied") ||
          message.includes("NotAllowed");
        setError(
          denied
            ? "Camera permission denied — allow access, then reload."
            : `Camera unavailable: ${message}`,
        );
        setStatus("stopped");
      }
    };

    void start();

    return () => {
      disposed = true;
      cancelAnimationFrame(raf);
      landmarker?.close();
      stream?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  return { videoRef, canvasRef, status, error, delegate, handsVisible };
}

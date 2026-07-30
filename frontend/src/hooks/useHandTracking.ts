/**
 * Camera capture on the main thread, MediaPipe detection in a Web Worker (P9).
 *
 * Main thread: getUserMedia, ImageBitmap capture at ~15 FPS, skeleton drawing,
 * UI state. Worker: HandLandmarker + buildFrameVector (imported there, so the
 * 126-float contract with ml/collect.py is unchanged).
 *
 * Backpressure: at most one frame is in flight. If the worker is still busy
 * when the next tick fires, that tick is skipped — latency stays bounded and
 * bitmaps never pile up in the message queue.
 *
 * Detection still runs on the RAW frame; only the preview is mirrored, in CSS.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { DrawingUtils, HandLandmarker } from "@mediapipe/tasks-vision";
import type {
  WorkerInMessage,
  WorkerOutMessage,
} from "../workers/mediapipe.worker";

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
  /** Measured detection results per second (0 until the loop runs). */
  fps: number;
  /** Worker-side inference time for the latest frame, ms. */
  inferMs: number;
  /** Latest raw 141-float frame, for the 3D avatar. */
  frame: Float32Array | null;
  /** Cameras available; labels populate once permission is granted. */
  devices: MediaDeviceInfo[];
  /** Switch camera; pass a deviceId from `devices`. */
  selectCamera: (deviceId: string) => void;
  cameraId: string | null;
  /** Camera on/off. Off releases the device (the OS light goes out). */
  cameraOn: boolean;
  setCameraOn: (on: boolean) => void;
}

export function useHandTracking(onVector: (vector: Float32Array) => void): HandTracking {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Held in a ref so a changing callback identity never restarts the pipeline.
  const onVectorRef = useRef(onVector);
  onVectorRef.current = onVector;

  const workerRef = useRef<Worker | null>(null);
  const drawingRef = useRef<DrawingUtils | null>(null);
  const pendingRef = useRef(false);

  const [workerReady, setWorkerReady] = useState(false);
  const [status, setStatus] = useState<TrackingStatus>("starting");
  const [error, setError] = useState<string | null>(null);
  const [delegate, setDelegate] = useState<"GPU" | "CPU" | null>(null);
  const [handsVisible, setHandsVisible] = useState(0);
  const [frame, setFrame] = useState<Float32Array | null>(null);
  const [fps, setFps] = useState(0);
  const [inferMs, setInferMs] = useState(0);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [cameraId, setCameraId] = useState<string | null>(null);
  const [cameraOn, setCameraOn] = useState(true);

  const fpsWindow = useRef({ start: 0, count: 0 });
  const handCount = useRef(-1);

  /* ------------------------------------------------ worker, created once -- */
  useEffect(() => {
    const worker = new Worker(
      new URL("../workers/mediapipe.worker.ts", import.meta.url),
      { type: "module" },
    );
    workerRef.current = worker;

    // A worker that fails to parse or throws at top level posts no message at
    // all — without this the UI sits on "Loading hand landmarker" forever.
    worker.onerror = (event) => {
      const detail = event.message || "worker failed to start";
      setError(`Hand tracking failed: ${detail}`);
      setStatus("stopped");
    };

    worker.onmessage = (event: MessageEvent<WorkerOutMessage>) => {
      const msg = event.data;
      if (msg.type === "ready") {
        setDelegate(msg.delegate);
        setWorkerReady(true);
        return;
      }
      if (msg.type === "error") {
        setError(`Hand tracking failed: ${msg.message}`);
        setStatus("stopped");
        return;
      }

      // result
      pendingRef.current = false;
      setInferMs(Math.round(msg.inferMs));

      const now = performance.now();
      const w = fpsWindow.current;
      w.count += 1;
      if (now - w.start >= 1000) {
        setFps(Math.round((w.count * 1000) / (now - w.start)));
        w.start = now;
        w.count = 0;
      }

      const canvas = canvasRef.current;
      const drawing = drawingRef.current;
      if (canvas && drawing) {
        const ctx = canvas.getContext("2d")!;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        msg.landmarks.forEach((hand, i) => {
          const label = msg.handedness[i]?.[0]?.categoryName ?? "";
          const colour = HAND_COLOURS[label] ?? "#94a3b8";
          drawing.drawConnectors(hand, HandLandmarker.HAND_CONNECTIONS, {
            color: colour,
            lineWidth: 3,
          });
          drawing.drawLandmarks(hand, { color: "#f8fafc", radius: 3 });
        });
      }

      if (msg.landmarks.length !== handCount.current) {
        handCount.current = msg.landmarks.length;
        setHandsVisible(msg.landmarks.length);
      }

      // vector.buffer is transferred, so hand a copy to the avatar before the
      // socket call consumes it
      setFrame(Float32Array.from(msg.vector));
      onVectorRef.current(msg.vector);
    };

    setStatus("loading-model");
    const init: WorkerInMessage = {
      type: "init",
      wasmPath: `${location.origin}/mediapipe/wasm`,
      modelPath: `${location.origin}/models/hand_landmarker.task`,
      poseModelPath: `${location.origin}/models/pose_landmarker_lite.task`,
    };
    worker.postMessage(init);

    return () => {
      worker.postMessage({ type: "close" } satisfies WorkerInMessage);
      worker.terminate();
      workerRef.current = null;
    };
  }, []);

  /* --------------------------------- camera + capture loop, per cameraId -- */
  useEffect(() => {
    if (!workerReady) return;
    if (!cameraOn) {
      // Camera off: the previous effect's cleanup already stopped the tracks.
      // Clear any prior camera error too — switching off is a deliberate act,
      // not a failure, and a stale "permission denied" here reads as broken.
      setStatus("stopped");
      setHandsVisible(0);
      setFrame(null);
      setFps(0);
      setError(null);
      return;
    }

    let disposed = false;
    let stream: MediaStream | null = null;
    let raf = 0;

    const start = async () => {
      try {
        setStatus("requesting-camera");
        stream = await navigator.mediaDevices.getUserMedia({
          video: cameraId
            ? { deviceId: { exact: cameraId }, width: 1280, height: 720 }
            : { width: 1280, height: 720 },
          audio: false,
        });
        if (disposed) return;

        // Labels are only exposed after permission; refresh the device list.
        navigator.mediaDevices
          .enumerateDevices()
          .then((all) => {
            if (!disposed)
              setDevices(all.filter((d) => d.kind === "videoinput"));
          })
          .catch(() => {});

        const video = videoRef.current;
        if (!video) return;
        video.srcObject = stream;
        await video.play();
        if (disposed) return;

        const canvas = canvasRef.current!;
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        drawingRef.current = new DrawingUtils(canvas.getContext("2d")!);

        setStatus("running");
        setError(null);
        fpsWindow.current = { start: performance.now(), count: 0 };

        let lastFrame = 0;
        let lastTimestamp = -1;

        const tick = () => {
          raf = requestAnimationFrame(tick);
          if (disposed) return;

          const now = performance.now();
          if (now - lastFrame < FRAME_INTERVAL_MS) return; // ~15 FPS
          if (pendingRef.current) return; // worker still busy — skip
          const v = videoRef.current;
          if (!v || v.readyState < 2) return;
          lastFrame = now;

          // detectForVideo requires strictly increasing timestamps.
          let ts = Math.round(now);
          if (ts <= lastTimestamp) ts = lastTimestamp + 1;
          lastTimestamp = ts;

          pendingRef.current = true;
          createImageBitmap(v)
            .then((bitmap) => {
              if (disposed || !workerRef.current) {
                bitmap.close();
                pendingRef.current = false;
                return;
              }
              workerRef.current.postMessage(
                { type: "frame", bitmap, timestamp: ts } satisfies WorkerInMessage,
                [bitmap],
              );
            })
            .catch(() => {
              pendingRef.current = false;
            });
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
      pendingRef.current = false;
      stream?.getTracks().forEach((track) => track.stop());
    };
  }, [workerReady, cameraId, cameraOn]);

  const selectCamera = useCallback((deviceId: string) => {
    setCameraId(deviceId || null);
  }, []);

  return {
    videoRef,
    canvasRef,
    status,
    error,
    delegate,
    handsVisible,
    fps,
    inferMs,
    frame,
    devices,
    selectCamera,
    cameraId,
    cameraOn,
    setCameraOn,
  };
}

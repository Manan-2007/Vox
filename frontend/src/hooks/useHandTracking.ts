/**
 * Camera capture on the main thread, MediaPipe detection in a Web Worker.
 *
 * Main thread: getUserMedia, ImageBitmap capture, overlay drawing, UI state.
 * Worker: both landmarkers, stabilisation, and the 141-float frame vector.
 *
 * ---------------------------------------------------------------------------
 * TWO FRAME RATES, ON PURPOSE
 * ---------------------------------------------------------------------------
 * Detection runs as fast as the machine allows (up to DETECT_FPS). Frames are
 * forwarded to the recogniser at SEND_FPS.
 *
 * They are different numbers because they answer different questions.
 * MediaPipe's VIDEO mode tracks each hand from its previous position, so the
 * shorter the gap between frames the less it has to search and the less often it
 * loses lock — detecting more often makes tracking strictly better. But the
 * recogniser was trained on sequences sampled at 15 FPS, so a 30-frame window is
 * two seconds of signing. Feeding it 30 FPS would hand it one second of signing
 * in the same 30 slots, and every sign would look twice as fast as anything it
 * was trained on.
 *
 * So: track fast, report at the rate training used. The 3D avatar and the
 * overlay use every detected frame, because for them smoother is simply better.
 *
 * Backpressure: at most one frame is in flight. If the worker is still busy when
 * the next tick fires, that tick is skipped — latency stays bounded and bitmaps
 * never pile up in the message queue.
 *
 * Detection runs on the RAW frame; only the preview is mirrored, in CSS.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { FRAME_FLOATS, fromLiveFrame } from "../avatar/signMotion";
import type {
  WorkerInMessage,
  WorkerOutMessage,
} from "../workers/mediapipe.worker";

/** Detection target. Capped by what the worker can actually keep up with. */
export const DETECT_FPS = 30;
/** The rate the recogniser is fed — must match ml/normalize.py's SEQUENCE_LENGTH basis. */
export const SEND_FPS = 15;

const DETECT_INTERVAL_MS = 1000 / DETECT_FPS;
const SEND_INTERVAL_MS = 1000 / SEND_FPS;

/** Camera request. Higher is better for small/distant hands; 720p is the floor. */
const CAPTURE = { width: 1280, height: 720 };

export type TrackingStatus =
  | "starting"
  | "requesting-camera"
  | "loading-model"
  | "running"
  | "stopped";

/** Per-frame honesty about what the tracker can actually see. */
export interface FrameQuality {
  hands: number;
  body: boolean;
  /** Consecutive missed frames per hand block, [left, right]. */
  gaps: [number, number];
  /** Rolling hand motion, normalized units per second. */
  motion: number;
}

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
  /** Latest stabilised 141-float frame — the recogniser's view. */
  frame: Float32Array | null;
  /** Latest 140-float avatar frame, so the live view uses the same 3D rig. */
  avatarFrame: Float32Array | null;
  quality: FrameQuality;
  /** Cameras available; labels populate once permission is granted. */
  devices: MediaDeviceInfo[];
  /** Switch camera; pass a deviceId from `devices`. */
  selectCamera: (deviceId: string) => void;
  cameraId: string | null;
  /** Camera on/off. Off releases the device (the OS light goes out). */
  cameraOn: boolean;
  setCameraOn: (on: boolean) => void;
}

/** MediaPipe hand topology, for the overlay. */
const HAND_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];
/** Left block, right block. Matches the avatar's own two-tone hands. */
const BLOCK_COLOURS = ["#7fd4c1", "#f0b775"];

const IDLE_QUALITY: FrameQuality = {
  hands: 0,
  body: false,
  gaps: [0, 0],
  motion: 0,
};

export function useHandTracking(
  onVector: (vector: Float32Array) => void,
): HandTracking {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Held in a ref so a changing callback identity never restarts the pipeline.
  const onVectorRef = useRef(onVector);
  onVectorRef.current = onVector;

  const workerRef = useRef<Worker | null>(null);
  const pendingRef = useRef(false);
  const lastSentRef = useRef(0);

  const [workerReady, setWorkerReady] = useState(false);
  const [status, setStatus] = useState<TrackingStatus>("starting");
  const [error, setError] = useState<string | null>(null);
  const [delegate, setDelegate] = useState<"GPU" | "CPU" | null>(null);
  const [handsVisible, setHandsVisible] = useState(0);
  const [frame, setFrame] = useState<Float32Array | null>(null);
  const [avatarFrame, setAvatarFrame] = useState<Float32Array | null>(null);
  const [quality, setQuality] = useState<FrameQuality>(IDLE_QUALITY);
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

      drawOverlay(canvasRef.current, msg.landmarks, msg.blocks);

      if (msg.hands !== handCount.current) {
        handCount.current = msg.hands;
        setHandsVisible(msg.hands);
      }
      setQuality({
        hands: msg.hands,
        body: msg.body,
        gaps: msg.gaps,
        motion: msg.motion,
      });

      // vector.buffer is transferred, so take the copies the UI needs before
      // the socket call consumes it.
      setFrame(Float32Array.from(msg.vector));
      setAvatarFrame(
        fromLiveFrame(msg.vector, msg.world, new Float32Array(FRAME_FLOATS)),
      );

      // Down-sample to the rate the recogniser was trained at.
      if (now - lastSentRef.current >= SEND_INTERVAL_MS) {
        lastSentRef.current = now;
        onVectorRef.current(msg.vector);
      }
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
      setAvatarFrame(null);
      setQuality(IDLE_QUALITY);
      setFps(0);
      setError(null);
      return;
    }

    let disposed = false;
    let stream: MediaStream | null = null;
    let raf = 0;

    // A fresh camera means a fresh track: drop the smoother's and the hand
    // assigner's history so the first frames are not blended with the old ones.
    workerRef.current?.postMessage({ type: "reset" } satisfies WorkerInMessage);

    const start = async () => {
      try {
        setStatus("requesting-camera");
        stream = await navigator.mediaDevices.getUserMedia({
          video: cameraId
            ? { deviceId: { exact: cameraId }, ...CAPTURE }
            : CAPTURE,
          audio: false,
        });
        if (disposed) return;

        // Labels are only exposed after permission; refresh the device list.
        navigator.mediaDevices
          .enumerateDevices()
          .then((all) => {
            if (!disposed) setDevices(all.filter((d) => d.kind === "videoinput"));
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

        setStatus("running");
        setError(null);
        fpsWindow.current = { start: performance.now(), count: 0 };

        let lastFrame = 0;
        let lastTimestamp = -1;

        const tick = () => {
          raf = requestAnimationFrame(tick);
          if (disposed) return;

          const now = performance.now();
          if (now - lastFrame < DETECT_INTERVAL_MS) return;
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
    avatarFrame,
    quality,
    devices,
    selectCamera,
    cameraId,
    cameraOn,
    setCameraOn,
  };
}

/** Draw the stabilised skeleton. Coordinates are normalized to the frame. */
function drawOverlay(
  canvas: HTMLCanvasElement | null,
  hands: { x: number; y: number }[][],
  blocks: number[],
): void {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);

  hands.forEach((points, i) => {
    const colour = BLOCK_COLOURS[blocks[i]] ?? "#94a3b8";
    ctx.lineCap = "round";
    ctx.lineWidth = Math.max(2, width / 320);
    ctx.strokeStyle = colour;
    ctx.shadowColor = colour;
    ctx.shadowBlur = 10;

    ctx.beginPath();
    for (const [from, to] of HAND_CONNECTIONS) {
      const a = points[from];
      const b = points[to];
      if (!a || !b) continue;
      ctx.moveTo(a.x * width, a.y * height);
      ctx.lineTo(b.x * width, b.y * height);
    }
    ctx.stroke();

    ctx.shadowBlur = 0;
    ctx.fillStyle = "#ffffff";
    const radius = Math.max(1.5, width / 480);
    for (const point of points) {
      ctx.beginPath();
      ctx.arc(point.x * width, point.y * height, radius, 0, Math.PI * 2);
      ctx.fill();
    }
  });
}

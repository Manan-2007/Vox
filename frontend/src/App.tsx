/**
 * Vox spike page — webcam in, recognized word out. Deliberately ugly.
 *
 * MediaPipe runs on the main thread here. Moving it to a Web Worker is a
 * Milestone 1 job; for the spike the only question is whether recognition works
 * at all.
 */
import { useEffect, useRef, useState } from "react";
import {
  DrawingUtils,
  FilesetResolver,
  HandLandmarker,
  type HandLandmarkerResult,
} from "@mediapipe/tasks-vision";
import { buildFrameVector } from "./landmarks";

const TARGET_FPS = 15;
const FRAME_INTERVAL_MS = 1000 / TARGET_FPS;
const WS_URL =
  (import.meta.env.VITE_VOX_WS as string | undefined) ?? "ws://localhost:8000/ws";

// Match the preview colours in ml/collect.py, so the two look the same.
const HAND_COLOURS: Record<string, string> = { Left: "#00b0ff", Right: "#ffc800" };

type Live = { top: string; confidence: number; stableFor: number };

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [status, setStatus] = useState("starting up");
  const [error, setError] = useState<string | null>(null);
  const [delegate, setDelegate] = useState<"GPU" | "CPU" | null>(null);
  const [socket, setSocket] = useState<"connecting" | "open" | "closed">("connecting");
  const [word, setWord] = useState<{ word: string; confidence: number } | null>(null);
  const [live, setLive] = useState<Live | null>(null);
  const [buffered, setBuffered] = useState<{ have: number; need: number } | null>(null);

  useEffect(() => {
    // StrictMode mounts effects twice in dev. Everything below checks this flag
    // after each await so a torn-down mount never leaves a camera or socket on.
    let disposed = false;
    let stream: MediaStream | null = null;
    let landmarker: HandLandmarker | null = null;
    let ws: WebSocket | null = null;
    let raf = 0;
    let reconnectTimer: number | undefined;

    const connect = () => {
      if (disposed) return;
      setSocket("connecting");
      ws = new WebSocket(WS_URL);
      ws.onopen = () => !disposed && setSocket("open");
      ws.onclose = () => {
        if (disposed) return;
        setSocket("closed");
        // The backend is often started after the page; retry quietly.
        reconnectTimer = window.setTimeout(connect, 2000);
      };
      ws.onmessage = (event) => {
        if (disposed) return;
        const msg = JSON.parse(event.data as string);
        if (msg.word) {
          setWord({ word: msg.word, confidence: msg.confidence });
          setBuffered(null);
        } else if (msg.error) {
          setError(`backend: ${msg.error}`);
        } else if (msg.top !== undefined) {
          setLive({ top: msg.top, confidence: msg.confidence, stableFor: msg.stable_for });
          setBuffered(null);
        } else if (msg.buffered !== undefined) {
          setBuffered({ have: msg.buffered, need: msg.needed });
          setLive(null);
        }
      };
    };

    const start = async () => {
      try {
        setStatus("requesting camera");
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480 },
          audio: false,
        });
        if (disposed) return;

        const video = videoRef.current;
        if (!video) return;
        video.srcObject = stream;
        await video.play();
        if (disposed) return;

        setStatus("loading hand landmarker");
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

        connect();
        setStatus("running");

        const canvas = canvasRef.current!;
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext("2d")!;
        const drawing = new DrawingUtils(ctx);

        let lastFrame = 0;
        let lastTimestamp = -1;

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
            const colour = HAND_COLOURS[label] ?? "#bbbbbb";
            drawing.drawConnectors(hand, HandLandmarker.HAND_CONNECTIONS, {
              color: colour,
              lineWidth: 3,
            });
            drawing.drawLandmarks(hand, { color: "#ffffff", radius: 3 });
          });

          const vector = buildFrameVector(result);
          if (ws?.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ landmarks: Array.from(vector) }));
          }
        };
        tick();
      } catch (err) {
        if (disposed) return;
        const message = err instanceof Error ? err.message : String(err);
        setError(
          message.includes("Permission") || message.includes("denied")
            ? "Camera permission denied — allow it in the browser, then reload."
            : message,
        );
        setStatus("stopped");
      }
    };

    void start();

    return () => {
      disposed = true;
      cancelAnimationFrame(raf);
      window.clearTimeout(reconnectTimer);
      if (ws) {
        ws.onclose = null; // don't schedule a reconnect on our own teardown
        ws.close();
      }
      landmarker?.close();
      stream?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  return (
    <main style={{ fontFamily: "system-ui, sans-serif", padding: 16, textAlign: "center" }}>
      <h1 style={{ fontSize: 18, fontWeight: 600 }}>Vox — spike</h1>

      {error && (
        <p style={{ color: "#c00", maxWidth: 640, margin: "8px auto" }}>{error}</p>
      )}

      {/* Mirrored for display only — the detector still sees the raw frame. */}
      <div
        style={{
          position: "relative",
          width: 640,
          maxWidth: "100%",
          margin: "0 auto",
          transform: "scaleX(-1)",
        }}
      >
        <video ref={videoRef} playsInline muted style={{ width: "100%", display: "block" }} />
        <canvas
          ref={canvasRef}
          style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}
        />
      </div>

      <div style={{ minHeight: 96, marginTop: 16 }}>
        <div style={{ fontSize: 64, fontWeight: 700, lineHeight: 1.1 }}>
          {word?.word ?? "—"}
        </div>
        <div style={{ fontSize: 16, color: "#666" }}>
          {word ? `confidence ${word.confidence.toFixed(3)}` : "no word yet"}
        </div>
      </div>

      <p style={{ fontSize: 13, color: "#666", marginTop: 12 }}>
        {status} · {delegate ?? "…"} · socket {socket} · {TARGET_FPS} FPS
        {buffered && ` · buffering ${buffered.have}/${buffered.need}`}
        {live &&
          ` · top ${live.top} ${live.confidence.toFixed(2)} (stable ${live.stableFor})`}
      </p>
    </main>
  );
}

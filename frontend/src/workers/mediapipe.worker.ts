/**
 * MediaPipe HandLandmarker in a Web Worker (P9).
 *
 * The main thread owns the camera and the UI; this worker owns detection. It
 * receives ImageBitmap frames, runs the landmarker, and posts back the
 * 126-float vector plus the landmarks needed to draw the skeleton.
 *
 * The vector is built with the SAME buildFrameVector as before — the module is
 * imported, not copied, so the byte-for-byte contract with ml/collect.py is
 * untouched by the move off the main thread.
 *
 * Module worker + a wasm-loader shim. tasks-vision loads its glue via
 * importScripts, which module workers do not have, so createFromOptions would
 * fail with "ModuleFactory not set." A classic worker has importScripts, but
 * Vite only honours worker.format on BUILD — its dev server serves workers as
 * ESM regardless, so classic works in production and breaks in dev. Hence the
 * shim below, which works in both.
 */
/// <reference lib="webworker" />
import {
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
  type HandLandmarkerResult,
  type PoseLandmarkerResult,
} from "@mediapipe/tasks-vision";
import { buildFrameVector } from "../landmarks";

export interface InitMessage {
  type: "init";
  wasmPath: string;
  modelPath: string;
  poseModelPath: string;
}
export interface FrameMessage {
  type: "frame";
  bitmap: ImageBitmap;
  timestamp: number;
}
export interface CloseMessage {
  type: "close";
}
export type WorkerInMessage = InitMessage | FrameMessage | CloseMessage;

export interface ReadyMessage {
  type: "ready";
  delegate: "GPU" | "CPU";
}
export interface ResultMessage {
  type: "result";
  timestamp: number;
  inferMs: number;
  vector: Float32Array;
  /** Plain-object copies, structured-clone safe, for skeleton drawing. */
  landmarks: { x: number; y: number; z: number; visibility: number }[][];
  handedness: { categoryName: string; score: number }[][];
}
export interface ErrorMessage {
  type: "error";
  message: string;
}
export type WorkerOutMessage = ReadyMessage | ResultMessage | ErrorMessage;

let landmarker: HandLandmarker | null = null;
let poseLandmarker: PoseLandmarker | null = null;

const post = (msg: WorkerOutMessage, transfer: Transferable[] = []) =>
  (self as unknown as DedicatedWorkerGlobalScope).postMessage(msg, transfer);

async function init(msg: InitMessage) {
  const fileset = await FilesetResolver.forVisionTasks(msg.wasmPath);

  // Fetch the exact loader the resolver chose (simd / nosimd) and evaluate it
  // into worker scope, where it registers self.ModuleFactory the way
  // importScripts would have.
  //
  // Each createFromOptions CONSUMES the factory, and it is not reliably
  // removed from globalThis afterwards — so prime UNCONDITIONALLY (deleting
  // any stale value first) before every task. Guarding with
  // `if (!("ModuleFactory" in self))` silently skips the re-prime and the
  // second task fails, taking hand tracking down with it.
  const loaderSource = await (await fetch(fileset.wasmLoaderPath)).text();
  const primeModuleFactory = () => {
    try {
      delete (self as unknown as Record<string, unknown>).ModuleFactory;
    } catch {
      /* non-configurable in some engines; the eval below still overwrites it */
    }
    (0, eval)(loaderSource);
  };

  const base = {
    baseOptions: { modelAssetPath: msg.modelPath },
    numHands: 2,
    runningMode: "VIDEO" as const,
  };
  const poseBase = {
    baseOptions: { modelAssetPath: msg.poseModelPath },
    numPoses: 1,
    runningMode: "VIDEO" as const,
  };

  const createBoth = async (delegate: "GPU" | "CPU") => {
    primeModuleFactory();
    const hands = await HandLandmarker.createFromOptions(fileset, {
      ...base,
      baseOptions: { ...base.baseOptions, delegate },
    });
    primeModuleFactory();
    const pose = await PoseLandmarker.createFromOptions(fileset, {
      ...poseBase,
      baseOptions: { ...poseBase.baseOptions, delegate },
    });
    return [hands, pose] as const;
  };

  let delegate: "GPU" | "CPU" = "GPU";
  try {
    [landmarker, poseLandmarker] = await createBoth("GPU");
  } catch {
    delegate = "CPU";
    landmarker?.close();
    landmarker = null;
    [landmarker, poseLandmarker] = await createBoth("CPU");
  }
  post({ type: "ready", delegate });
}

function detect(msg: FrameMessage) {
  if (!landmarker) {
    msg.bitmap.close();
    return;
  }
  const started = performance.now();
  const result: HandLandmarkerResult = landmarker.detectForVideo(
    msg.bitmap,
    msg.timestamp,
  );
  // The pose block is the body anchor normalization needs — without it every
  // frame normalizes to zeros and the model sees nothing.
  const pose: PoseLandmarkerResult | null = poseLandmarker
    ? poseLandmarker.detectForVideo(msg.bitmap, msg.timestamp)
    : null;
  msg.bitmap.close();

  const vector = buildFrameVector(result, pose);
  post(
    {
      type: "result",
      timestamp: msg.timestamp,
      inferMs: performance.now() - started,
      vector,
      landmarks: result.landmarks.map((hand) =>
        hand.map((p) => ({ x: p.x, y: p.y, z: p.z, visibility: p.visibility ?? 1 })),
      ),
      handedness: result.handedness.map((cats) =>
        cats.length
          ? [{ categoryName: cats[0].categoryName, score: cats[0].score }]
          : [],
      ),
    },
    [vector.buffer],
  );
}

self.onmessage = (event: MessageEvent<WorkerInMessage>) => {
  const msg = event.data;
  try {
    if (msg.type === "init") void init(msg).catch((err) =>
      post({ type: "error", message: err instanceof Error ? err.message : String(err) }),
    );
    else if (msg.type === "frame") detect(msg);
    else if (msg.type === "close") {
      landmarker?.close();
      poseLandmarker?.close();
      landmarker = null;
      poseLandmarker = null;
    }
  } catch (err) {
    post({
      type: "error",
      message: err instanceof Error ? err.message : String(err),
    });
  }
};

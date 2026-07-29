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
 */
/// <reference lib="webworker" />
import {
  FilesetResolver,
  HandLandmarker,
  type HandLandmarkerResult,
} from "@mediapipe/tasks-vision";
import { buildFrameVector } from "../landmarks";

export interface InitMessage {
  type: "init";
  wasmPath: string;
  modelPath: string;
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

const post = (msg: WorkerOutMessage, transfer: Transferable[] = []) =>
  (self as unknown as DedicatedWorkerGlobalScope).postMessage(msg, transfer);

async function init(msg: InitMessage) {
  const fileset = await FilesetResolver.forVisionTasks(msg.wasmPath);

  // tasks-vision loads its wasm glue with importScripts, which module workers
  // don't have — createFromOptions then dies with "ModuleFactory not set."
  // Load the exact loader the resolver selected (simd/nosimd) ourselves.
  // Indirect eval runs it in the worker's global scope, where the glue
  // registers self.ModuleFactory just as importScripts would have.
  if (!("ModuleFactory" in self)) {
    const loaderSource = await (await fetch(fileset.wasmLoaderPath)).text();
    (0, eval)(loaderSource);
  }
  const base = {
    baseOptions: { modelAssetPath: msg.modelPath },
    numHands: 2,
    runningMode: "VIDEO" as const,
  };
  let delegate: "GPU" | "CPU" = "GPU";
  try {
    landmarker = await HandLandmarker.createFromOptions(fileset, {
      ...base,
      baseOptions: { ...base.baseOptions, delegate: "GPU" },
    });
  } catch {
    delegate = "CPU";
    landmarker = await HandLandmarker.createFromOptions(fileset, {
      ...base,
      baseOptions: { ...base.baseOptions, delegate: "CPU" },
    });
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
  msg.bitmap.close();

  const vector = buildFrameVector(result);
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
      landmarker = null;
    }
  } catch (err) {
    post({
      type: "error",
      message: err instanceof Error ? err.message : String(err),
    });
  }
};

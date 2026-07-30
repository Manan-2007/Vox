/**
 * MediaPipe hand + pose detection in a Web Worker.
 *
 * The main thread owns the camera and the UI; this worker owns detection and
 * stabilisation. It receives ImageBitmap frames, runs both landmarkers, and
 * posts back the 141-float frame vector plus what the UI needs to draw.
 *
 * ---------------------------------------------------------------------------
 * WHY THE DETECTOR IS CONFIGURED THE WAY IT IS
 * ---------------------------------------------------------------------------
 * MediaPipe's defaults (0.5 for all three confidences) are tuned for photos of
 * people holding a hand up, not for signing. Signing breaks two of their
 * assumptions: the hands move fast enough to motion-blur at webcam shutter
 * speeds, and they spend much of their time overlapping each other or the face.
 * At 0.5 the tracker drops lock several times per sign, and every drop is a
 * hole in the 30-frame window the recogniser reads.
 *
 * Measured on the ISLRTC dictionary clip for "eat" (45 frames at 15 FPS, hands
 * up for 20 of them), detection inside the signing segment:
 *
 *     confidences 0.5 / 0.5 / 0.5     19 frames
 *     confidences 0.3 / 0.3 / 0.3     21 frames
 *     confidences 0.2 / 0.2 / 0.2     23 frames
 *
 * Lowering the thresholds admits some weak detections, which is the right
 * trade: a slightly wrong hand position is a small error, a missing hand is a
 * zeroed 63-float block and a discontinuity. Frame rate helps more than any
 * threshold — see FPS in useHandTracking — because MediaPipe's VIDEO mode tracks
 * from the previous frame and short inter-frame motion is easier to follow.
 *
 * Module worker + a wasm-loader shim: tasks-vision loads its glue via
 * importScripts, which module workers do not have, so createFromOptions would
 * fail with "ModuleFactory not set." A classic worker has importScripts, but
 * Vite only honours worker.format on BUILD — its dev server serves workers as
 * ESM regardless. Hence the shim below, which works in both.
 */
/// <reference lib="webworker" />
import {
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
  type HandLandmarkerResult,
  type PoseLandmarkerResult,
} from "@mediapipe/tasks-vision";
import { POSE_LANDMARK_INDICES } from "../landmarks";
import {
  aspectScale,
  assembleFrame,
  FrameSmoother,
  HandAssigner,
  MotionEnergy,
} from "../tracking";

/** See the module docstring for why these are well below MediaPipe's defaults. */
const DETECTION_CONFIDENCE = 0.3;
const PRESENCE_CONFIDENCE = 0.3;
const TRACKING_CONFIDENCE = 0.3;

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
export interface ResetMessage {
  type: "reset";
}
export interface CloseMessage {
  type: "close";
}
export type WorkerInMessage =
  | InitMessage
  | FrameMessage
  | ResetMessage
  | CloseMessage;

export interface ReadyMessage {
  type: "ready";
  delegate: "GPU" | "CPU";
}
export interface ResultMessage {
  type: "result";
  timestamp: number;
  inferMs: number;
  /** Stabilised 141-float frame — what the recogniser reads. */
  vector: Float32Array;
  /**
   * 126 floats: 2 hands x 21 metric world landmarks (x, y, z), same block order
   * as `vector`. Image-space landmarks place a hand; these carry its real 3D
   * shape, which is what the avatar is built from. See ml/build_motion.py.
   */
  world: Float32Array;
  /** Hands actually resolved this frame, 0-2. */
  hands: number;
  /** True when the pose block carries a usable shoulder anchor. */
  body: boolean;
  /** Consecutive frames each block has been missing, [left, right]. */
  gaps: [number, number];
  /** Rolling hand-motion magnitude, normalized units per second. */
  motion: number;
  /** Plain-object copies, structured-clone safe, for the 2D overlay. */
  landmarks: { x: number; y: number }[][];
  /** Block index per entry of `landmarks`: 0 = left, 1 = right. */
  blocks: number[];
}
export interface ErrorMessage {
  type: "error";
  message: string;
}
export type WorkerOutMessage = ReadyMessage | ResultMessage | ErrorMessage;

let landmarker: HandLandmarker | null = null;
let poseLandmarker: PoseLandmarker | null = null;

const smoother = new FrameSmoother();
const assigner = new HandAssigner();
const motion = new MotionEnergy();

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

  const createBoth = async (delegate: "GPU" | "CPU") => {
    primeModuleFactory();
    const hands = await HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: msg.modelPath, delegate },
      numHands: 2,
      runningMode: "VIDEO",
      minHandDetectionConfidence: DETECTION_CONFIDENCE,
      minHandPresenceConfidence: PRESENCE_CONFIDENCE,
      minTrackingConfidence: TRACKING_CONFIDENCE,
    });
    primeModuleFactory();
    const pose = await PoseLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: msg.poseModelPath, delegate },
      numPoses: 1,
      runningMode: "VIDEO",
      minPoseDetectionConfidence: DETECTION_CONFIDENCE,
      minPosePresenceConfidence: PRESENCE_CONFIDENCE,
      minTrackingConfidence: TRACKING_CONFIDENCE,
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
  // Read the frame's shape before it is consumed: the aspect correction below
  // needs it, and `close()` makes the bitmap unreadable.
  const xScale = aspectScale(msg.bitmap.width, msg.bitmap.height);
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

  // Assign to Left/Right blocks with continuity, THEN build the vector, THEN
  // smooth. Order matters: smoothing a block whose contents just swapped hands
  // would blend two different hands together.
  const assigned = assigner.assign(
    result.landmarks ?? [],
    result.handedness ?? [],
    result.worldLandmarks ?? [],
  );
  const raw = assembleFrame(
    assigned,
    pose?.landmarks?.[0],
    POSE_LANDMARK_INDICES,
    xScale,
  );
  const vector = smoother.smooth(raw, msg.timestamp);
  const energy = motion.update(vector, msg.timestamp);

  // World landmarks ride along with their own hand, so a swap decided by the
  // assigner moves both representations together.
  const world = new Float32Array(126);
  for (let block = 0; block < 2; block += 1) {
    const points = assigned.world[block];
    if (!points) continue;
    for (let i = 0; i < 21 && i < points.length; i += 1) {
      const base = block * 63 + i * 3;
      world[base] = points[i].x;
      world[base + 1] = points[i].y;
      world[base + 2] = points[i].z;
    }
  }

  // Draw from the STABILISED vector, so the overlay shows what the model reads
  // rather than a second, differently-jittering version of the same hands.
  const landmarks: { x: number; y: number }[][] = [];
  const blocks: number[] = [];
  for (let block = 0; block < 2; block += 1) {
    if (!assigned.image[block]) continue;
    const base = block * 63;
    const points: { x: number; y: number }[] = [];
    for (let i = 0; i < 21; i += 1) {
      points.push({ x: vector[base + i * 3], y: vector[base + i * 3 + 1] });
    }
    landmarks.push(points);
    blocks.push(block);
  }

  post(
    {
      type: "result",
      timestamp: msg.timestamp,
      inferMs: performance.now() - started,
      vector,
      world,
      hands: landmarks.length,
      body: vector.subarray(126).some((v) => v !== 0),
      gaps: [assigner.gapFor(0), assigner.gapFor(1)],
      motion: energy,
      landmarks,
      blocks,
    },
    [vector.buffer, world.buffer],
  );
}

self.onmessage = (event: MessageEvent<WorkerInMessage>) => {
  const msg = event.data;
  try {
    if (msg.type === "init")
      void init(msg).catch((err) =>
        post({
          type: "error",
          message: err instanceof Error ? err.message : String(err),
        }),
      );
    else if (msg.type === "frame") detect(msg);
    else if (msg.type === "reset") {
      // Camera switched or restarted: the old track is not evidence any more.
      smoother.reset();
      assigner.reset();
      motion.reset();
    } else if (msg.type === "close") {
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

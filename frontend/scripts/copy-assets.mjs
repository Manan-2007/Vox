/**
 * Stage MediaPipe assets into public/ before dev/build.
 *
 * Runs automatically via the `predev` / `prebuild` npm scripts, so a fresh
 * clone works without a manual step. Both directories are gitignored — the
 * wasm is ~42 MB and the model ~7.8 MB.
 *
 * The hand landmarker model is copied from ml/models/ when it exists, so the
 * browser and ml/collect.py run the BYTE-IDENTICAL model file. Different
 * variants (float16 vs float32) emit slightly different landmarks, which would
 * put collection and inference back out of sync.
 */
import { existsSync, mkdirSync, copyFileSync, readdirSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const FRONTEND = dirname(HERE);
const REPO = dirname(FRONTEND);

const WASM_SRC = join(FRONTEND, "node_modules/@mediapipe/tasks-vision/wasm");
const WASM_DST = join(FRONTEND, "public/mediapipe/wasm");
const MODEL_SRC = join(REPO, "ml/models/hand_landmarker.task");
const MODEL_DST = join(FRONTEND, "public/models/hand_landmarker.task");
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/" +
  "hand_landmarker/float16/1/hand_landmarker.task";

mkdirSync(WASM_DST, { recursive: true });
mkdirSync(dirname(MODEL_DST), { recursive: true });

if (!existsSync(WASM_SRC)) {
  console.error("@mediapipe/tasks-vision is not installed — run: npm install");
  process.exit(1);
}

let copied = 0;
for (const name of readdirSync(WASM_SRC)) {
  const from = join(WASM_SRC, name);
  const to = join(WASM_DST, name);
  if (!existsSync(to) || statSync(from).size !== statSync(to).size) {
    copyFileSync(from, to);
    copied += 1;
  }
}
console.log(`mediapipe wasm: ${copied} file(s) copied, ${readdirSync(WASM_DST).length} present`);

if (existsSync(MODEL_DST)) {
  console.log("hand_landmarker.task: already staged");
} else if (existsSync(MODEL_SRC)) {
  copyFileSync(MODEL_SRC, MODEL_DST);
  console.log("hand_landmarker.task: copied from ml/models (matches collect.py)");
} else {
  console.log(`hand_landmarker.task: not in ml/models, downloading from ${MODEL_URL}`);
  const res = await fetch(MODEL_URL);
  if (!res.ok) {
    console.error(`download failed: ${res.status} ${res.statusText}`);
    process.exit(1);
  }
  const { writeFileSync } = await import("node:fs");
  writeFileSync(MODEL_DST, Buffer.from(await res.arrayBuffer()));
  console.log("hand_landmarker.task: downloaded");
}

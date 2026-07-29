# Vox frontend

React + Vite + TypeScript. See the [repo README](../README.md) for setup and
[ARCHITECTURE.md](../ARCHITECTURE.md) for the design.

- `npm run dev` — dev server on :5173 (stages MediaPipe wasm + model first)
- `npm run build` — typecheck + production build
- `src/landmarks.ts` — the 126-float frame contract; mirrors `ml/collect.py`
  byte-for-byte. Change both together or not at all.
- `src/workers/mediapipe.worker.ts` — HandLandmarker runs here, off the main
  thread.

/** Left column: webcam, skeleton overlay, live confidence, current word, FPS. */
import { SignAvatar3D } from "./SignAvatar3D";
import type { HandTracking } from "../hooks/useHandTracking";
import type { LiveGuess } from "../hooks/useVoxSocket";

interface Props {
  tracking: HandTracking;
  live: LiveGuess | null;
  buffered: { have: number; need: number } | null;
  latestWord: { text: string; confidence: number } | null;
  threshold: number;
  /** What the model is considering right now — the honest view. */
  top3: { word: string; confidence: number }[];
  /** Whether the current frame can be used at all. */
  quality: { hands: number; body: boolean; usable: boolean } | null;
}

const STATUS_LABEL: Record<string, string> = {
  starting: "Starting up",
  "requesting-camera": "Requesting camera",
  "loading-model": "Loading hand landmarker",
  running: "Tracking",
  stopped: "Stopped",
};

export function CameraPanel({
  tracking, live, buffered, latestWord, threshold, top3, quality,
}: Props) {
  const { videoRef, canvasRef, status, error, delegate, handsVisible, fps, inferMs,
          cameraOn, setCameraOn } = tracking;
  const confidence = live?.confidence ?? null;

  // One clear sentence about what is blocking recognition, if anything.
  const hint = !cameraOn
    ? "Camera is off."
    : quality && !quality.body
      ? "Move back a little — your shoulders need to be in frame."
      : quality && quality.hands === 0
        ? "No hands detected — raise your hands into view."
        : null;

  return (
    <section className="panel panel--camera" aria-label="Camera and hand tracking">
      <header className="panel__head">
        <h2 className="panel__title">Camera</h2>
        <div className="panel__head-actions">
          <span className={`badge badge--${status === "running" ? "live" : "idle"}`}>
            {STATUS_LABEL[status] ?? status}
            {delegate && status === "running" ? ` · ${delegate}` : ""}
          </span>
          <button
            type="button"
            className={`button button--icon ${cameraOn ? "" : "button--off"}`}
            onClick={() => setCameraOn(!cameraOn)}
            aria-pressed={cameraOn}
            title={cameraOn ? "Turn the camera off (releases the device)" : "Turn the camera on"}
          >
            {cameraOn ? "◉ Camera on" : "○ Camera off"}
          </button>
        </div>
      </header>

      <div className="panel__body panel__body--flush">
        {/* Mirrored for display only — the detector still sees the raw frame. */}
        <div className="stage">
          <video ref={videoRef} className="stage__video" playsInline muted />
          <canvas ref={canvasRef} className="stage__overlay" />

          {cameraOn && error && (
            <div className="stage__cover">
              <p className="stage__cover-title">Camera unavailable</p>
              <p className="stage__cover-body">{error}</p>
            </div>
          )}

          {!cameraOn && (
            <div className="stage__cover">
              <p className="stage__cover-title">Camera off</p>
              <p className="stage__cover-body">
                The camera is released — the indicator light is out. Turn it back
                on to resume signing.
              </p>
            </div>
          )}

          {cameraOn && !error && status !== "running" && (
            <div className="stage__cover">
              <p className="stage__cover-title">{STATUS_LABEL[status] ?? status}</p>
            </div>
          )}

          {status === "running" && hint && (
            <div className="stage__hint">{hint}</div>
          )}

          {status === "running" && (
            <>
              <div className="stage__hands">
                {handsVisible === 0
                  ? "No hands detected"
                  : `${handsVisible} hand${handsVisible > 1 ? "s" : ""}`}
              </div>
              <div className="stage__fps" title="detections per second · worker inference time">
                {fps} fps · {inferMs} ms
              </div>
            </>
          )}
        </div>
      </div>

      {top3.length > 0 && (
        <div className="top3" aria-label="What the model is considering">
          {top3.map((c, i) => (
            <div key={c.word} className={`top3__row ${i === 0 ? "top3__row--lead" : ""}`}>
              <span className="top3__word">{c.word}</span>
              <span className="top3__track">
                <span
                  className="top3__fill"
                  style={{ width: `${Math.max(2, Math.round(c.confidence * 100))}%` }}
                />
              </span>
              <span className="top3__pct">{Math.round(c.confidence * 100)}%</span>
            </div>
          ))}
        </div>
      )}

      <div className="avatar3d-strip">
        <SignAvatar3D frame={tracking.frame} height={150} />
        <span className="avatar3d-strip__label">
          3D holistic motion — arms, both hands, body-anchored
        </span>
      </div>

      <footer className="readout">
        <div className="readout__word">
          <span className="readout__label">Detected</span>
          <span className="readout__value" title={latestWord?.text ?? undefined}>
            {latestWord?.text ?? "—"}
          </span>
        </div>

        <div className="readout__meter">
          <div className="readout__meter-head">
            <span className="readout__label">
              {buffered ? `Buffering ${buffered.have}/${buffered.need}` : "Live"}
            </span>
            <span className="readout__pct">
              {live && <span className="readout__top">{live.top}</span>}
              {confidence === null ? "—" : `${Math.round(confidence * 100)}%`}
            </span>
          </div>
          <div className="bar">
            <div
              className="bar__fill"
              style={{ width: `${Math.round((confidence ?? 0) * 100)}%` }}
            />
            {/* The backend only accepts a word above this confidence. */}
            <div
              className="bar__threshold"
              style={{ left: `${Math.round(threshold * 100)}%` }}
              aria-hidden
            />
          </div>
        </div>
      </footer>
    </section>
  );
}

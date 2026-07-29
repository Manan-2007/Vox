/** Left column: webcam, skeleton overlay, live confidence, current word. */
import type { HandTracking } from "../hooks/useHandTracking";
import type { LiveGuess } from "../hooks/useVoxSocket";

interface Props {
  tracking: HandTracking;
  live: LiveGuess | null;
  buffered: { have: number; need: number } | null;
  latestWord: { text: string; confidence: number } | null;
}

const STATUS_LABEL: Record<string, string> = {
  starting: "Starting up",
  "requesting-camera": "Requesting camera",
  "loading-model": "Loading hand landmarker",
  running: "Tracking",
  stopped: "Stopped",
};

export function CameraPanel({ tracking, live, buffered, latestWord }: Props) {
  const { videoRef, canvasRef, status, error, delegate, handsVisible } = tracking;
  const confidence = live?.confidence ?? null;

  return (
    <section className="panel panel--camera" aria-label="Camera and hand tracking">
      <header className="panel__head">
        <h2 className="panel__title">Camera</h2>
        <span className={`badge badge--${status === "running" ? "live" : "idle"}`}>
          {STATUS_LABEL[status] ?? status}
          {delegate && status === "running" ? ` · ${delegate}` : ""}
        </span>
      </header>

      <div className="panel__body panel__body--flush">
        {/* Mirrored for display only — the detector still sees the raw frame. */}
        <div className="stage">
          <video ref={videoRef} className="stage__video" playsInline muted />
          <canvas ref={canvasRef} className="stage__overlay" />

          {error && (
            <div className="stage__cover">
              <p className="stage__cover-title">Camera unavailable</p>
              <p className="stage__cover-body">{error}</p>
            </div>
          )}

          {!error && status !== "running" && (
            <div className="stage__cover">
              <p className="stage__cover-title">{STATUS_LABEL[status] ?? status}</p>
            </div>
          )}

          {status === "running" && (
            <div className="stage__hands">
              {handsVisible === 0 ? "No hands detected" : `${handsVisible} hand${handsVisible > 1 ? "s" : ""}`}
            </div>
          )}
        </div>
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
            <div className="bar__threshold" style={{ left: "85%" }} aria-hidden />
          </div>
        </div>
      </footer>
    </section>
  );
}

/**
 * Settings, and the place the product is honest about itself.
 *
 * The diagnostics at the bottom are not developer clutter. When recognition is
 * not working the cause is almost always one of four things — the camera is on
 * the wrong device, detection has fallen back to CPU and is too slow, the body
 * anchor is missing, or the confidence bar is set higher than the model can
 * reach. All four are visible here, in the order you would check them.
 */
import { useEffect } from "react";
import {
  DETECT_FPS,
  SEND_FPS,
  type HandTracking,
} from "../hooks/useHandTracking";
import type { RecognitionQuality } from "../isl/useRecognitionQuality";

interface Props {
  open: boolean;
  onClose: () => void;
  threshold: number;
  onThreshold: (value: number) => void;
  voiceOn: boolean;
  onVoiceOn: (value: boolean) => void;
  ttsSupported: boolean;
  tracking: HandTracking;
  signCount: number;
  recognition: RecognitionQuality;
}

export function SettingsDrawer({
  open,
  onClose,
  threshold,
  onThreshold,
  voiceOn,
  onVoiceOn,
  ttsSupported,
  tracking,
  signCount,
  recognition,
}: Props) {
  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <>
      <div className="scrim" onClick={onClose} aria-hidden />
      <aside className="drawer" role="dialog" aria-label="Settings">
        <header className="drawer__head">
          <h2 className="drawer__title">Settings</h2>
          <button type="button" className="btn btn--ghost" onClick={onClose}>
            Close
          </button>
        </header>

        <div className="drawer__body">
          <div className="field">
            <label className="field__label" htmlFor="threshold">
              Acceptance confidence — {Math.round(threshold * 100)}%
            </label>
            <input
              id="threshold"
              type="range"
              min={0.5}
              max={0.99}
              step={0.01}
              value={threshold}
              onChange={(event) => onThreshold(Number(event.target.value))}
            />
            <p className="field__hint">
              How sure the model must be before a word is accepted. Lower it if
              your signing is not being picked up; raise it if wrong words
              appear. Both trades are real — there is no setting with neither.
            </p>
          </div>

          <div className="field">
            <label className="field__label" htmlFor="camera">
              Camera
            </label>
            <select
              id="camera"
              value={tracking.cameraId ?? ""}
              onChange={(event) => tracking.selectCamera(event.target.value)}
            >
              <option value="">Default camera</option>
              {tracking.devices.map((device) => (
                <option key={device.deviceId} value={device.deviceId}>
                  {device.label || "Camera"}
                </option>
              ))}
            </select>
            <div className="field field--row">
              <input
                id="camera-on"
                type="checkbox"
                checked={tracking.cameraOn}
                onChange={(event) => tracking.setCameraOn(event.target.checked)}
              />
              <label className="field__label" htmlFor="camera-on">
                Camera on
              </label>
            </div>
            <p className="field__hint">
              Turning the camera off releases the device, so the indicator light
              goes out. No video is uploaded either way — only landmark numbers
              leave the browser, and only to the interpreter running on this
              machine.
            </p>
          </div>

          <div className="field field--row">
            <input
              id="voice"
              type="checkbox"
              checked={voiceOn}
              disabled={!ttsSupported}
              onChange={(event) => onVoiceOn(event.target.checked)}
            />
            <label className="field__label" htmlFor="voice">
              Read signed sentences aloud
            </label>
          </div>

          {recognition.loaded && (
            <div className="field">
              <span className="field__label">How good is recognition?</span>
              <div>
                <Stat
                  label="Correct when it speaks"
                  value={`${Math.round(recognition.gatedPrecision * 100)}%`}
                />
                <Stat
                  label="Top-1 over all words"
                  value={`${Math.round(recognition.top1 * 100)}%`}
                />
                <Stat
                  label="Top-3 over all words"
                  value={`${Math.round(recognition.top3 * 100)}%`}
                />
                <Stat label="Words it knows" value={String(recognition.classes)} />
                <Stat
                  label="Verified at 80%+"
                  value={String(recognition.reliable.size)}
                />
                <Stat
                  label="Never measured"
                  value={String(
                    Math.max(0, recognition.classes - recognition.measured.size),
                  )}
                />
              </div>
              <p className="field__hint">
                Measured on recordings the model never trained on. It stays
                silent below the confidence bar, which is why &ldquo;correct when
                it speaks&rdquo; is much higher than raw top-1 — a wrong word is
                worse than no word. Words never measured have only one recording
                in the dictionary, so nothing could be held back to test them;
                the transcript underlines those.
              </p>
            </div>
          )}

          <div className="field">
            <span className="field__label">Diagnostics</span>
            <div>
              <Stat label="Detection" value={tracking.delegate ?? "starting…"} />
              <Stat
                label="Detection rate"
                value={`${tracking.fps} / ${DETECT_FPS} fps`}
              />
              <Stat label="Sent to model" value={`${SEND_FPS} fps`} />
              <Stat label="Inference" value={`${tracking.inferMs} ms`} />
              <Stat label="Hands tracked" value={String(tracking.quality.hands)} />
              <Stat
                label="Body anchor"
                value={tracking.quality.body ? "found" : "missing"}
              />
              <Stat
                label="Dropped frames (L / R)"
                value={tracking.quality.gaps.join(" / ")}
              />
              <Stat
                label="Hand motion"
                value={tracking.quality.motion.toFixed(2)}
              />
              <Stat label="Signs in library" value={String(signCount)} />
            </div>
            <p className="field__hint">
              If detection says CPU, the browser refused GPU acceleration for
              MediaPipe and everything will be slower. If the body anchor is
              missing, no prediction is possible at all — every coordinate is
              measured relative to your shoulders.
            </p>
          </div>
        </div>
      </aside>
    </>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat">
      <span>{label}</span>
      <span className="stat__value">{value}</span>
    </div>
  );
}

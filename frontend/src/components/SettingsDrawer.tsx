/** Settings drawer (P11): threshold, camera, auto-speak, clear conversation. */
import type { HandTracking } from "../hooks/useHandTracking";

interface Props {
  open: boolean;
  onClose: () => void;
  threshold: number;
  onThreshold: (value: number) => void;
  autoSpeak: boolean;
  onAutoSpeak: (value: boolean) => void;
  ttsSupported: boolean;
  tracking: HandTracking;
  onClearConversation: () => void;
}

export function SettingsDrawer({
  open,
  onClose,
  threshold,
  onThreshold,
  autoSpeak,
  onAutoSpeak,
  ttsSupported,
  tracking,
  onClearConversation,
}: Props) {
  if (!open) return null;

  return (
    <>
      <div className="drawer__scrim" onClick={onClose} aria-hidden />
      <aside className="drawer" role="dialog" aria-label="Settings">
        <header className="drawer__head">
          <h2 className="drawer__title">Settings</h2>
          <button type="button" className="button button--quiet" onClick={onClose}>
            Close
          </button>
        </header>

        <div className="drawer__body">
          <label className="field">
            <span className="field__label">
              Confidence threshold — {Math.round(threshold * 100)}%
            </span>
            <input
              type="range"
              min={0.5}
              max={0.99}
              step={0.01}
              value={threshold}
              onChange={(e) => onThreshold(Number(e.target.value))}
            />
            <span className="field__hint">
              A word is only accepted above this confidence. Raise it if you see
              wrong words; lower it if signs are missed.
            </span>
          </label>

          <label className="field">
            <span className="field__label">Camera</span>
            <select
              className="input"
              value={tracking.cameraId ?? ""}
              onChange={(e) => tracking.selectCamera(e.target.value)}
              disabled={tracking.devices.length === 0}
            >
              <option value="">Default camera</option>
              {tracking.devices.map((device, i) => (
                <option key={device.deviceId} value={device.deviceId}>
                  {device.label || `Camera ${i + 1}`}
                </option>
              ))}
            </select>
            {tracking.devices.length === 0 && (
              <span className="field__hint">
                Camera list appears once camera permission is granted.
              </span>
            )}
          </label>

          <label className="field field--row">
            <input
              type="checkbox"
              checked={autoSpeak}
              onChange={(e) => onAutoSpeak(e.target.checked)}
              disabled={!ttsSupported}
            />
            <span className="field__label">
              Voice output — read finished sentences aloud
              {!ttsSupported && " (unavailable in this browser)"}
            </span>
          </label>

          <button
            type="button"
            className="button"
            onClick={() => {
              onClearConversation();
              onClose();
            }}
          >
            Clear conversation
          </button>
        </div>
      </aside>
    </>
  );
}

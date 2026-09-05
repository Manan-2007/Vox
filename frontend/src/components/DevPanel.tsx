/**
 * Developer mode: what the pipeline is actually doing, live.
 *
 * Every figure here answers a question that was previously unanswerable from
 * the outside:
 *
 *   FPS / frame ms      is the render loop the bottleneck, or the tracker?
 *   infer ms            how long MediaPipe took on the last frame
 *   tracking states     per channel — which part failed, not "it broke"
 *   presence            how much of the pose is measurement vs. rest
 *   peak step           the largest single-frame bone rotation, in degrees.
 *                       The rig's central promise is that it never snaps, and
 *                       this is the number that either keeps or breaks it. A
 *                       human joint tops out near 25 rad/s, which at 60 fps is
 *                       about 24°/frame — anything above that is a snap, and it
 *                       is flagged red rather than merely printed.
 *   candidate           what the recogniser is considering and why it was
 *                       refused — the thing the old UI showed as if it were an
 *                       answer.
 */

import type { AvatarDiagnostics } from "./SignAvatar";
import type { FrameQuality } from "../hooks/useHandTracking";
import type { LiveGuess } from "../hooks/useVoxSocket";

interface Props {
  diagnostics: AvatarDiagnostics | null;
  tracking: {
    fps: number;
    inferMs: number;
    delegate: "GPU" | "CPU" | null;
    quality: FrameQuality;
  };
  live: LiveGuess | null;
  onClose: () => void;
}

/** Above this many degrees in one frame, the motion is not human. */
const SNAP_DEGREES = 24;

const CHANNEL_LABELS: [string, string][] = [
  ["body", "body"],
  ["head", "head"],
  ["armL", "arm L"],
  ["armR", "arm R"],
  ["handL", "hand L"],
  ["handR", "hand R"],
];

export function DevPanel({ diagnostics, tracking, live, onClose }: Props) {
  const snapping = (diagnostics?.peakStep ?? 0) > SNAP_DEGREES;

  return (
    <aside className="devpanel" aria-label="Developer diagnostics">
      <div className="devpanel__title">
        <span>Diagnostics</span>
        <button
          type="button"
          className="iconbtn iconbtn--tiny"
          onClick={onClose}
          aria-label="Close diagnostics"
        >
          ×
        </button>
      </div>

      <div className="devpanel__row">
        <span>render</span>
        <span>
          {diagnostics?.fps ?? 0} fps · {diagnostics?.frameMs.toFixed(1) ?? "—"} ms
        </span>
      </div>
      <div className="devpanel__row">
        <span>draws / tris</span>
        <span>
          {diagnostics?.drawCalls ?? 0} · {formatCount(diagnostics?.triangles ?? 0)}
        </span>
      </div>
      <div className={`devpanel__row ${snapping ? "devpanel__row--alarm" : ""}`}>
        <span>peak step</span>
        <span>{(diagnostics?.peakStep ?? 0).toFixed(1)}°/frame</span>
      </div>

      <div className="devpanel__group">
        <div className="devpanel__row">
          <span>tracker</span>
          <span>
            {tracking.delegate ?? "—"} · {tracking.fps} fps
          </span>
        </div>
        <div className="devpanel__row">
          <span>inference</span>
          <span>{tracking.inferMs} ms</span>
        </div>
        <div className="devpanel__row">
          <span>hands / body</span>
          <span>
            {tracking.quality.hands} · {tracking.quality.body ? "seen" : "lost"}
          </span>
        </div>
        <div className="devpanel__row">
          <span>gaps L/R</span>
          <span>
            {tracking.quality.gaps[0]} / {tracking.quality.gaps[1]}
          </span>
        </div>
        <div className="devpanel__row">
          <span>motion</span>
          <span>{tracking.quality.motion.toFixed(2)} u/s</span>
        </div>
      </div>

      <div className="devpanel__group">
        {CHANNEL_LABELS.map(([key, label]) => {
          const state = diagnostics?.states?.[key] ?? "idle";
          const presence = diagnostics?.presence?.[key] ?? 0;
          return (
            <div className="devpanel__row" key={key}>
              <span>{label}</span>
              <span>
                <span className={`devpanel__state devpanel__state--${state}`}>
                  {state}
                </span>{" "}
                {Math.round(presence * 100)}%
              </span>
            </div>
          );
        })}
      </div>

      <div className="devpanel__group">
        <div className="devpanel__row">
          <span>candidate</span>
          <span>{live?.candidate ?? "—"}</span>
        </div>
        <div className="devpanel__row">
          <span>confidence</span>
          <span>{live ? `${(live.confidence * 100).toFixed(1)}%` : "—"}</span>
        </div>
        <div className="devpanel__row">
          <span>refused</span>
          <span>{live?.reason ?? "—"}</span>
        </div>
        <div className="devpanel__row">
          <span>stable for</span>
          <span>{live?.stableFor ?? 0}</span>
        </div>
      </div>
    </aside>
  );
}

function formatCount(value: number): string {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1000) return `${(value / 1000).toFixed(0)}k`;
  return String(value);
}

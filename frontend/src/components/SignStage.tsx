/**
 * The stage: one lit space that shows either the reference signing or your own.
 *
 * This is the component that replaced the old three-panel layout, where a video
 * clip played in one box and a wireframe of the same sign twitched in another
 * underneath it. Two renderings of the same thing, in two places, at two
 * fidelities — the viewer had to do the work of relating them.
 *
 * Now there is one rig and one stage. When the app is signing to you, the rig
 * plays the reference motion. When you are signing to it, the same rig mirrors
 * your hands, so you can see exactly what the recogniser sees. The camera itself
 * is a small picture-in-picture, because its job is framing feedback — am I in
 * shot, is the light okay — and not to be the main event.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { AvatarControls, SignAvatar, type AvatarDiagnostics } from "./SignAvatar";
import { DevPanel } from "./DevPanel";
import type { SignPlayer, PlayerState, QueueItem } from "../avatar/signMotion";
import type { HandTracking } from "../hooks/useHandTracking";
import type { LiveGuess } from "../hooks/useVoxSocket";

export type StageMode = "reference" | "live";

interface Props {
  mode: StageMode;
  queue: readonly QueueItem[];
  tracking: HandTracking;
  /** Whether the recogniser's socket is connected and has a model. */
  ready: boolean;
  loadingSigns: boolean;
  /** Fires when the reference queue has played to the end. */
  onQueueDone: () => void;
  /** The recogniser's current candidate, for the diagnostics panel. */
  live: LiveGuess | null;
}

/** Playback preferences live here so they survive a mode switch. */
export function SignStage({
  mode,
  queue,
  tracking,
  ready,
  loadingSigns,
  onQueueDone,
  live,
}: Props) {
  const [speed, setSpeed] = useState(1);
  const [loop, setLoop] = useState(false);
  const [mirror, setMirror] = useState(true);
  /* Hands by default.
     For a signing tool this is the better default, not a lesser one: it puts the
     whole frame on the part that carries the meaning and sidesteps the
     uncanny-valley problem completely. The whole figure stays one click away,
     because it is what carries non-manual grammar. */
  const [presentation, setPresentation] = useState<"figure" | "hands">("hands");
  const [state, setState] = useState<PlayerState>({
    index: 0,
    progress: 0,
    playing: false,
    // NOT `finished: true`. An idle player is "finished" in the sense that it
    // has nothing left to play, and reporting that as the initial state fired
    // the end-of-queue handler the instant a phrase arrived — the stage flipped
    // straight back to the camera without ever showing the sign.
    finished: false,
    transitioning: false,
  });
  const [player, setPlayer] = useState<SignPlayer | null>(null);

  /* Developer mode.

     Kept out of Settings on purpose: it is for whoever is working on the rig,
     not for the person having a conversation. The panel updates at frame rate,
     so its state is held in a ref and mirrored into React only while it is
     open — a closed panel costs nothing. */
  const [dev, setDev] = useState(false);
  const [diagnostics, setDiagnostics] = useState<AvatarDiagnostics | null>(null);
  const devRef = useRef(dev);
  devRef.current = dev;
  const lastPush = useRef(0);

  const handleDiagnostics = useCallback((next: AvatarDiagnostics) => {
    if (!devRef.current) return;
    // Throttle to ~8 Hz. At 60 Hz this would re-render the panel on every frame
    // and the diagnostics would themselves become the bottleneck they measure.
    const now = performance.now();
    if (now - lastPush.current < 120) return;
    lastPush.current = now;
    setDiagnostics(next);
  }, []);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      // Ctrl/Cmd + Alt + D. Deliberately awkward: it must not fire while
      // someone is typing a phrase into the composer.
      if (event.altKey && (event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "d") {
        event.preventDefault();
        setDev((on) => !on);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // "Nothing tracked yet" is only true before the first frame ever arrives.
  // Losing the body mid-session is a different situation with its own hint, and
  // covering the idling figure with an empty state every time a shoulder drops
  // out made the stage flash.
  const seenTracking = useRef(false);
  if (tracking.avatarFrame) seenTracking.current = true;

  const showingReference = mode === "reference" && queue.length > 0;
  const current = showingReference ? queue[state.index] : null;

  // Hand the stage back to the camera once the phrase has finished playing —
  // on the transition into "finished", not on the state being finished, so a
  // queue that has not started yet does not immediately count as over.
  const doneRef = useRef(onQueueDone);
  doneRef.current = onQueueDone;
  const wasPlaying = useRef(false);
  useEffect(() => {
    if (!showingReference) {
      wasPlaying.current = false;
      return;
    }
    if (state.playing) wasPlaying.current = true;
    else if (wasPlaying.current && state.finished) {
      wasPlaying.current = false;
      doneRef.current();
    }
  }, [showingReference, state.playing, state.finished]);

  return (
    <section className="card card--stage">
      <header className="card__head">
        <h2 className="card__title">
          {showingReference ? "Signing to you" : "Your signing"}
        </h2>
        <div className="topbar__tools">
          {tracking.delegate && (
            <span className="chip" title="Where MediaPipe is running">
              {tracking.delegate}
              {tracking.fps > 0 && ` · ${tracking.fps} fps`}
            </span>
          )}
          <div className="segmented" role="group" aria-label="What to show">
            <button
              type="button"
              className={`segmented__option ${presentation === "hands" ? "is-active" : ""}`}
              onClick={() => setPresentation("hands")}
              title="Hands only — the whole frame on the part that carries meaning"
            >
              Hands
            </button>
            <button
              type="button"
              className={`segmented__option ${presentation === "figure" ? "is-active" : ""}`}
              onClick={() => setPresentation("figure")}
              title="Whole signer — includes the face, which carries question and negation marking"
            >
              Signer
            </button>
          </div>
          <button
            type="button"
            className={`iconbtn ${dev ? "is-active" : ""}`}
            onClick={() => setDev(!dev)}
            aria-pressed={dev}
            title="Developer diagnostics (Ctrl/Cmd + Alt + D)"
          >
            ⚙
          </button>
        </div>
      </header>

      <div className="card__body">
        <div className="stage">
          <SignAvatar
            queue={showingReference ? queue : undefined}
            liveFrame={showingReference ? null : tracking.avatarFrame}
            mirror={mirror}
            playbackSpeed={speed}
            loop={loop}
            onState={setState}
            onPlayer={setPlayer}
            onDiagnostics={handleDiagnostics}
            showSkeleton={dev}
            presentation={presentation}
          />

          {dev && (
            <DevPanel
              diagnostics={diagnostics}
              tracking={{
                fps: tracking.fps,
                inferMs: tracking.inferMs,
                delegate: tracking.delegate,
                quality: tracking.quality,
              }}
              live={live}
              onClose={() => setDev(false)}
            />
          )}

          <StageHint
            mode={mode}
            tracking={tracking}
            ready={ready}
            loadingSigns={loadingSigns}
          />

          {showingReference && current && (
            <div className="stage__caption">
              <p className="stage__word">{current.label}</p>
              <p className="stage__gloss">
                {current.missing
                  ? "no sign in the library yet"
                  : state.transitioning
                    ? `${current.gloss} →`
                    : current.gloss}
              </p>
            </div>
          )}

          {!showingReference && !tracking.avatarFrame && !seenTracking.current &&
            !tracking.error && tracking.cameraOn && (
            <div className="stage__empty">
              <p className="stage__empty-title">Nothing tracked yet</p>
              <p className="stage__empty-body">
                Stand back so your head and both shoulders are in frame. The
                figure here mirrors what the recogniser sees, landmark for
                landmark.
              </p>
            </div>
          )}

          <CameraPip tracking={tracking} />
        </div>

        {player && showingReference && (
          <AvatarControls
            player={player}
            state={state}
            speed={speed}
            onSpeed={setSpeed}
            loop={loop}
            onLoop={setLoop}
            mirror={mirror}
            onMirror={setMirror}
          />
        )}
      </div>
    </section>
  );
}

/**
 * The camera, small and in the corner.
 *
 * Kept visible even in reference mode: a signer needs to know they are still in
 * frame while they watch a sign being demonstrated, and discovering afterwards
 * that they had drifted out of shot is the most annoying possible failure.
 */
function CameraPip({ tracking }: { tracking: HandTracking }) {
  if (!tracking.cameraOn) {
    return (
      <div className="pip pip--off">
        <span>Camera off — recognition paused</span>
      </div>
    );
  }
  return (
    <div className="pip">
      <video ref={tracking.videoRef} className="pip__video" playsInline muted />
      <canvas ref={tracking.canvasRef} className="pip__overlay" />
      <span className="pip__tag">
        {tracking.quality.hands === 0
          ? "no hands"
          : `${tracking.quality.hands} hand${tracking.quality.hands > 1 ? "s" : ""}`}
      </span>
    </div>
  );
}

/**
 * One line saying what is wrong, in priority order, or nothing at all.
 *
 * Every condition here was a silent failure at some point: the app looked
 * identical whether the camera was denied, the model was missing, the signer was
 * out of frame, or they simply had not started signing yet. Naming the actual
 * blocker is the difference between a product and a demo that "just doesn't
 * work sometimes".
 */
function StageHint({
  mode,
  tracking,
  ready,
  loadingSigns,
}: {
  mode: StageMode;
  tracking: HandTracking;
  ready: boolean;
  loadingSigns: boolean;
}) {
  const [settled, setSettled] = useState(false);
  useEffect(() => {
    // Do not flash a complaint during the first second of startup.
    const timer = window.setTimeout(() => setSettled(true), 1200);
    return () => window.clearTimeout(timer);
  }, []);

  if (loadingSigns) {
    return <Hint tone="">Loading signs…</Hint>;
  }
  if (tracking.error) {
    return <Hint tone="error">{tracking.error}</Hint>;
  }
  if (mode === "reference") return null;

  if (tracking.status === "loading-model") {
    return <Hint tone="">Starting the hand tracker…</Hint>;
  }
  if (tracking.status === "requesting-camera") {
    return <Hint tone="">Waiting for camera permission…</Hint>;
  }
  if (!tracking.cameraOn) {
    return <Hint tone="warn">Camera is off</Hint>;
  }
  if (!settled) return null;

  if (!ready) {
    return (
      <Hint tone="warn">
        Recognition is off — no trained model loaded
      </Hint>
    );
  }
  if (!tracking.quality.body) {
    return (
      <Hint tone="warn">
        Can&rsquo;t see your shoulders — step back and centre yourself
      </Hint>
    );
  }
  if (tracking.quality.hands === 0) {
    return <Hint tone="warn">Raise your hands into frame</Hint>;
  }
  if (Math.max(...tracking.quality.gaps) > 6) {
    return (
      <Hint tone="warn">
        Losing a hand — more light, or move it away from the background
      </Hint>
    );
  }
  return (
    <Hint tone="good">
      Tracking {tracking.quality.hands === 2 ? "both hands" : "one hand"}
    </Hint>
  );
}

function Hint({
  tone,
  children,
}: {
  tone: "good" | "warn" | "error" | "";
  children: React.ReactNode;
}) {
  return (
    <div className={`hint ${tone ? `hint--${tone}` : ""}`}>
      <span className="hint__dot" />
      <span>{children}</span>
    </div>
  );
}

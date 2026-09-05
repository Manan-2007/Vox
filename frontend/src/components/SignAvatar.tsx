/**
 * The signing avatar — the product's primary output.
 *
 * One rig, two sources:
 *   replay — a queue of signs from the motion library (what to sign)
 *   live   — frames from the camera worker (what you are signing)
 *
 * Using one rig for both is deliberate: a learner comparing their own hands to
 * the reference should be looking at the same object posed two ways, not at a
 * polished avatar next to a wireframe of themselves. Both go through the same
 * retargeter, the same joint limits and the same blender, so the only difference
 * between them is where the numbers came from.
 *
 * This component owns React state and nothing else. The render loop lives in
 * `AvatarStage`, outside React entirely — the previous version turned every
 * tracking frame into a `setState`, which re-rendered the whole session page at
 * up to 30 Hz.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AvatarStage } from "../avatar/AvatarStage";
import type { BlendReport, TrackingState } from "../avatar/pose/blend";
import { PoseRetargeter } from "../avatar/retarget";
import { AvatarError, loadVrmAvatar } from "../avatar/VrmAvatar";
import { ProceduralAvatar } from "../avatar/ProceduralAvatar";
import type { SigningAvatar } from "../avatar/SigningAvatar";
import { blankPose, type HumanoidPose } from "../core/humanoid";
import {
  FRAME_FLOATS,
  SignPlayer,
  type PlayerState,
  type QueueItem,
} from "../avatar/signMotion";
import { NEUTRAL_FACE } from "../avatar/nonManual";

/**
 * Where the humanoid lives.
 *
 * No avatar ships with the repository, and that is a licensing decision rather
 * than an oversight: a VRM is someone's copyrighted asset with terms that govern
 * redistribution. Drop your own in `frontend/public/avatar/signer.vrm`, or point
 * `VITE_VOX_AVATAR` somewhere else.
 */
const AVATAR_URL =
  (import.meta.env.VITE_VOX_AVATAR as string | undefined) ?? "/avatar/signer.vrm";

export interface AvatarDiagnostics {
  fps: number;
  frameMs: number;
  drawCalls: number;
  triangles: number;
  peakStep: number;
  states: Record<string, TrackingState>;
  presence: Record<string, number>;
}

interface Props {
  /** Replay queue. When empty, `liveFrame` drives the rig instead. */
  queue?: readonly QueueItem[];
  /** An avatar-format frame from the camera, for the live view. */
  liveFrame?: Float32Array | null;
  /** Mirror the figure, so a learner can copy it directly. */
  mirror?: boolean;
  /** Draw the bone overlay. */
  showSkeleton?: boolean;
  /** Draw the whole signer, or hands only. */
  presentation?: "figure" | "hands";
  playbackSpeed?: number;
  loop?: boolean;
  onState?: (state: PlayerState) => void;
  onPlayer?: (player: SignPlayer) => void;
  onDiagnostics?: (diagnostics: AvatarDiagnostics) => void;
  className?: string;
}

type LoadState =
  | { status: "loading"; progress: number }
  | { status: "ready" }
  /** Running on the built-in signer because no .vrm was found. */
  | { status: "fallback" }
  /** Running on the built-in signer because the .vrm was unusable. */
  | { status: "fallback-error"; message: string; remedy: string };

export function SignAvatar({
  queue,
  liveFrame,
  mirror = true,
  showSkeleton = false,
  presentation = "hands",
  playbackSpeed = 1,
  loop = false,
  onState,
  onPlayer,
  onDiagnostics,
  className = "",
}: Props) {
  const mountRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<AvatarStage | null>(null);
  const avatarRef = useRef<SigningAvatar | null>(null);

  const playerRef = useRef<SignPlayer>(null as unknown as SignPlayer);
  if (!playerRef.current) playerRef.current = new SignPlayer();
  const player = playerRef.current;
  player.speed = playbackSpeed;
  player.loop = loop;

  // Refs, so the render loop reads current values without being rebuilt.
  const liveRef = useRef<Float32Array | null>(null);
  liveRef.current = liveFrame ?? null;
  const onStateRef = useRef(onState);
  onStateRef.current = onState;
  const onDiagnosticsRef = useRef(onDiagnostics);
  onDiagnosticsRef.current = onDiagnostics;

  const [load, setLoad] = useState<LoadState>({ status: "loading", progress: 0 });

  const onPlayerRef = useRef(onPlayer);
  onPlayerRef.current = onPlayer;
  useEffect(() => {
    onPlayerRef.current?.(player);
  }, [player]);

  useEffect(() => {
    if (queue && queue.length) player.setQueue([...queue]);
    else player.clear();
  }, [queue, player]);

  /* ------------------------------------------------------------- stage -- */
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    let stage: AvatarStage;
    try {
      stage = new AvatarStage(mount, {
        mirror,
        skeleton: showSkeleton,
        presentation,
      });
    } catch (error) {
      setLoad({
        status: "fallback-error",
        message: "This browser could not start WebGL.",
        remedy: "Recognition and speech still work; the 3D signer does not.",
      });
      return;
    }
    stageRef.current = stage;

    /* Two retargeters, never one.

       Each owns landmark filters with their own history, and sharing them would
       filter a recorded sign against the user's live hand the moment the stage
       switched sources — producing a frame that is neither. */
    let replayRetargeter: PoseRetargeter | null = null;
    let liveRetargeter: PoseRetargeter | null = null;
    const replayPose: HumanoidPose = blankPose();
    const livePose: HumanoidPose = blankPose();
    const frame = new Float32Array(FRAME_FLOATS);
    let lastReported = -1;

    stage.source = (dt, time) => {
      const current = playerRef.current;

      if (current.items.length > 0 && replayRetargeter) {
        const advanced = current.advance(dt, frame);
        const state = current.state;
        // Report only on change; this runs sixty times a second.
        const fingerprint =
          state.index * 1000 +
          Math.round(state.progress * 100) +
          (state.playing ? 200000 : 0) +
          (state.transitioning ? 400000 : 0);
        if (onStateRef.current && fingerprint !== lastReported) {
          lastReported = fingerprint;
          onStateRef.current(state);
        }
        if (!advanced) return null;
        const ok = replayRetargeter.solve(frame, replayPose, {
          expression: current.face,
          time,
          timeMs: performance.now(),
        });
        return ok ? replayPose : null;
      }

      const live = liveRef.current;
      if (!live || !liveRetargeter) return null;
      const ok = liveRetargeter.solve(live, livePose, {
        expression: NEUTRAL_FACE,
        time,
        timeMs: performance.now(),
      });
      return ok ? livePose : null;
    };

    stage.onFrame = (report: BlendReport, stats) => {
      onDiagnosticsRef.current?.({
        fps: stats.fps,
        frameMs: stats.frameMs,
        drawCalls: stats.drawCalls,
        triangles: stats.triangles,
        peakStep: stats.peakStep,
        states: report.states,
        presence: report.presence,
      });
    };

    stage.start();

    const observer = new ResizeObserver(() => stage.resize());
    observer.observe(mount);

    /* ------------------------------------------------------------ load -- */
    let cancelled = false;

    const install = (avatar: SigningAvatar) => {
      avatarRef.current = avatar;
      // One retargeter per SOURCE, both built from this body's own measured
      // rest pose — which is why swapping avatars needs no other change.
      replayRetargeter = new PoseRetargeter(avatar.rest);
      liveRetargeter = new PoseRetargeter(avatar.rest);
      stage.setAvatar(avatar);
    };

    /* The built-in signer goes in FIRST, always, and a .vrm replaces it if one
       loads. Two reasons it is not the other way round:

       it removes the blank-stage state entirely — there is never a moment where
       the product looks broken — and it means a bad .vrm degrades to a working
       signer rather than to nothing. Generating it is a few milliseconds of
       geometry, so nothing is paid for the insurance. */
    install(new ProceduralAvatar());

    void (async () => {
      try {
        const head = await fetch(AVATAR_URL, { method: "HEAD" }).catch(() => null);
        if (cancelled) return;
        // A dev server happily serves index.html for a missing asset, so a 200
        // is not enough — an HTML content type means the file is not there.
        const type = head?.headers.get("content-type") ?? "";
        if (!head?.ok || type.includes("text/html")) {
          setLoad({ status: "fallback" });
          return;
        }

        const { avatar } = await loadVrmAvatar(AVATAR_URL, (fraction) => {
          if (!cancelled) {
            setLoad({ status: "loading", progress: fraction < 0 ? 0 : fraction });
          }
        });
        if (cancelled) {
          avatar.dispose();
          return;
        }
        const previous = avatarRef.current;
        install(avatar);
        previous?.dispose();
        setLoad({ status: "ready" });
      } catch (error) {
        if (cancelled) return;
        // Stay on the built-in signer and say why the .vrm was refused.
        if (error instanceof AvatarError) {
          setLoad({
            status: "fallback-error",
            message: error.message,
            remedy: error.remedy,
          });
        } else {
          setLoad({
            status: "fallback-error",
            message: error instanceof Error ? error.message : String(error),
            remedy: "Check the file is a valid .vrm.",
          });
        }
      }
    })();

    return () => {
      cancelled = true;
      observer.disconnect();
      stage.dispose();
      stageRef.current = null;
      avatarRef.current = null;
    };
    // Built once. Mirror and skeleton are pushed through `setOptions` below so
    // toggling them never tears down the WebGL context.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    stageRef.current?.setOptions({ mirror, skeleton: showSkeleton, presentation });
  }, [mirror, showSkeleton, presentation]);

  return (
    <div className={`avatar ${className}`}>
      <div ref={mountRef} className="avatar__canvas" />
      {load.status === "loading" && <AvatarNotice state={load} />}
      {(load.status === "fallback" || load.status === "fallback-error") && (
        <AvatarBadge state={load} />
      )}
    </div>
  );
}

/**
 * What to show when there is no avatar.
 *
 * Named the actual blocker rather than a spinner. Every one of these states was
 * previously indistinguishable from "it is just slow", and the difference
 * between "no file" and "the file has no finger bones" is the difference between
 * a two-minute fix and an afternoon.
 */
function AvatarNotice({ state }: { state: LoadState }) {
  if (state.status === "loading") {
    return (
      <div className="avatar__notice">
        <p className="avatar__notice-title">Loading the signer…</p>
        {state.progress > 0 && (
          <div className="avatar__progress" aria-hidden>
            <div style={{ width: `${Math.round(state.progress * 100)}%` }} />
          </div>
        )}
      </div>
    );
  }

  return null;
}

/**
 * A corner badge, not a blocking notice.
 *
 * The built-in signer is a working signer, so covering it with a full-panel
 * "no avatar" message — which is what this used to do — hid a functioning
 * product behind a setup screen. The badge says which body is on stage and how
 * to change it, and gets out of the way.
 */
function AvatarBadge({
  state,
}: {
  state: { status: "fallback" } | { status: "fallback-error"; message: string; remedy: string };
}) {
  const [open, setOpen] = useState(false);
  const refused = state.status === "fallback-error";

  return (
    <div className={`avatar__badge ${refused ? "avatar__badge--warn" : ""}`}>
      <button
        type="button"
        className="avatar__badge-button"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
      >
        {refused ? "Built-in signer — your .vrm was refused" : "Built-in signer"}
      </button>
      {open && (
        <div className="avatar__badge-body">
          {refused ? (
            <>
              <p>{state.message}</p>
              <p className="avatar__notice-body--small">{state.remedy}</p>
            </>
          ) : (
            <p>
              Vox is using its own generated signer. To use a different one, put a
              VRM here and reload:
            </p>
          )}
          <code className="avatar__path">frontend/public/avatar/signer.vrm</code>
          <p className="avatar__notice-body--small">
            It must be a VRM 0.x or 1.0 humanoid <strong>with full finger bones</strong>
            {" "}— all fifteen per hand. A model without them cannot form
            handshapes, so it is refused rather than loaded into a signer that
            cannot sign.
          </p>
        </div>
      )}
    </div>
  );
}

/** Shared playback bar, used by the panels that host an avatar. */
export function AvatarControls({
  player,
  state,
  speed,
  onSpeed,
  loop,
  onLoop,
  mirror,
  onMirror,
}: {
  player: SignPlayer;
  state: PlayerState;
  speed: number;
  onSpeed: (value: number) => void;
  loop: boolean;
  onLoop: (value: boolean) => void;
  mirror: boolean;
  onMirror: (value: boolean) => void;
}) {
  const [, force] = useState(0);
  const rerender = useCallback(() => force((n) => n + 1), []);
  const items = player.items;
  const speeds = useMemo(() => [0.5, 0.75, 1], []);

  return (
    <div className="playback">
      <div className="playback__row">
        <button
          type="button"
          className="iconbtn iconbtn--primary"
          onClick={() => {
            player.toggle();
            rerender();
          }}
          disabled={items.length === 0}
          aria-label={state.playing ? "Pause" : "Play"}
        >
          {state.playing ? "❙❙" : "▶"}
        </button>
        <button
          type="button"
          className="iconbtn"
          onClick={() => {
            player.restart();
            rerender();
          }}
          disabled={items.length === 0}
          aria-label="Play from the beginning"
        >
          ↻
        </button>

        <div className="playback__track" aria-hidden>
          <div
            className="playback__fill"
            style={{ width: `${Math.round(state.progress * 100)}%` }}
          />
        </div>

        <div className="segmented" role="group" aria-label="Playback speed">
          {speeds.map((value) => (
            <button
              key={value}
              type="button"
              className={`segmented__option ${speed === value ? "is-active" : ""}`}
              onClick={() => onSpeed(value)}
            >
              {value}×
            </button>
          ))}
        </div>

        <button
          type="button"
          className={`iconbtn ${loop ? "is-active" : ""}`}
          onClick={() => onLoop(!loop)}
          aria-pressed={loop}
          title="Loop the phrase"
        >
          ⟳
        </button>
        <button
          type="button"
          className={`iconbtn ${mirror ? "is-active" : ""}`}
          onClick={() => onMirror(!mirror)}
          aria-pressed={mirror}
          title={
            mirror
              ? "Mirrored — copy the hand you see on your own side"
              : "Facing you — as another signer would appear"
          }
        >
          ⇋
        </button>
      </div>

      {items.length > 0 && (
        <ol className="playback__words">
          {items.map((item, index) => (
            <li key={`${item.gloss}-${index}`}>
              <button
                type="button"
                className={`wordchip ${index === state.index ? "is-active" : ""} ${
                  item.missing ? "wordchip--missing" : ""
                }`}
                onClick={() => {
                  player.step(index);
                  player.play();
                  rerender();
                }}
                title={
                  item.missing
                    ? `No sign in the library for "${item.label}"`
                    : `Jump to "${item.label}"`
                }
              >
                {item.label}
              </button>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

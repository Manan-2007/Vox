/**
 * Right column: speech -> ISL playback, fronted by the Vox orb.
 *
 * The orb is the panel's resting face and reacts to real activity — listening
 * while the mic captures or a phrase is being typed, speaking while TTS reads
 * a sentence, thinking while clips play, and a happy hop when a sign lands.
 *
 * Spoken (or typed) English is glossed into tokens; tokens with a clip in
 * public/clips/manifest.json queue up and play in sequence. Clips are
 * user-supplied — a queued word whose file is missing shows a card briefly and
 * advances instead of stalling. Browsers without SpeechRecognition fall back
 * to the typed input, which drives the identical pipeline.
 */
import { useEffect, useRef, useState } from "react";
import { Orb, type OrbState } from "./Orb";
import { SignAvatar3D } from "./SignAvatar3D";
import type { IslQueue } from "../isl/useIslQueue";
import type { SpeechRecognitionState } from "../hooks/useSpeechRecognition";

const MISSING_CLIP_MS = 1400;
const HAPPY_MS = 900;

interface Props {
  isl: IslQueue;
  recognition: SpeechRecognitionState;
  /** Typed phrases follow the same path as heard speech (transcript + queue). */
  onPhrase: (text: string) => void;
  /** True while a sentence is being read aloud. */
  ttsSpeaking: boolean;
  /** Timestamp of the last recognized sign word (for the happy hop). */
  lastWordAt: number | null;
  /** Reference motion for the clip playing now, if it has been extracted. */
  replayFrame: Float32Array | null;
}

export function SignVideoPanel({
  isl,
  recognition,
  onPhrase,
  ttsSpeaking,
  lastWordAt,
  replayFrame,
}: Props) {
  const { manifest, queue, nowPlaying, paused, lastUnmatched, next, togglePause, clearQueue } = isl;
  const videoRef = useRef<HTMLVideoElement>(null);
  const [missing, setMissing] = useState(false);
  const [typed, setTyped] = useState("");
  const [happy, setHappy] = useState(false);

  /* A recognized sign gives the orb a brief hop. */
  useEffect(() => {
    if (lastWordAt === null) return;
    setHappy(true);
    const timer = window.setTimeout(() => setHappy(false), HAPPY_MS);
    return () => window.clearTimeout(timer);
  }, [lastWordAt]);

  // A missing clip file shows a card, then advances on a timer.
  useEffect(() => {
    setMissing(false);
  }, [nowPlaying?.id]);

  useEffect(() => {
    if (!missing) return;
    const timer = window.setTimeout(next, MISSING_CLIP_MS);
    return () => window.clearTimeout(timer);
  }, [missing, next]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !nowPlaying || missing) return;
    if (paused) video.pause();
    else void video.play().catch(() => {});
  }, [paused, nowPlaying, missing]);

  const submitTyped = () => {
    const text = typed.trim();
    if (!text) return;
    onPhrase(text);
    setTyped("");
  };

  const clipCount = manifest ? Object.keys(manifest).length : 0;
  const showVideo = nowPlaying && !missing;

  /* Orb state: most-specific activity wins. */
  const orbState: OrbState = recognition.error
    ? "confused"
    : happy
      ? "happy"
      : ttsSpeaking
        ? "speaking"
        : recognition.listening || typed.length > 0
          ? "listening"
          : nowPlaying
            ? "thinking"
            : "idle";

  return (
    <section className="panel panel--video" aria-label="ISL video playback">
      <header className="panel__head">
        <h2 className="panel__title">Speech → ISL</h2>
        <span className={`badge ${queue.length ? "badge--live" : "badge--pending"}`}>
          {queue.length ? `${queue.length} queued` : `${clipCount} clips`}
        </span>
      </header>

      <div className="panel__body panel__body--flush isl">
        <div className={`isl__stage ${showVideo ? "isl__stage--video" : ""}`}>
          {showVideo && (
            <video
              ref={videoRef}
              key={nowPlaying.id}
              className="isl__video"
              src={nowPlaying.src}
              autoPlay
              muted
              playsInline
              onEnded={next}
              onError={() => setMissing(true)}
              // Chrome pauses muted video in occluded tabs to save power;
              // resume unless the pause is ours or the clip just ended.
              onPause={(e) => {
                const el = e.currentTarget;
                if (!paused && !el.ended) void el.play().catch(() => {});
              }}
            />
          )}

          {nowPlaying && missing && (
            <div className="isl__card">
              <p className="isl__card-word">{nowPlaying.word}</p>
              <p className="isl__card-note">
                clip file missing — supply clips/{manifest?.[nowPlaying.word]}
              </p>
            </div>
          )}

          {!nowPlaying && (
            <div className="isl__card">
              <Orb state={orbState} size={130} />
              <p className="isl__card-note">
                {clipCount === 0
                  ? "No clips supplied yet. Add videos to frontend/public/clips/ and list them in manifest.json."
                  : orbState === "listening"
                    ? "Listening…"
                    : orbState === "speaking"
                      ? "Speaking the sentence aloud…"
                      : "Speak or type a phrase to play its signs."}
              </p>
            </div>
          )}

          {nowPlaying && (
            <span className="isl__now">{nowPlaying.word}</span>
          )}

          {showVideo && replayFrame && (
            <div className="isl__avatar" title="the same sign as 3D motion">
              <SignAvatar3D frame={replayFrame} height={110} />
            </div>
          )}
        </div>

        {queue.length > 1 && (
          <div className="isl__queue" aria-label="Playback queue">
            {queue.slice(1, 7).map((item) => (
              <span key={item.id} className="chip">
                {item.word}
              </span>
            ))}
            {queue.length > 7 && (
              <span className="chip chip--faint">+{queue.length - 7}</span>
            )}
          </div>
        )}

        {lastUnmatched.length > 0 && (
          <p className="isl__unmatched">
            no clip for: {lastUnmatched.join(", ")}
          </p>
        )}

        <div className="isl__input">
          {recognition.supported ? (
            <button
              type="button"
              className={`button ${recognition.listening ? "button--recording" : ""}`}
              onClick={recognition.listening ? recognition.stop : recognition.start}
            >
              {recognition.listening ? "● Listening…" : "🎤 Listen"}
            </button>
          ) : (
            <span className="isl__nosr" title="SpeechRecognition needs Chrome or Edge">
              mic n/a
            </span>
          )}
          <input
            className="input"
            type="text"
            placeholder={recognition.interim || "or type a phrase…"}
            value={typed}
            onChange={(e) => setTyped(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") submitTyped();
            }}
          />
          <button type="button" className="button" onClick={submitTyped} disabled={!typed.trim()}>
            Play
          </button>
        </div>

        {recognition.error && <p className="isl__error">{recognition.error}</p>}
      </div>

      <footer className="panel__foot">
        <button type="button" className="button" onClick={togglePause} disabled={!nowPlaying}>
          {paused ? "Resume" : "Pause"}
        </button>
        <button type="button" className="button" onClick={next} disabled={!nowPlaying}>
          Skip
        </button>
        <button
          type="button"
          className="button button--quiet"
          onClick={clearQueue}
          disabled={queue.length === 0}
        >
          Clear queue
        </button>
      </footer>
    </section>
  );
}

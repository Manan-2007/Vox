/**
 * Right column: speech -> ISL playback (P10).
 *
 * Spoken (or typed) English is glossed into tokens; tokens with a clip in
 * public/clips/manifest.json queue up and play in sequence. Clips are
 * user-supplied — a queued word whose file is missing shows a card briefly and
 * advances instead of stalling. Browsers without SpeechRecognition fall back
 * to the typed input, which drives the identical pipeline.
 */
import { useEffect, useRef, useState } from "react";
import type { IslQueue } from "../isl/useIslQueue";
import type { SpeechRecognitionState } from "../hooks/useSpeechRecognition";

const MISSING_CLIP_MS = 1400;

interface Props {
  isl: IslQueue;
  recognition: SpeechRecognitionState;
  /** Typed phrases follow the same path as heard speech (transcript + queue). */
  onPhrase: (text: string) => void;
}

export function SignVideoPanel({ isl, recognition, onPhrase }: Props) {
  const { manifest, queue, nowPlaying, paused, lastUnmatched, next, togglePause, clearQueue } = isl;
  const videoRef = useRef<HTMLVideoElement>(null);
  const [missing, setMissing] = useState(false);
  const [typed, setTyped] = useState("");

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

  return (
    <section className="panel panel--video" aria-label="ISL video playback">
      <header className="panel__head">
        <h2 className="panel__title">Speech → ISL</h2>
        <span className={`badge ${queue.length ? "badge--live" : "badge--pending"}`}>
          {queue.length ? `${queue.length} queued` : `${clipCount} clips`}
        </span>
      </header>

      <div className="panel__body panel__body--flush isl">
        <div className="isl__stage">
          {nowPlaying && !missing && (
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
            <div className="isl__card isl__card--idle">
              <p className="isl__card-note">
                {clipCount === 0
                  ? "No clips supplied yet. Add videos to frontend/public/clips/ and list them in manifest.json."
                  : "Speak or type a phrase to play its signs."}
              </p>
            </div>
          )}

          {nowPlaying && (
            <span className="isl__now">{nowPlaying.word}</span>
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

/**
 * ISL clip queue (P10): manifest loading, glossing a phrase into queued clips,
 * and sequential playback state. The <video> element lives in ISLPlayer; it
 * calls `next()` when a clip ends or errors.
 *
 * Clips are user-supplied files under public/clips/, keyed by
 * public/clips/manifest.json. A manifest entry whose file is missing shows a
 * card and auto-advances rather than stalling the queue.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { gloss, type GlossResult } from "./gloss";

const MANIFEST_URL = "/clips/manifest.json";
const MOTION_URL = "/clips/motion.json";

export interface QueueItem {
  id: number;
  word: string;
  src: string; // resolved clip URL
}

/** Reference motion per word: (T, 141) frames extracted from the clip. */
export type MotionLibrary = Record<string, number[][]>;

export interface IslQueue {
  /** null while loading; empty object if the manifest is absent/invalid. */
  manifest: Record<string, string> | null;
  available: ReadonlySet<string>;
  queue: QueueItem[];
  nowPlaying: QueueItem | null;
  paused: boolean;
  /** Words from the last phrase that had no clip. */
  lastUnmatched: string[];
  /** Reference 3D motion for the clip currently playing, frame by frame. */
  replayFrame: Float32Array | null;
  enqueuePhrase: (text: string) => GlossResult;
  next: () => void;
  togglePause: () => void;
  clearQueue: () => void;
}

let nextId = 1;

export function useIslQueue(): IslQueue {
  const [manifest, setManifest] = useState<Record<string, string> | null>(null);
  const [queue, setQueue] = useState<QueueItem[]>([]);
  const [paused, setPaused] = useState(false);
  const [lastUnmatched, setLastUnmatched] = useState<string[]>([]);
  const [replayFrame, setReplayFrame] = useState<Float32Array | null>(null);
  const availableRef = useRef<Set<string>>(new Set());
  const motionRef = useRef<MotionLibrary>({});

  useEffect(() => {
    let disposed = false;
    fetch(MANIFEST_URL)
      .then((res) => (res.ok ? res.json() : Promise.reject(res.status)))
      .then((data: unknown) => {
        if (disposed) return;
        const clips =
          data && typeof data === "object" && "clips" in data
            ? (data as { clips: Record<string, string> }).clips
            : {};
        setManifest(clips);
        availableRef.current = new Set(Object.keys(clips));
      })
      .catch(() => {
        if (disposed) return;
        setManifest({}); // no manifest yet — panel explains what to supply
        availableRef.current = new Set();
      });
    return () => {
      disposed = true;
    };
  }, []);

  /* Reference motion, extracted from the same clips by ml/clip_motion.py.
     Optional: without it the player just shows the video. */
  useEffect(() => {
    let disposed = false;
    fetch(MOTION_URL)
      .then((res) => (res.ok ? res.json() : Promise.reject(res.status)))
      .then((data: MotionLibrary) => {
        if (!disposed) motionRef.current = data ?? {};
      })
      .catch(() => {});
    return () => {
      disposed = true;
    };
  }, []);

  /* Play the reference motion in step with the clip that is on screen. */
  const playingWord = queue[0]?.word ?? null;
  useEffect(() => {
    if (!playingWord) {
      setReplayFrame(null);
      return;
    }
    const frames = motionRef.current[playingWord];
    if (!frames?.length) {
      setReplayFrame(null);
      return;
    }
    let index = 0;
    setReplayFrame(Float32Array.from(frames[0]));
    const timer = window.setInterval(() => {
      index = (index + 1) % frames.length;
      setReplayFrame(Float32Array.from(frames[index]));
    }, 1000 / 15); // clips were re-encoded at 15 fps
    return () => window.clearInterval(timer);
  }, [playingWord]);

  const enqueuePhrase = useCallback(
    (text: string): GlossResult => {
      const result = gloss(text, availableRef.current);
      setLastUnmatched(result.unmatched);
      if (result.matched.length) {
        setQueue((existing) => [
          ...existing,
          ...result.matched.map((word) => ({
            id: nextId++,
            word,
            src: `/clips/${(manifest ?? {})[word]}`,
          })),
        ]);
        setPaused(false);
      }
      return result;
    },
    [manifest],
  );

  const next = useCallback(() => setQueue((q) => q.slice(1)), []);
  const togglePause = useCallback(() => setPaused((p) => !p), []);
  const clearQueue = useCallback(() => {
    setQueue([]);
    setPaused(false);
  }, []);

  return {
    manifest,
    available: availableRef.current,
    queue,
    nowPlaying: queue[0] ?? null,
    paused,
    lastUnmatched,
    replayFrame,
    enqueuePhrase,
    next,
    togglePause,
    clearQueue,
  };
}

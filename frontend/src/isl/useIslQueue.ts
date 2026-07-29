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

export interface QueueItem {
  id: number;
  word: string;
  src: string; // resolved clip URL
}

export interface IslQueue {
  /** null while loading; empty object if the manifest is absent/invalid. */
  manifest: Record<string, string> | null;
  available: ReadonlySet<string>;
  queue: QueueItem[];
  nowPlaying: QueueItem | null;
  paused: boolean;
  /** Words from the last phrase that had no clip. */
  lastUnmatched: string[];
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
  const availableRef = useRef<Set<string>>(new Set());

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
    enqueuePhrase,
    next,
    togglePause,
    clearQueue,
  };
}

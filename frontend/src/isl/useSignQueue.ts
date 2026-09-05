/**
 * Turning what a hearing person said into something the avatar can sign.
 *
 * Replaces the old clip queue. The pipeline is the same shape — text in, signs
 * out — but each step got the part it was missing:
 *
 *   text -> ISL gloss      now reordered by ./grammar.ts, not word-for-word
 *   gloss -> motion        now landmark motion fetched per sign, not a video file
 *   motion -> screen       now one 3D rig, not a <video> with a skeleton beneath it
 *
 * Words with no sign in the library are kept in the queue and flagged rather
 * than dropped, because "there is no sign for AMBULANCE yet" is information and
 * a silently shorter sentence is a lie.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import {
  loadManifest,
  loadSign,
  prefetchSigns,
  type ManifestEntry,
  type QueueItem,
} from "../avatar/signMotion";
import { toGloss, type GlossResult } from "./grammar";

export interface SignQueue {
  /** null while the library manifest is loading. */
  signs: Record<string, ManifestEntry> | null;
  available: ReadonlySet<string>;
  queue: QueueItem[];
  /** The last phrase that was glossed, for showing the ISL notation. */
  lastGloss: GlossResult | null;
  /** Words in that phrase with no sign in the library. */
  unmatched: string[];
  loading: boolean;
  /** Gloss a phrase, load its signs, and queue them. */
  speak: (text: string) => Promise<GlossResult>;
  clear: () => void;
}

export function useSignQueue(): SignQueue {
  const [signs, setSigns] = useState<Record<string, ManifestEntry> | null>(null);
  const [queue, setQueue] = useState<QueueItem[]>([]);
  const [lastGloss, setLastGloss] = useState<GlossResult | null>(null);
  const [unmatched, setUnmatched] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const availableRef = useRef<Set<string>>(new Set());
  /** Guards against an earlier, slower phrase overwriting a later one. */
  const requestRef = useRef(0);

  useEffect(() => {
    let disposed = false;
    void loadManifest().then((manifest) => {
      if (disposed) return;
      setSigns(manifest.signs ?? {});
      availableRef.current = new Set(Object.keys(manifest.signs ?? {}));
      // The words most likely to be needed first, so the first phrase of a
      // session does not wait on the network.
      prefetchSigns(
        ["hello", "yes", "no", "thankyou", "please", "sorry", "help", "what"]
          .filter((gloss) => availableRef.current.has(gloss)),
      );
    });
    return () => {
      disposed = true;
    };
  }, []);

  const speak = useCallback(async (text: string): Promise<GlossResult> => {
    const result = toGloss(text, availableRef.current);
    setLastGloss(result);
    setUnmatched(result.unmatched);
    if (result.tokens.length === 0) {
      setQueue([]);
      return result;
    }

    const request = ++requestRef.current;
    setLoading(true);
    prefetchSigns(result.tokens.filter((t) => !t.missing).map((t) => t.gloss));

    const items = await Promise.all(
      result.tokens.map(async (token): Promise<QueueItem> => {
        const motion = token.missing ? null : await loadSign(token.gloss);
        return {
          gloss: token.gloss,
          label: token.label,
          motion,
          missing: motion === null,
          // The clause-type marking the grammar decided on. It travels with the
          // sign because it is scoped to the sign: a head shake that starts one
          // word early negates the wrong thing.
          face: token.face,
        };
      }),
    );

    if (request !== requestRef.current) return result; // superseded
    setQueue(items);
    setLoading(false);
    return result;
  }, []);

  const clear = useCallback(() => {
    requestRef.current += 1;
    setQueue([]);
    setLastGloss(null);
    setUnmatched([]);
    setLoading(false);
  }, []);

  return {
    signs,
    available: availableRef.current,
    queue,
    lastGloss,
    unmatched,
    loading,
    speak,
    clear,
  };
}

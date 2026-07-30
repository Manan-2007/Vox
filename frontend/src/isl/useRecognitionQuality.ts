/**
 * What the recogniser has actually been measured at, loaded from the file
 * ml/evaluate.py writes.
 *
 * This exists so the interface can tell the truth about itself without anyone
 * having to remember to update a number in the copy. A sign-language tool that
 * presents every prediction with the same confidence is the thing that gets
 * someone into trouble: a word measured at 95% on held-out recordings and a word
 * that has never been measured at all should not look identical on screen.
 *
 * Missing file is not an error — a fresh clone has no metrics until the model is
 * trained, and the app works fine without them; it simply claims less.
 */
import { useEffect, useState } from "react";

export interface RecognitionQuality {
  classes: number;
  /** Top-1 accuracy over the held-out test split. */
  top1: number;
  top3: number;
  /** How often the recogniser is right *when it speaks* — the number that matters. */
  gatedPrecision: number;
  /** Words scoring 80%+ on recordings never trained on. */
  reliable: ReadonlySet<string>;
  /** Words with any held-out recording at all; the rest are unmeasured. */
  measured: ReadonlySet<string>;
  loaded: boolean;
}

const EMPTY: RecognitionQuality = {
  classes: 0,
  top1: 0,
  top3: 0,
  gatedPrecision: 0,
  reliable: new Set(),
  measured: new Set(),
  loaded: false,
};

export function useRecognitionQuality(): RecognitionQuality {
  const [quality, setQuality] = useState<RecognitionQuality>(EMPTY);

  useEffect(() => {
    let disposed = false;
    fetch("/signs/recognition.json")
      .then((response) => (response.ok ? response.json() : null))
      .then((data) => {
        if (disposed || !data) return;
        setQuality({
          classes: data.classes ?? 0,
          top1: data.top1 ?? 0,
          top3: data.top3 ?? 0,
          gatedPrecision: data.gatedPrecision ?? 0,
          reliable: new Set<string>(data.reliable ?? []),
          measured: new Set<string>(data.measured ?? []),
          loaded: true,
        });
      })
      .catch(() => {});
    return () => {
      disposed = true;
    };
  }, []);

  return quality;
}

export type Confidence = "verified" | "measured" | "unmeasured";

/** How much this word's prediction should be trusted. */
export function confidenceOf(
  gloss: string,
  quality: RecognitionQuality,
): Confidence {
  if (quality.reliable.has(gloss)) return "verified";
  if (quality.measured.has(gloss)) return "measured";
  return "unmeasured";
}

/**
 * What the recogniser has decided — and, when it has not decided, why.
 *
 * ---------------------------------------------------------------------------
 * THE BUG THIS FILE EXISTED TO CAUSE
 * ---------------------------------------------------------------------------
 * The previous version rendered `top3[0].word` in the headline slot whenever a
 * candidate existed:
 *
 *     {!active ? "Recognition paused" : lead ? lead.word : …}
 *
 * The backend was correct — it gates emission on confidence, stability and the
 * reject class, and it refused. But it also sends its top-3 on every frame for
 * the diagnostic strip, and the headline showed the argmax regardless. With 242
 * classes at 38% top-1 accuracy, that argmax is usually a word with two training
 * samples and no measured accuracy at all.
 *
 * That is the reported "I open both hands and it says Headache". The application
 * was not inventing a gesture; it was displaying a REJECTED one in the position
 * reserved for accepted ones. `headache` has nine training samples and 0%
 * held-out accuracy.
 *
 * ---------------------------------------------------------------------------
 * THE RULE NOW
 * ---------------------------------------------------------------------------
 * The headline shows a word ONLY when the backend accepted it. Everything else
 * says what is actually happening, and the candidate list is visibly separated
 * as diagnostics rather than results — labelled, dimmed, and never in the slot a
 * user reads as an answer.
 */

import type { Candidate, LiveGuess, RejectReason } from "../hooks/useVoxSocket";

interface Props {
  top3: Candidate[];
  live: LiveGuess | null;
  threshold: number;
  quality: { hands: number; body: boolean; usable: boolean } | null;
  active: boolean;
  /** The last word the backend actually accepted, if it is still recent. */
  accepted: { word: string; confidence: number } | null;
}

/** Plain-English reasons. Each names a fix the user can act on. */
const REASONS: Record<RejectReason, string> = {
  resting: "Hands at rest",
  unverified:
    "Gesture not recognised — the closest match is not one of the verified words",
  "low-confidence": "Gesture not recognised",
  unstable: "Hold the sign a moment longer",
  "already-emitted": "Sign registered — return to rest, then sign again",
};

export function Recognition({
  top3,
  live,
  threshold,
  quality,
  active,
  accepted,
}: Props) {
  const confidence = live?.confidence ?? 0;

  /* The headline. A word appears here only if it was ACCEPTED. */
  let headline: string;
  let tone: "accepted" | "idle" | "rejected";

  if (!active) {
    headline = "Recognition paused";
    tone = "idle";
  } else if (accepted) {
    headline = accepted.word;
    tone = "accepted";
  } else if (quality?.usable === false) {
    headline = "Waiting for a clear view";
    tone = "idle";
  } else if (!live) {
    headline = "Listening for a sign";
    tone = "idle";
  } else {
    headline = REASONS[live.reason] ?? "Gesture not recognised";
    tone = live.reason === "resting" || live.reason === "already-emitted"
      ? "idle"
      : "rejected";
  }

  return (
    <div className="readout">
      <div className="readout__head">
        <span className={`readout__word readout__word--${tone}`}>{headline}</span>
        {active && accepted && (
          <span className="readout__pct">
            {Math.round(accepted.confidence * 100)}%
          </span>
        )}
      </div>

      <div className="meter">
        <div
          className="meter__fill"
          style={{ width: `${Math.round((active ? confidence : 0) * 100)}%` }}
        />
        <div
          className="meter__threshold"
          style={{ left: `${Math.round(threshold * 100)}%` }}
          title={`A word is accepted above ${Math.round(threshold * 100)}%`}
        />
      </div>

      {active && top3.length > 0 && (
        <div className="ranked">
          {/* Labelled, so this can never again be mistaken for a result. */}
          <p className="ranked__label">
            Candidates being considered — not recognised words
          </p>
          {top3.map((entry, index) => (
            <div
              key={entry.word}
              className={`ranked__row ${index === 0 ? "ranked__row--lead" : ""}`}
            >
              <span className="ranked__word">{entry.word}</span>
              <span className="ranked__track">
                <span
                  className="ranked__fill"
                  style={{ width: `${Math.round(entry.confidence * 100)}%` }}
                />
              </span>
              <span className="ranked__pct">
                {Math.round(entry.confidence * 100)}%
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

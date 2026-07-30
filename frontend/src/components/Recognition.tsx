/**
 * What the recogniser is thinking, shown continuously.
 *
 * A classifier that stays silent until it is 85% sure is, from the outside,
 * indistinguishable from one that is broken. This strip is the fix: the top
 * three candidates and their scores, always visible, with the acceptance
 * threshold drawn on the meter. A sign that is nearly recognised then looks
 * different from one the model has never heard of, and a user can tell whether
 * to hold the sign longer, move closer, or give up on that word.
 */
import type { Candidate } from "../hooks/useVoxSocket";

interface Props {
  top3: Candidate[];
  threshold: number;
  quality: { hands: number; body: boolean; usable: boolean } | null;
  active: boolean;
}

export function Recognition({ top3, threshold, quality, active }: Props) {
  const lead = top3[0];
  const confidence = lead?.confidence ?? 0;

  return (
    <div className="readout">
      <div className="readout__head">
        <span
          className={`readout__word ${lead && active ? "" : "readout__word--idle"}`}
        >
          {!active
            ? "Recognition paused"
            : lead
              ? lead.word
              : quality?.usable === false
                ? "Waiting for a clear view"
                : "Listening for a sign"}
        </span>
        {active && lead && (
          <span className="readout__pct">{Math.round(confidence * 100)}%</span>
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

      {active && top3.length > 1 && (
        <div className="ranked">
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

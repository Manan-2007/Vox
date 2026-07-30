/**
 * The conversation: the shared record both people are talking through.
 *
 * Each signed turn shows two lines — the English sentence, and beneath it the
 * gloss that was actually recognised. Showing both is not a debug affordance.
 * The English is a reconstruction: the grammar engine put back a copula, an
 * article and a tense that ISL never signed. If it reconstructs wrongly, the
 * gloss underneath is how anyone can see that it did, and a Deaf user can tell
 * immediately whether the machine heard them correctly or invented a sentence.
 */
import { useEffect, useRef } from "react";
import type { Conversation as ConversationState, Turn } from "../hooks/useConversation";
import { toEnglish } from "../isl/grammar";
import {
  confidenceOf,
  type RecognitionQuality,
} from "../isl/useRecognitionQuality";

interface Props {
  conversation: ConversationState;
  quality: RecognitionQuality;
  /** Commit the in-progress signed turn now, rather than waiting for the pause. */
  onFinishTurn: () => void;
  speaking: boolean;
  voiceOn: boolean;
}

export function Conversation({
  conversation,
  quality,
  onFinishTurn,
  speaking,
  voiceOn,
}: Props) {
  const endRef = useRef<HTMLDivElement>(null);
  const { turns, current } = conversation;

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns.length, current.words.length]);

  const empty = turns.length === 0 && current.words.length === 0;

  return (
    <div className="thread">
      {empty ? (
        <p className="thread__empty">
          Nothing said yet. Sign to the camera, or type below and watch it
          signed back.
        </p>
      ) : (
        <>
          {turns.map((turn) => (
            <TurnBubble key={turn.id} turn={turn} quality={quality} />
          ))}
          {current.words.length > 0 && (
            <TurnBubble
              turn={current}
              quality={quality}
              live
              onFinish={onFinishTurn}
            />
          )}
        </>
      )}
      {speaking && voiceOn && (
        <p className="turn__meta" style={{ alignSelf: "center" }}>
          speaking aloud…
        </p>
      )}
      <div ref={endRef} />
    </div>
  );
}

function TurnBubble({
  turn,
  quality,
  live = false,
  onFinish,
}: {
  turn: Turn;
  quality: RecognitionQuality;
  live?: boolean;
  onFinish?: () => void;
}) {
  const glosses = turn.words.map((word) => word.text);
  const signed = turn.speaker === "signer";
  // The hearing side already arrives as English; only the signed side is
  // reconstructed from gloss.
  const text = signed ? toEnglish(glosses) : glosses.join(" ");

  const weakest = turn.words.reduce(
    (lowest, word) => Math.min(lowest, word.confidence),
    1,
  );

  return (
    <article
      className={`turn turn--${signed ? "signer" : "other"} ${live ? "turn--live" : ""}`}
    >
      <span className="turn__who">{signed ? "Signed" : "Spoken"}</span>
      <p className="turn__text">
        {text}
        {live && <span className="caret" aria-hidden />}
      </p>
      {signed && (
        <span className="turn__gloss">
          {glosses.map((gloss, index) => {
            const confidence = quality.loaded
              ? confidenceOf(gloss, quality)
              : "measured";
            return (
              <span
                key={`${gloss}-${index}`}
                className={`glossword glossword--${confidence}`}
                title={
                  confidence === "verified"
                    ? "80%+ correct on recordings this model never trained on"
                    : confidence === "measured"
                      ? "measured, but below 80% on held-out recordings"
                      : "never measured — only one recording of this sign exists, so nothing could be held back to test it"
                }
              >
                {gloss.toUpperCase()}
              </span>
            );
          })}
        </span>
      )}
      {signed && !live && weakest < 0.92 && (
        <span className="turn__meta">
          lowest confidence {Math.round(weakest * 100)}%
        </span>
      )}
      {live && onFinish && (
        <span className="turn__meta">
          <button type="button" className="btn btn--ghost" onClick={onFinish}>
            Finish sentence
          </button>
        </span>
      )}
    </article>
  );
}

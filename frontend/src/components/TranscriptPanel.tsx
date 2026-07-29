/**
 * Centre column: the running conversation.
 *
 * Committed turns and the in-progress turn share one scroll region that sticks
 * to the bottom, so the newest text is always in view and the panel itself
 * never scrolls off the page.
 */
import { useEffect, useRef } from "react";
import { turnText, type Conversation } from "../hooks/useConversation";

interface Props {
  conversation: Conversation;
}

export function TranscriptPanel({ conversation }: Props) {
  const { turns, current, newTurn, undoWord, clear } = conversation;
  const logRef = useRef<HTMLDivElement>(null);

  // Pin to the bottom as words arrive. Set scrollTop directly rather than using
  // a smooth scrollIntoView: words land every few hundred ms, and a smooth
  // animation is restarted by the next word before it ever reaches the end, so
  // the newest turn drifts out of view exactly when the log gets long.
  useEffect(() => {
    const log = logRef.current;
    if (log) log.scrollTop = log.scrollHeight;
  }, [turns.length, current.words.length]);

  const isEmpty = turns.length === 0 && current.words.length === 0;

  return (
    <section className="panel panel--transcript" aria-label="Conversation">
      <header className="panel__head">
        <h2 className="panel__title">Conversation</h2>
        <span className="panel__count">
          {turns.length + (current.words.length ? 1 : 0)} turn
          {turns.length + (current.words.length ? 1 : 0) === 1 ? "" : "s"}
        </span>
      </header>

      <div ref={logRef} className="panel__body transcript" role="log" aria-live="polite">
        {isEmpty && (
          <p className="transcript__empty">
            Sign to begin. Confirmed words appear here and build into a sentence.
          </p>
        )}

        {turns.map((turn) => (
          <article key={turn.id} className={`bubble bubble--${turn.speaker}`}>
            <p className="bubble__text">{turnText(turn)}</p>
          </article>
        ))}

        {current.words.length > 0 && (
          <article className="bubble bubble--signer bubble--active">
            <p className="bubble__text">
              {current.words.map((word) => (
                <span key={word.id} className="bubble__word" title={`${Math.round(word.confidence * 100)}% confident`}>
                  {word.text}
                </span>
              ))}
              <span className="bubble__caret" aria-hidden />
            </p>
          </article>
        )}

      </div>

      <footer className="panel__foot">
        <button
          type="button"
          className="button button--primary"
          onClick={newTurn}
          disabled={current.words.length === 0}
        >
          New sentence
        </button>
        <button
          type="button"
          className="button"
          onClick={undoWord}
          disabled={current.words.length === 0}
        >
          Undo word
        </button>
        <button type="button" className="button button--quiet" onClick={clear} disabled={isEmpty}>
          Clear
        </button>
      </footer>
    </section>
  );
}

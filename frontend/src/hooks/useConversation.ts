/**
 * Conversation state: confirmed words accumulate into the current turn, and a
 * turn is committed to the transcript when the signer starts a new one.
 *
 * Turns carry a `speaker`: the signer's side is built word-by-word from the
 * recognizer; the other side arrives as whole utterances from speech
 * recognition (`addUtterance`). `spoken` marks turns that were read aloud.
 *
 * `turns` and `current` live in ONE state value updated by pure updaters.
 * An earlier version called setTurns() inside a setCurrent() updater — a side
 * effect in an updater, which StrictMode double-invokes, committing every
 * turn twice. Keep updaters pure.
 */
import { useCallback, useState } from "react";
import type { ConfirmedWord } from "./useVoxSocket";

export type Speaker = "signer" | "other";

export interface TurnWord {
  id: number;
  text: string;
  confidence: number;
}

export interface Turn {
  id: number;
  speaker: Speaker;
  words: TurnWord[];
  /** True when this turn was read aloud by TTS. */
  spoken?: boolean;
}

interface ConversationState {
  turns: Turn[];
  current: Turn;
}

export interface Conversation {
  /** Committed turns, oldest first. */
  turns: Turn[];
  /** The signer turn currently being built. Never null. */
  current: Turn;
  appendWord: (word: ConfirmedWord) => void;
  /** Commit the current turn and start an empty one. No-op when empty. */
  newTurn: (options?: { spoken?: boolean }) => void;
  /** Append a whole utterance (the hearing side) directly to the transcript. */
  addUtterance: (text: string, speaker?: Speaker) => void;
  /** Drop the last word of the current turn. */
  undoWord: () => void;
  clear: () => void;
}

let nextId = 1;
const newId = () => nextId++;

const emptyTurn = (speaker: Speaker = "signer"): Turn => ({
  id: newId(),
  speaker,
  words: [],
});

export function useConversation(): Conversation {
  const [state, setState] = useState<ConversationState>(() => ({
    turns: [],
    current: emptyTurn(),
  }));

  const appendWord = useCallback(({ word, confidence }: ConfirmedWord) => {
    setState((s) => ({
      ...s,
      current: {
        ...s.current,
        words: [...s.current.words, { id: newId(), text: word, confidence }],
      },
    }));
  }, []);

  const newTurn = useCallback((options?: { spoken?: boolean }) => {
    setState((s) => {
      if (s.current.words.length === 0) return s; // nothing to commit
      return {
        turns: [...s.turns, { ...s.current, spoken: options?.spoken ?? false }],
        current: emptyTurn(s.current.speaker),
      };
    });
  }, []);

  const addUtterance = useCallback((text: string, speaker: Speaker = "other") => {
    const trimmed = text.trim();
    if (!trimmed) return;
    const turn: Turn = {
      id: newId(),
      speaker,
      words: trimmed
        .split(/\s+/)
        .map((word) => ({ id: newId(), text: word, confidence: 1 })),
    };
    setState((s) => ({ ...s, turns: [...s.turns, turn] }));
  }, []);

  const undoWord = useCallback(() => {
    setState((s) => ({
      ...s,
      current: { ...s.current, words: s.current.words.slice(0, -1) },
    }));
  }, []);

  const clear = useCallback(() => {
    setState({ turns: [], current: emptyTurn() });
  }, []);

  return {
    turns: state.turns,
    current: state.current,
    appendWord,
    newTurn,
    addUtterance,
    undoWord,
    clear,
  };
}

export const turnText = (turn: Turn): string =>
  turn.words.map((word) => word.text).join(" ");

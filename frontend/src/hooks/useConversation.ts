/**
 * Conversation state: confirmed words accumulate into the current turn, and a
 * turn is committed to the transcript when the signer starts a new one.
 *
 * Turns carry a `speaker` so the speech side can add its own bubbles later
 * without reshaping this; for now only the signer produces text.
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
}

export interface Conversation {
  /** Committed turns, oldest first. */
  turns: Turn[];
  /** The turn currently being built. Never null. */
  current: Turn;
  appendWord: (word: ConfirmedWord) => void;
  /** Commit the current turn and start an empty one. No-op when empty. */
  newTurn: () => void;
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
  const [turns, setTurns] = useState<Turn[]>([]);
  const [current, setCurrent] = useState<Turn>(() => emptyTurn());

  const appendWord = useCallback(({ word, confidence }: ConfirmedWord) => {
    setCurrent((turn) => ({
      ...turn,
      words: [...turn.words, { id: newId(), text: word, confidence }],
    }));
  }, []);

  const newTurn = useCallback(() => {
    setCurrent((turn) => {
      if (turn.words.length === 0) return turn; // nothing to commit
      setTurns((committed) => [...committed, turn]);
      return emptyTurn(turn.speaker);
    });
  }, []);

  const undoWord = useCallback(() => {
    setCurrent((turn) => ({ ...turn, words: turn.words.slice(0, -1) }));
  }, []);

  const clear = useCallback(() => {
    setTurns([]);
    setCurrent(emptyTurn());
  }, []);

  return { turns, current, appendWord, newTurn, undoWord, clear };
}

export const turnText = (turn: Turn): string =>
  turn.words.map((word) => word.text).join(" ");

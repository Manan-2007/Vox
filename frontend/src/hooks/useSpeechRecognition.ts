/**
 * Speech-to-text via the Web Speech API (P10).
 *
 * SpeechRecognition is Chrome/Edge-only (webkit prefix) and needs both a
 * microphone and, in Chrome, a network connection (recognition runs
 * server-side). `supported` is false elsewhere; the ISL panel then falls back
 * to its typed-phrase input, which exercises the identical gloss -> queue ->
 * playback path.
 */
import { useCallback, useEffect, useRef, useState } from "react";

/* Minimal typings — lib.dom has no SpeechRecognition declarations. */
interface SRAlternative {
  transcript: string;
}
interface SRResult {
  isFinal: boolean;
  0: SRAlternative;
}
interface SREvent {
  resultIndex: number;
  results: { length: number; [index: number]: SRResult };
}
interface SR {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  onresult: ((event: SREvent) => void) | null;
  onend: (() => void) | null;
  onerror: ((event: { error: string }) => void) | null;
  start: () => void;
  stop: () => void;
  abort: () => void;
}
type SRConstructor = new () => SR;

const getSR = (): SRConstructor | null => {
  if (typeof window === "undefined") return null;
  const w = window as unknown as Record<string, unknown>;
  return (w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null) as SRConstructor | null;
};

export interface SpeechRecognitionState {
  supported: boolean;
  listening: boolean;
  interim: string;
  error: string | null;
  start: () => void;
  stop: () => void;
}

/** `onFinal` fires once per finished utterance with the recognized text. */
export function useSpeechRecognition(
  onFinal: (text: string) => void,
): SpeechRecognitionState {
  const ctor = getSR();
  const supported = ctor !== null;

  const onFinalRef = useRef(onFinal);
  onFinalRef.current = onFinal;

  const recognizerRef = useRef<SR | null>(null);
  const [listening, setListening] = useState(false);
  const [interim, setInterim] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    return () => recognizerRef.current?.abort();
  }, []);

  const start = useCallback(() => {
    if (!ctor || recognizerRef.current) return;
    const rec = new ctor();
    rec.lang = "en-IN";
    rec.continuous = false;
    rec.interimResults = true;

    rec.onresult = (event) => {
      let interimText = "";
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const result = event.results[i];
        if (result.isFinal) {
          const text = result[0].transcript.trim();
          if (text) onFinalRef.current(text);
        } else {
          interimText += result[0].transcript;
        }
      }
      setInterim(interimText);
    };
    rec.onend = () => {
      recognizerRef.current = null;
      setListening(false);
      setInterim("");
    };
    rec.onerror = (event) => {
      setError(
        event.error === "not-allowed"
          ? "Microphone permission denied."
          : `Speech recognition error: ${event.error}`,
      );
    };

    recognizerRef.current = rec;
    setError(null);
    setListening(true);
    rec.start();
  }, [ctor]);

  const stop = useCallback(() => {
    recognizerRef.current?.stop();
  }, []);

  return { supported, listening, interim, error, start, stop };
}

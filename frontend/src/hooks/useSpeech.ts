/**
 * Text-to-speech via the Web Speech API (P8).
 *
 * SpeechSynthesis is available in every mainstream browser, but not guaranteed
 * (some WebViews, some Linux builds lack voices). `supported` reflects the API
 * being present; if it is absent the app still builds sentences — they are just
 * not read aloud, and the UI says so instead of failing.
 */
import { useCallback, useEffect, useRef, useState } from "react";

export interface Speech {
  supported: boolean;
  speaking: boolean;
  speak: (text: string) => boolean;
  cancel: () => void;
}

export function useSpeech(): Speech {
  const supported =
    typeof window !== "undefined" && "speechSynthesis" in window;
  const [speaking, setSpeaking] = useState(false);
  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);

  useEffect(() => {
    return () => {
      if (supported) window.speechSynthesis.cancel();
    };
  }, [supported]);

  const speak = useCallback(
    (text: string): boolean => {
      if (!supported || !text.trim()) return false;
      window.speechSynthesis.cancel(); // never queue behind an old sentence
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.95;
      utterance.onstart = () => setSpeaking(true);
      utterance.onend = () => setSpeaking(false);
      utterance.onerror = () => setSpeaking(false);
      utteranceRef.current = utterance; // keep alive; some engines GC mid-speech
      window.speechSynthesis.speak(utterance);
      return true;
    },
    [supported],
  );

  const cancel = useCallback(() => {
    if (supported) window.speechSynthesis.cancel();
    setSpeaking(false);
  }, [supported]);

  return { supported, speaking, speak, cancel };
}

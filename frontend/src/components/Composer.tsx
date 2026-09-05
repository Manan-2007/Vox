/**
 * The hearing side's input: speak or type, and the avatar signs it.
 *
 * Typing is deliberately as prominent as the microphone. Speech recognition
 * fails on accents, in noise, and in the exact places this product is most
 * needed — a hospital corridor, a ticket counter. A text box that always works
 * is not a fallback, it is the reliable path.
 */
import { useCallback, useState } from "react";
import type { GlossResult } from "../isl/grammar";

interface Props {
  onSay: (text: string) => void;
  listening: boolean;
  onToggleListening: () => void;
  speechSupported: boolean;
  /** Live partial transcript while the mic is open. */
  interim: string;
  lastGloss: GlossResult | null;
  busy: boolean;
}

/** Openers worth one tap, chosen for the situations this product is for. */
const QUICK = [
  "How can I help you?",
  "What is your name?",
  "Where is the pain?",
  "Please wait here",
  "Do you need a doctor?",
];

export function Composer({
  onSay,
  listening,
  onToggleListening,
  speechSupported,
  interim,
  lastGloss,
  busy,
}: Props) {
  const [text, setText] = useState("");

  const submit = useCallback(() => {
    const trimmed = text.trim();
    if (!trimmed) return;
    onSay(trimmed);
    setText("");
  }, [text, onSay]);

  return (
    <div className="composer">
      {lastGloss && lastGloss.tokens.length > 0 && (
        <p className="turn__gloss" style={{ marginTop: 0 }}>
          ISL · {lastGloss.notation}
          {/* The non-manual marking is grammar, so it is named rather than left
              for the viewer to notice. Without this the face just looks like
              the avatar having a mood. */}
          {lastGloss.markers.length > 0 && (
            <span className="turn__marker"> · {lastGloss.markers.join(" · ")}</span>
          )}
          {lastGloss.unmatched.length > 0 && (
            <span style={{ color: "var(--warn)" }}>
              {"  ·  no sign yet for "}
              {lastGloss.unmatched.join(", ")}
            </span>
          )}
        </p>
      )}

      <div className="composer__row">
        <button
          type="button"
          className={`mic ${listening ? "is-live" : ""}`}
          onClick={onToggleListening}
          disabled={!speechSupported}
          aria-pressed={listening}
          title={
            speechSupported
              ? listening
                ? "Stop listening"
                : "Speak, and it will be signed"
              : "This browser has no speech recognition — type instead"
          }
        >
          {listening ? "■" : "🎙"}
        </button>

        <input
          className="input"
          value={listening && interim ? interim : text}
          onChange={(event) => setText(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") submit();
          }}
          placeholder={
            listening ? "Listening…" : "Type what you want signed, then press Enter"
          }
          readOnly={listening}
          aria-label="Text to sign"
        />

        <button
          type="button"
          className="btn btn--primary"
          onClick={submit}
          disabled={!text.trim() || busy}
        >
          {busy ? "…" : "Sign it"}
        </button>
      </div>

      <div className="playback__words">
        {QUICK.map((phrase) => (
          <button
            key={phrase}
            type="button"
            className="wordchip"
            onClick={() => onSay(phrase)}
          >
            {phrase}
          </button>
        ))}
      </div>
    </div>
  );
}

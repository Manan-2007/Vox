/**
 * The session: a stage on the left, the conversation on the right.
 *
 * The stage switches focus by itself. While the hearing side is being signed
 * back, it plays the reference motion; the moment that finishes, or the signer
 * starts moving, it returns to mirroring the camera. Nobody has to press a mode
 * button in the middle of a conversation, which is the one thing guaranteed not
 * to happen when two people are actually trying to talk.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { Composer } from "../components/Composer";
import { Conversation } from "../components/Conversation";
import { Recognition } from "../components/Recognition";
import { SettingsDrawer } from "../components/SettingsDrawer";
import { SignStage, type StageMode } from "../components/SignStage";
import { turnText, useConversation } from "../hooks/useConversation";
import { useHandTracking } from "../hooks/useHandTracking";
import { useSpeech } from "../hooks/useSpeech";
import { useSpeechRecognition } from "../hooks/useSpeechRecognition";
import { useVoxSocket, type ConfirmedWord } from "../hooks/useVoxSocket";
import { toEnglish } from "../isl/grammar";
import { useRecognitionQuality } from "../isl/useRecognitionQuality";
import { useSignQueue } from "../isl/useSignQueue";

/** A signed sentence is considered finished after this long without a new word. */
const SENTENCE_PAUSE_MS = 3500;
/** How long the stage stays on the reference after its last sign ends. */
const REFERENCE_HOLD_MS = 2500;
const DEFAULT_THRESHOLD = 0.85;
const VOICE_KEY = "vox-voice-output";

export function SessionPage() {
  const conversation = useConversation();
  const speech = useSpeech();
  const signs = useSignQueue();
  const recognitionQuality = useRecognitionQuality();

  const [settingsOpen, setSettingsOpen] = useState(false);
  const [threshold, setThresholdState] = useState(DEFAULT_THRESHOLD);
  const [stageMode, setStageMode] = useState<StageMode>("live");
  const [voiceOn, setVoiceOnState] = useState<boolean>(() => {
    try {
      const stored = localStorage.getItem(VOICE_KEY);
      return stored === null ? true : stored === "true";
    } catch {
      return true;
    }
  });

  const setVoiceOn = useCallback((value: boolean) => {
    setVoiceOnState(value);
    try {
      localStorage.setItem(VOICE_KEY, String(value));
    } catch {
      /* private mode — the toggle still works for this session */
    }
  }, []);

  // Refs so timers read current state without re-arming on every render.
  const currentRef = useRef(conversation.current);
  currentRef.current = conversation.current;
  const voiceRef = useRef(voiceOn);
  voiceRef.current = voiceOn;
  const pauseTimer = useRef<number | undefined>(undefined);
  const stageTimer = useRef<number | undefined>(undefined);

  /* ------------------------------------------------- sentence completion -- */
  const finishSentence = useCallback(() => {
    window.clearTimeout(pauseTimer.current);
    const turn = currentRef.current;
    if (turn.words.length === 0) return;
    // Speak the reconstructed English, not the raw gloss: "What is your name?"
    // is what the hearing person needs to hear, not "you name what".
    const sentence = toEnglish(turn.words.map((word) => word.text)) || turnText(turn);
    const willSpeak = speech.supported && voiceRef.current;
    conversation.newTurn({ spoken: willSpeak });
    if (willSpeak) speech.speak(sentence);
  }, [conversation, speech]);

  const finishRef = useRef(finishSentence);
  finishRef.current = finishSentence;

  const handleWord = useCallback(
    (word: ConfirmedWord) => {
      conversation.appendWord(word);
      // The signer has taken the floor: stop showing the reference.
      setStageMode("live");
      window.clearTimeout(stageTimer.current);
      window.clearTimeout(pauseTimer.current);
      pauseTimer.current = window.setTimeout(
        () => finishRef.current(),
        SENTENCE_PAUSE_MS,
      );
    },
    [conversation],
  );

  useEffect(
    () => () => {
      window.clearTimeout(pauseTimer.current);
      window.clearTimeout(stageTimer.current);
    },
    [],
  );

  /* ------------------------------------------------------- socket + cam -- */
  const { socket, error, noModel, top3, quality, send, setThreshold } =
    useVoxSocket(handleWord);
  const tracking = useHandTracking(send);

  const handleThreshold = useCallback(
    (value: number) => {
      setThresholdState(value);
      setThreshold(value); // forwarded to the backend, re-sent on reconnect
    },
    [setThreshold],
  );

  /* ------------------------------------------------- speech -> sign path -- */
  const say = useCallback(
    (text: string) => {
      conversation.addUtterance(text, "other");
      setStageMode("reference");
      window.clearTimeout(stageTimer.current);
      void signs.speak(text);
    },
    [conversation, signs],
  );

  const recognition = useSpeechRecognition(say);

  // Once the queue has played through, hand the stage back to the camera.
  const handleQueueDone = useCallback(() => {
    window.clearTimeout(stageTimer.current);
    stageTimer.current = window.setTimeout(
      () => setStageMode("live"),
      REFERENCE_HOLD_MS,
    );
  }, []);

  const ready = socket === "open" && !noModel;

  const status = useMemo(() => {
    if (error) return { tone: "error" as const, text: error };
    if (socket !== "open") return { tone: "error" as const, text: `Backend ${socket}` };
    if (noModel) return { tone: "warn" as const, text: "No recognition model" };
    return { tone: "live" as const, text: "Ready" };
  }, [error, socket, noModel]);

  return (
    <div className="app">
      <header className="topbar">
        <Link to="/" className="brand" aria-label="Vox home">
          <span className="brand__mark" aria-hidden />
          <span className="brand__name">Vox</span>
          <span className="brand__sub">Indian Sign Language interpreter</span>
        </Link>

        <div className="topbar__tools">
          <span className={`chip chip--${status.tone}`}>
            <span className="chip__dot" />
            {status.text}
          </span>
          <button
            type="button"
            className={`btn ${voiceOn && speech.supported ? "btn--primary" : ""}`}
            onClick={() => setVoiceOn(!voiceOn)}
            disabled={!speech.supported}
            aria-pressed={voiceOn}
            title={
              speech.supported
                ? "Read finished signed sentences aloud"
                : "This browser has no speech output"
            }
          >
            {speech.supported ? (voiceOn ? "Voice on" : "Text only") : "No voice"}
          </button>
          <button
            type="button"
            className="btn"
            onClick={() => setSettingsOpen(true)}
          >
            Settings
          </button>
        </div>
      </header>

      <main className="workspace">
        <SignStage
          mode={stageMode}
          queue={signs.queue}
          tracking={tracking}
          ready={ready}
          loadingSigns={signs.loading}
          onQueueDone={handleQueueDone}
        />

        <section className="card card--talk card--signer">
          <header className="card__head">
            <h2 className="card__title">Conversation</h2>
            <button
              type="button"
              className="btn btn--ghost"
              onClick={conversation.clear}
              disabled={conversation.turns.length === 0}
            >
              Clear
            </button>
          </header>

          <div className="card__body">
            <Conversation
              conversation={conversation}
              quality={recognitionQuality}
              onFinishTurn={finishSentence}
              speaking={speech.speaking}
              voiceOn={voiceOn && speech.supported}
            />
            <Recognition
              top3={top3}
              threshold={threshold}
              quality={quality}
              active={ready && tracking.cameraOn}
            />
            <Composer
              onSay={say}
              listening={recognition.listening}
              onToggleListening={
                recognition.listening ? recognition.stop : recognition.start
              }
              speechSupported={recognition.supported}
              interim={recognition.interim}
              lastGloss={signs.lastGloss}
              busy={signs.loading}
            />
          </div>
        </section>
      </main>

      <SettingsDrawer
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        threshold={threshold}
        onThreshold={handleThreshold}
        voiceOn={voiceOn}
        onVoiceOn={setVoiceOn}
        ttsSupported={speech.supported}
        tracking={tracking}
        signCount={signs.signs ? Object.keys(signs.signs).length : 0}
        recognition={recognitionQuality}
      />
    </div>
  );
}

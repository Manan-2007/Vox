/**
 * The session view: camera left, conversation centre, speech -> ISL right.
 *
 * This is the only place the data sources join:
 *   hand tracking -> socket -> confirmed words -> current sentence
 *   sentence completion (pause timeout or button) -> commit + optional TTS
 *   speech recognition -> transcript ("heard") + ISL clip queue
 *
 * Voice output is a first-class toggle in the top bar (persisted): Deaf, mute,
 * or hearing users each pick whether sentences are voiced or stay text-only.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { CameraPanel } from "../components/CameraPanel";
import { SettingsDrawer } from "../components/SettingsDrawer";
import { SignVideoPanel } from "../components/SignVideoPanel";
import { TranscriptPanel } from "../components/TranscriptPanel";
import { turnText, useConversation } from "../hooks/useConversation";
import { useHandTracking } from "../hooks/useHandTracking";
import { useSpeech } from "../hooks/useSpeech";
import { useSpeechRecognition } from "../hooks/useSpeechRecognition";
import { useVoxSocket, type ConfirmedWord } from "../hooks/useVoxSocket";
import { useIslQueue } from "../isl/useIslQueue";

/** A sentence is considered finished after this long without a new word. */
const SENTENCE_PAUSE_MS = 3500;
const DEFAULT_THRESHOLD = 0.85;
const VOICE_KEY = "vox-voice-output";

export function SessionPage() {
  const conversation = useConversation();
  const speech = useSpeech();
  const isl = useIslQueue();

  const [latestWord, setLatestWord] = useState<{
    text: string;
    confidence: number;
    at: number;
  } | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [threshold, setThresholdState] = useState(DEFAULT_THRESHOLD);
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
      /* private mode etc. — the toggle still works for the session */
    }
  }, []);

  // Refs so timers see current state without re-arming on every render.
  const currentRef = useRef(conversation.current);
  currentRef.current = conversation.current;
  const voiceRef = useRef(voiceOn);
  voiceRef.current = voiceOn;
  const pauseTimer = useRef<number | undefined>(undefined);

  /* ------------------------------------------------- sentence completion -- */
  const finishSentence = useCallback(() => {
    window.clearTimeout(pauseTimer.current);
    const turn = currentRef.current;
    if (turn.words.length === 0) return;
    const text = turnText(turn);
    const willSpeak = speech.supported && voiceRef.current;
    conversation.newTurn({ spoken: willSpeak });
    if (willSpeak) speech.speak(text);
  }, [conversation, speech]);

  const finishRef = useRef(finishSentence);
  finishRef.current = finishSentence;

  const handleWord = useCallback(
    (word: ConfirmedWord) => {
      setLatestWord({ text: word.word, confidence: word.confidence, at: Date.now() });
      conversation.appendWord(word);
      window.clearTimeout(pauseTimer.current);
      pauseTimer.current = window.setTimeout(
        () => finishRef.current(),
        SENTENCE_PAUSE_MS,
      );
    },
    [conversation],
  );

  useEffect(() => () => window.clearTimeout(pauseTimer.current), []);

  /* ------------------------------------------------------- socket + cam -- */
  const { socket, live, buffered, error, noModel, send, setThreshold } =
    useVoxSocket(handleWord);
  const tracking = useHandTracking(send);

  const handleThreshold = useCallback(
    (value: number) => {
      setThresholdState(value);
      setThreshold(value); // forwarded to the backend, re-sent on reconnect
    },
    [setThreshold],
  );

  /* --------------------------------------------- speech -> ISL pipeline -- */
  const handleHeard = useCallback(
    (text: string) => {
      conversation.addUtterance(text, "other");
      isl.enqueuePhrase(text);
    },
    [conversation, isl],
  );
  const recognition = useSpeechRecognition(handleHeard);

  return (
    <div className="session">
      <header className="topbar">
        <div className="topbar__brand">
          <Link to="/" className="topbar__home" aria-label="Vox home">
            <span className="topbar__mark" aria-hidden />
            <h1 className="topbar__title">Vox</h1>
          </Link>
          <span className="topbar__sub">Indian Sign Language interpreter</span>
        </div>

        <div className="topbar__status">
          {error && <span className="badge badge--error">{error}</span>}
          {noModel && (
            <span
              className="badge badge--error"
              title="The backend is running but ml/models/vox_lstm.keras is missing. Run: python ml/extract.py --videos-dir <videos> && python ml/preprocess.py && python ml/train.py"
            >
              recognition off — no model
            </span>
          )}
          <button
            type="button"
            className={`button voice-toggle ${voiceOn && speech.supported ? "voice-toggle--on" : ""}`}
            onClick={() => setVoiceOn(!voiceOn)}
            disabled={!speech.supported}
            title={
              speech.supported
                ? "Choose whether finished sentences are read aloud or stay as text"
                : "Speech output is not available in this browser"
            }
            aria-pressed={voiceOn}
          >
            {speech.supported ? (voiceOn ? "🔊 Voice on" : "🔇 Text only") : "🔇 No voice"}
          </button>
          <span className={`badge badge--${socket === "open" ? "live" : "error"}`}>
            {socket === "open" ? "Backend connected" : `Backend ${socket}…`}
          </span>
          <button
            type="button"
            className="button"
            onClick={() => setSettingsOpen(true)}
          >
            Settings
          </button>
        </div>
      </header>

      <main className="workspace">
        <CameraPanel
          tracking={tracking}
          live={live}
          buffered={buffered}
          latestWord={latestWord}
          threshold={threshold}
        />
        <TranscriptPanel
          conversation={conversation}
          onSpeakNow={finishSentence}
          speaking={speech.speaking}
          voiceOn={voiceOn && speech.supported}
        />
        <SignVideoPanel
          isl={isl}
          recognition={recognition}
          onPhrase={handleHeard}
          ttsSpeaking={speech.speaking}
          lastWordAt={latestWord?.at ?? null}
        />
      </main>

      <SettingsDrawer
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        threshold={threshold}
        onThreshold={handleThreshold}
        autoSpeak={voiceOn}
        onAutoSpeak={setVoiceOn}
        ttsSupported={speech.supported}
        tracking={tracking}
        onClearConversation={conversation.clear}
      />
    </div>
  );
}

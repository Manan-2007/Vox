/**
 * The session view: camera left, conversation centre, speech -> ISL right.
 *
 * This is the only place the data sources join:
 *   hand tracking -> socket -> confirmed words -> current sentence
 *   sentence completion (pause timeout or button) -> commit + TTS   (P8)
 *   speech recognition -> transcript ("heard") + ISL clip queue     (P10)
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

export function SessionPage() {
  const conversation = useConversation();
  const speech = useSpeech();
  const isl = useIslQueue();

  const [latestWord, setLatestWord] = useState<{ text: string; confidence: number } | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [threshold, setThresholdState] = useState(DEFAULT_THRESHOLD);
  const [autoSpeak, setAutoSpeak] = useState(true);

  // Refs so timers see current state without re-arming on every render.
  const currentRef = useRef(conversation.current);
  currentRef.current = conversation.current;
  const autoSpeakRef = useRef(autoSpeak);
  autoSpeakRef.current = autoSpeak;
  const pauseTimer = useRef<number | undefined>(undefined);

  /* ------------------------------------------------- sentence completion -- */
  const finishSentence = useCallback(
    (viaButton: boolean) => {
      window.clearTimeout(pauseTimer.current);
      const turn = currentRef.current;
      if (turn.words.length === 0) return;
      const text = turnText(turn);
      const willSpeak = speech.supported && (viaButton || autoSpeakRef.current);
      conversation.newTurn({ spoken: willSpeak });
      if (willSpeak) speech.speak(text);
    },
    [conversation, speech],
  );

  const finishRef = useRef(finishSentence);
  finishRef.current = finishSentence;

  const handleWord = useCallback(
    (word: ConfirmedWord) => {
      setLatestWord({ text: word.word, confidence: word.confidence });
      conversation.appendWord(word);
      window.clearTimeout(pauseTimer.current);
      pauseTimer.current = window.setTimeout(
        () => finishRef.current(false),
        SENTENCE_PAUSE_MS,
      );
    },
    [conversation],
  );

  useEffect(() => () => window.clearTimeout(pauseTimer.current), []);

  /* ------------------------------------------------------- socket + cam -- */
  const { socket, live, buffered, error, send, setThreshold } =
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
          {!speech.supported && (
            <span className="badge badge--idle" title="SpeechSynthesis missing">
              no TTS
            </span>
          )}
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
          onSpeakNow={() => finishSentence(true)}
          speaking={speech.speaking}
        />
        <SignVideoPanel isl={isl} recognition={recognition} onPhrase={handleHeard} />
      </main>

      <SettingsDrawer
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        threshold={threshold}
        onThreshold={handleThreshold}
        autoSpeak={autoSpeak}
        onAutoSpeak={setAutoSpeak}
        ttsSupported={speech.supported}
        tracking={tracking}
        onClearConversation={conversation.clear}
      />
    </div>
  );
}

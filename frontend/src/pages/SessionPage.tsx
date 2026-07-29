/**
 * The session view: camera on the left, conversation in the middle, ISL
 * playback on the right.
 *
 * This is the only place the three data sources are joined — hand tracking
 * feeds the socket, and the socket's confirmed words feed the conversation.
 */
import { useCallback, useState } from "react";
import { CameraPanel } from "../components/CameraPanel";
import { SignVideoPanel } from "../components/SignVideoPanel";
import { TranscriptPanel } from "../components/TranscriptPanel";
import { useConversation } from "../hooks/useConversation";
import { useHandTracking } from "../hooks/useHandTracking";
import { useVoxSocket, type ConfirmedWord } from "../hooks/useVoxSocket";

export function SessionPage() {
  const conversation = useConversation();
  const [latestWord, setLatestWord] = useState<{ text: string; confidence: number } | null>(
    null,
  );

  const handleWord = useCallback(
    (word: ConfirmedWord) => {
      setLatestWord({ text: word.word, confidence: word.confidence });
      conversation.appendWord(word);
    },
    [conversation],
  );

  const { socket, live, buffered, error, send } = useVoxSocket(handleWord);
  const tracking = useHandTracking(send);

  return (
    <div className="session">
      <header className="topbar">
        <div className="topbar__brand">
          <span className="topbar__mark" aria-hidden />
          <h1 className="topbar__title">Vox</h1>
          <span className="topbar__sub">Indian Sign Language interpreter</span>
        </div>

        <div className="topbar__status">
          {error && <span className="badge badge--error">{error}</span>}
          <span className={`badge badge--${socket === "open" ? "live" : "idle"}`}>
            Backend {socket}
          </span>
        </div>
      </header>

      <main className="workspace">
        <CameraPanel
          tracking={tracking}
          live={live}
          buffered={buffered}
          latestWord={latestWord}
        />
        <TranscriptPanel conversation={conversation} />
        <SignVideoPanel cue={latestWord?.text ?? null} />
      </main>
    </div>
  );
}

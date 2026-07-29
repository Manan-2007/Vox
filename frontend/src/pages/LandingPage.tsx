/**
 * Landing page: what Vox set out to do, what it verifiably does today, how to
 * operate it, and a starter guide of signs and phrases to try.
 *
 * The claims here are kept honest on purpose — "what works today" lists only
 * behaviour that is implemented and tested in this repo.
 */
import { Link } from "react-router-dom";
import { Orb } from "../components/Orb";

const AIMS = [
  "A full two-way interpreter: a Deaf or hard-of-hearing signer and a hearing speaker holding a natural conversation.",
  "Recognition that generalizes across signers, lighting, and regional ISL variation.",
  "Continuous signing — whole sentences signed fluidly, not word by word.",
  "Facial expression and body pose as part of meaning, the way real ISL uses them.",
];

const DONE = [
  "Live hand tracking in the browser at 15 FPS — MediaPipe in a Web Worker, so the page never stutters.",
  "Isolated-sign recognition over a 2-second window (LSTM), gated on confidence and stability before a word is accepted.",
  "Sentences build word by word and are spoken aloud — with voice output fully optional.",
  "Spoken or typed replies are glossed into ISL tokens and played back as sign clips, with a queue.",
  "Privacy by construction: video never leaves the machine; only 126 landmark numbers per frame cross a local socket.",
  "Graceful degradation: backend reconnects automatically; every capability (mic, voice, clips) has a fallback.",
];

const STEPS = [
  {
    title: "Allow the camera",
    body: "Open a session and grant camera access. The skeleton overlay confirms your hands are being tracked.",
  },
  {
    title: "Sign, and hold",
    body: "Face the camera and hold each sign steady for about two seconds. The confidence bar shows how sure the model is; a word is accepted only past the marker.",
  },
  {
    title: "Pause to finish",
    body: "Words collect into a sentence. Pause signing for a few seconds — or press “Speak sentence” — and the sentence is committed, and voiced if voice is on.",
  },
  {
    title: "Reply in speech",
    body: "The other person taps the mic and speaks, or types a phrase. Vox glosses it into sign tokens and plays the matching ISL clips.",
  },
];

/**
 * Starter vocabulary guide. The descriptions are indicative common forms —
 * ISL varies regionally, and the model recognizes signs AS RECORDED during
 * data collection, so the recorded form is always the ground truth.
 */
const SIGNS = [
  { word: "hello", glyph: "👋", how: "Open hand raised beside the head, palm out — a small wave." },
  { word: "thanks", glyph: "🙏", how: "Flat hand at the chin moving forward, or palms pressed together." },
  { word: "yes", glyph: "👍", how: "A closed fist nodding at the wrist, like a head nodding." },
  { word: "no", glyph: "✋", how: "Open hand swinging side to side, palm facing out." },
  { word: "please", glyph: "🤲", how: "Flat hand circling over the chest." },
  { word: "help", glyph: "🤝", how: "One fist resting on the opposite flat palm, both lifted together." },
  { word: "eat", glyph: "🤌", how: "Fingertips bunched, tapping toward the mouth." },
  { word: "bye", glyph: "🖐️", how: "Raised open hand, fingers folding down and up — a goodbye wave." },
];

const PHRASES = [
  "Hello, thanks for your help",
  "Please help me",
  "Yes, eat now",
  "No thanks, bye",
];

export function LandingPage() {
  return (
    <div className="landing">
      <header className="landing__nav">
        <div className="topbar__brand">
          <span className="topbar__mark" aria-hidden />
          <span className="topbar__title">Vox</span>
        </div>
        <Link to="/session" className="button button--primary">
          Open the interpreter
        </Link>
      </header>

      <main>
        <section className="hero">
          <div className="hero__copy">
            <p className="hero__kicker">Real-time Indian Sign Language</p>
            <h1 className="hero__title">
              A two-way conversation,
              <br />
              <span className="hero__accent">signed and spoken.</span>
            </h1>
            <p className="hero__body">
              Vox watches your hands, recognizes ISL signs as you make them, and
              builds them into sentences the other person can read — or hear.
              Their spoken reply comes back as sign video. Voice is always a
              choice, never a requirement.
            </p>
            <div className="hero__actions">
              <Link to="/session" className="button button--primary button--lg">
                Start a session
              </Link>
              <a className="button button--lg" href="#guide">
                How to use it ↓
              </a>
            </div>
          </div>
          <div className="hero__orb">
            <Orb state="idle" size={190} />
            <span className="orb-caption">the Vox orb — it reacts as you talk</span>
          </div>
        </section>

        <section className="section">
          <h2 className="section__heading">What we set out to build — and where it stands</h2>
          <div className="claims">
            <article className="claims__card claims__card--done">
              <h3 className="claims__title">Working today</h3>
              <ul className="claims__list">
                {DONE.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </article>
            <article className="claims__card claims__card--aim">
              <h3 className="claims__title">The larger goal (not all of it is here yet)</h3>
              <ul className="claims__list">
                {AIMS.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
              <p className="disclaimer" style={{ marginTop: 14 }}>
                Today Vox recognizes a small, fixed vocabulary of isolated
                signs, trained on data collected with its own tool. Accuracy
                depends on that data — it is a working proof of the pipeline,
                not a general ISL translator.
              </p>
            </article>
          </div>
        </section>

        <section className="section" id="guide">
          <h2 className="section__heading">How to operate it</h2>
          <div className="steps__grid">
            {STEPS.map((step, index) => (
              <article key={step.title} className="step">
                <span className="step__n">{index + 1}</span>
                <h3 className="step__title">{step.title}</h3>
                <p className="step__body">{step.body}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="section">
          <h2 className="section__heading">Starter signs to try</h2>
          <div className="signs">
            {SIGNS.map((sign) => (
              <article key={sign.word} className="sign-card">
                <span className="sign-card__glyph" aria-hidden>{sign.glyph}</span>
                <span className="sign-card__word">{sign.word}</span>
                <span className="sign-card__how">{sign.how}</span>
              </article>
            ))}
          </div>
          <p className="disclaimer">
            These descriptions are indicative common forms. ISL varies by
            region, and Vox recognizes each sign <strong>as it was recorded
            during training</strong> — if your dataset signs “hello”
            differently, sign it your way. The recorded form is the ground
            truth.
          </p>
        </section>

        <section className="section">
          <h2 className="section__heading">Phrases to speak at it</h2>
          <div className="phrases">
            {PHRASES.map((phrase) => (
              <span key={phrase} className="phrase">
                “<strong>{phrase}</strong>”
              </span>
            ))}
          </div>
        </section>

        <section className="section">
          <h2 className="section__heading">Under the hood</h2>
          <div className="pipeline">
            <span className="pipeline__node">Webcam</span>
            <span className="pipeline__arrow">→</span>
            <span className="pipeline__node">MediaPipe · Web Worker</span>
            <span className="pipeline__arrow">→</span>
            <span className="pipeline__node">126 landmarks/frame</span>
            <span className="pipeline__arrow">→</span>
            <span className="pipeline__node">WebSocket</span>
            <span className="pipeline__arrow">→</span>
            <span className="pipeline__node">LSTM + gating</span>
            <span className="pipeline__arrow">→</span>
            <span className="pipeline__node">Sentence</span>
            <span className="pipeline__arrow">→</span>
            <span className="pipeline__node">Voice (optional)</span>
            <span style={{ flexBasis: "100%" }} />
            No video ever leaves your machine — the camera feed is processed in
            the browser and only landmark numbers reach the local backend.
          </div>
        </section>
      </main>

      <footer className="landing__foot">
        <span>Vox — real-time ISL interpreter</span>
        <span>
          Built for Deaf, hard-of-hearing, and hearing users alike — read it,
          or hear it. Your choice.
        </span>
      </footer>
    </div>
  );
}

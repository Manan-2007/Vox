/** Landing page: what Vox is, how it works, and the way into the session. */
import { Link } from "react-router-dom";

const FEATURES = [
  {
    title: "Sign → speech",
    body: "Hands are tracked in the browser at 15 FPS; an LSTM recognizes each sign, sentences build word by word and are spoken aloud.",
  },
  {
    title: "Speech → sign",
    body: "Spoken English is transcribed, glossed into ISL tokens, and played back as a sequence of sign clips.",
  },
  {
    title: "Private by design",
    body: "No video ever leaves your machine. Only 126 hand-landmark numbers per frame cross the wire — to a backend you run yourself.",
  },
] as const;

const STEPS = [
  { n: "01", title: "Camera", body: "MediaPipe finds 21 landmarks per hand, in a Web Worker so the page never stutters." },
  { n: "02", title: "Recognize", body: "A compact LSTM watches a rolling 2-second window and gates words on confidence and stability." },
  { n: "03", title: "Converse", body: "Words become sentences, sentences are voiced, and replies come back as ISL video." },
] as const;

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
          <p className="hero__kicker">Real-time Indian Sign Language</p>
          <h1 className="hero__title">
            A two-way conversation,
            <br />
            <span className="hero__accent">signed and spoken.</span>
          </h1>
          <p className="hero__body">
            Vox watches your hands, recognizes ISL signs as you make them, and
            speaks the sentence for the person across from you — then turns
            their spoken reply back into sign.
          </p>
          <div className="hero__actions">
            <Link to="/session" className="button button--primary button--lg">
              Start a session
            </Link>
            <a
              className="button button--lg"
              href="https://github.com"
              onClick={(e) => e.preventDefault()}
              title="Repository is local for now"
            >
              How it's built ↓
            </a>
          </div>
        </section>

        <section className="features">
          {FEATURES.map((f) => (
            <article key={f.title} className="feature">
              <h2 className="feature__title">{f.title}</h2>
              <p className="feature__body">{f.body}</p>
            </article>
          ))}
        </section>

        <section className="steps" id="how">
          <h2 className="steps__heading">How it works</h2>
          <div className="steps__grid">
            {STEPS.map((s) => (
              <article key={s.n} className="step">
                <span className="step__n">{s.n}</span>
                <h3 className="step__title">{s.title}</h3>
                <p className="step__body">{s.body}</p>
              </article>
            ))}
          </div>
        </section>
      </main>

      <footer className="landing__foot">
        <span>Vox — Milestone 1</span>
        <span>
          Recognition quality depends on the trained vocabulary; see the README
          for the current sign list.
        </span>
      </footer>
    </div>
  );
}

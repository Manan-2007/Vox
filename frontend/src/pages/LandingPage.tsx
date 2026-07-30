/**
 * The landing page.
 *
 * It leads with the avatar actually signing a sentence, because that is the
 * product, and because a still image of a sign-language tool tells you nothing.
 *
 * The vocabulary count is read from the shipped manifest rather than typed in,
 * so the page cannot claim a vocabulary the build does not have. The limitations
 * section is not modesty — a sign-language tool that overclaims gets relied on
 * in a hospital and fails there.
 */
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { SignAvatar } from "../components/SignAvatar";
import {
  loadManifest,
  loadSign,
  type ManifestEntry,
  type QueueItem,
} from "../avatar/signMotion";
import { useRecognitionQuality } from "../isl/useRecognitionQuality";

/** The hero loop: a real sentence, not a word list. */
const HERO_PHRASE: [string, string][] = [
  ["i", "I"],
  ["you", "you"],
  ["help", "help"],
  ["can", "can"],
];

/** Shown if present in the library — the words the product was built around. */
const HIGHLIGHTS = [
  "help", "water", "eat", "doctor", "hospital", "pain", "police", "medicine",
  "money", "what", "where", "howmuch", "please", "sorry", "yes", "no",
  "mother", "home", "work", "understand", "tomorrow", "emergency",
];

export function LandingPage() {
  const [signs, setSigns] = useState<Record<string, ManifestEntry>>({});
  const [queue, setQueue] = useState<QueueItem[]>([]);
  // Numbers come from the measurement file, never from the copy — a page that
  // claims an accuracy the build cannot reach is exactly the failure this
  // project is trying not to be.
  const recognition = useRecognitionQuality();

  useEffect(() => {
    let disposed = false;
    void loadManifest().then(async (manifest) => {
      if (disposed) return;
      setSigns(manifest.signs ?? {});
      const items = await Promise.all(
        HERO_PHRASE.map(async ([gloss, label]) => {
          const motion = await loadSign(gloss);
          return { gloss, label, motion, missing: motion === null };
        }),
      );
      if (!disposed) setQueue(items.filter((item) => !item.missing));
    });
    return () => {
      disposed = true;
    };
  }, []);

  const available = useMemo(() => Object.keys(signs), [signs]);
  const sample = useMemo(
    () =>
      HIGHLIGHTS.filter((gloss) => gloss in signs).map(
        (gloss) => signs[gloss].english,
      ),
    [signs],
  );

  return (
    <div className="landing">
      <nav className="landing__nav">
        <span className="brand">
          <span className="brand__mark" aria-hidden />
          <span className="brand__name">Vox</span>
        </span>
        <Link className="btn btn--primary" to="/session">
          Open the interpreter
        </Link>
      </nav>

      <header className="hero">
        <div>
          <p className="hero__kicker">
            <span className="chip__dot" style={{ background: "var(--good)" }} />
            Runs on your machine — no video ever leaves it
          </p>
          <h1 className="hero__title">
            Indian Sign Language,
            <br />
            <span className="hero__accent">translated both ways.</span>
          </h1>
          <p className="hero__body">
            Sign to the camera and Vox assembles the sentence and speaks it. Say
            something back and a 3D signer performs it in ISL — reordered into
            ISL grammar, not English word-for-word on the hands. Slow it down,
            loop it, turn it around and see the sign from any angle.
          </p>
          <div className="hero__actions">
            <Link className="btn btn--primary btn--lg" to="/session">
              Start a conversation
            </Link>
            <a
              className="btn btn--lg"
              href="https://islrtc.nic.in/"
              target="_blank"
              rel="noreferrer noopener"
            >
              About ISLRTC
            </a>
          </div>
        </div>

        <div className="hero__stage">
          <SignAvatar queue={queue} loop mirror />
          {queue.length === 0 && (
            <div className="stage__empty">
              <p className="stage__empty-title">Loading the signer…</p>
            </div>
          )}
        </div>
      </header>

      <section className="section">
        <h2 className="section__title">What it does</h2>
        <p className="section__lede">
          Two directions, one conversation. The signs come from the official
          Indian Sign Language dictionary published by ISLRTC, an autonomous body
          under the Government of India.
        </p>
        <div className="grid">
          <article className="tile">
            <p className="tile__metric">{available.length || "—"}</p>
            <h3 className="tile__title">signs the avatar can perform</h3>
            <p className="tile__body">
              Verbs, pronouns, question words, health and emergency vocabulary —
              chosen so you can ask for a doctor, not only say hello.
            </p>
          </article>
          <article className="tile">
            <h3 className="tile__title">Grammar, not substitution</h3>
            <p className="tile__body">
              &ldquo;What is your name?&rdquo; is signed{" "}
              <strong>YOU NAME WHAT</strong>. Question words go last in ISL, the
              copula is dropped, and time comes first and carries the tense.
              Signed sentences are rebuilt into English by the same rules,
              backwards.
            </p>
          </article>
          <article className="tile">
            <h3 className="tile__title">A signer, not a clip</h3>
            <p className="tile__body">
              Each sign is landmark motion driving one 3D figure built from
              metric hand geometry. That is why it can be slowed, looped and
              orbited — and why nobody&rsquo;s likeness ships with the app.
            </p>
          </article>
        </div>
      </section>

      {recognition.loaded && (
        <section className="section">
          <h2 className="section__title">How well it recognises</h2>
          <p className="section__lede">
            Measured on recordings the model never trained on, split by source
            video so no recording appears on both sides. These are the real
            numbers, not a best case.
          </p>
          <div className="grid">
            <article className="tile">
              <p className="tile__metric">
                {Math.round(recognition.gatedPrecision * 100)}%
              </p>
              <h3 className="tile__title">correct when it speaks</h3>
              <p className="tile__body">
                Vox stays silent unless it clears a confidence bar. That is the
                number that matters in use, because a wrong word is worse than
                no word — it makes a hearing person act on something the signer
                never said.
              </p>
            </article>
            <article className="tile">
              <p className="tile__metric">
                {Math.round(recognition.top1 * 100)}%
              </p>
              <h3 className="tile__title">
                top-1 across all {recognition.classes} words
              </h3>
              <p className="tile__body">
                Ungated, every word treated equally, against a{" "}
                {(100 / Math.max(1, recognition.classes)).toFixed(1)}% chance
                baseline. Low, and honestly so: most words have a single
                recording to learn from.
              </p>
            </article>
            <article className="tile">
              <p className="tile__metric">{recognition.reliable.size}</p>
              <h3 className="tile__title">words verified at 80%+</h3>
              <p className="tile__body">
                Words with enough recordings to hold one back and test properly.
                The app marks every other word in the transcript so you can see
                which predictions carry weight.
              </p>
            </article>
          </div>
        </section>
      )}

      {sample.length > 0 && (
        <section className="section">
          <h2 className="section__title">Some of the vocabulary</h2>
          <p className="section__lede">
            The words the product was built around. The full list is in the app.
          </p>
          <div className="wordcloud">
            {sample.map((word) => (
              <span className="wordcloud__item" key={word}>
                {word}
              </span>
            ))}
          </div>
        </section>
      )}

      <section className="section">
        <h2 className="section__title">What it cannot do yet</h2>
        <p className="section__lede">
          Stated plainly. Each of these is a real limit of this build, not a
          rough edge that polish would fix.
        </p>
        <div className="grid">
          <article className="tile tile--limit">
            <h3 className="tile__title">No facial grammar</h3>
            <p className="tile__body">
              ISL marks questions, negation and intensity on the face and body —
              eyebrow raise, head shake, mouth morphemes. The avatar has none of
              it and the recogniser is not trained on it, so output is
              grammatical but flat.
            </p>
          </article>
          <article className="tile tile--limit">
            <h3 className="tile__title">One sign at a time</h3>
            <p className="tile__body">
              Recognition reads a two-second window and names one sign. Fluent
              signing runs signs together with no gaps between them, and
              separating those is an open research problem, not a setting.
            </p>
          </article>
          <article className="tile tile--limit">
            <h3 className="tile__title">Most words are unverified</h3>
            <p className="tile__body">
              The dictionary gives one signer per word. Accuracy can only be
              measured for words with a spare recording to hold back; the rest
              are trainable but unmeasured, and the transcript underlines them.
              Recording yourself is the fastest way to fix this for the words you
              actually use.
            </p>
          </article>
        </div>
      </section>

      <footer className="landing__foot">
        <span>
          Sign data from the{" "}
          <a href="https://islrtc.nic.in/" target="_blank" rel="noreferrer noopener">
            ISLRTC
          </a>{" "}
          ISL dictionary and the{" "}
          <a
            href="https://zenodo.org/records/4010759"
            target="_blank"
            rel="noreferrer noopener"
          >
            INCLUDE
          </a>{" "}
          dataset (CC-BY-4.0).
        </span>
        <span>Landmarks only — no video is stored or transmitted.</span>
      </footer>
    </div>
  );
}

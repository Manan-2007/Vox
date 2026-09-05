/**
 * What the recogniser can actually be trusted with — stated up front.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS IS ON THE LANDING PAGE AND NOT BURIED IN SETTINGS
 * ---------------------------------------------------------------------------
 * The model has 242 classes and 38% top-1 accuracy, because it was trained on
 * 1,065 samples — a median of two per word. Only 37 words reach 80% on
 * recordings they were never trained on, and those are the only ones the backend
 * will emit.
 *
 * A user cannot discover that by using the product. They sign something outside
 * the verified set, get "gesture not recognised", and conclude the app is
 * broken — when in fact it is refusing to guess, which is the correct behaviour
 * and the safe one. The difference between "this app does not work" and "this
 * app knows 37 words and is honest about it" is entirely a matter of saying so.
 *
 * For an accessibility tool used in medical conversations, overstating the
 * vocabulary is the actual hazard. A confidently wrong word is worse than no
 * word, so the scope is published rather than implied.
 */

import { useMemo } from "react";
import type { RecognitionQuality } from "../isl/useRecognitionQuality";

interface Props {
  quality: RecognitionQuality;
  /** Glosses present in the motion library — what Vox can SIGN back. */
  signable?: ReadonlySet<string>;
}

/** Turn a gloss into something readable: "goodmorning" → "good morning". */
const SURFACE: Record<string, string> = {
  goodmorning: "good morning",
  goodafternoon: "good afternoon",
  goodevening: "good evening",
  goodnight: "good night",
  goodbye: "goodbye",
  howareyou: "how are you",
  howmany: "how many",
  howmuch: "how much",
  thankyou: "thank you",
  dontknow: "don't know",
  hearingaid: "hearing aid",
  signlanguage: "sign language",
  wakeup: "wake up",
};

function label(gloss: string): string {
  return SURFACE[gloss] ?? gloss;
}

export function Vocabulary({ quality, signable }: Props) {
  const verified = useMemo(
    () => [...quality.reliable].map(label).sort((a, b) => a.localeCompare(b)),
    [quality.reliable],
  );

  if (!quality.loaded) {
    return (
      <section className="vocab">
        <h2 className="vocab__title">Recognition vocabulary</h2>
        <p className="vocab__lead">
          No measurements found. Train and evaluate a model to see what it can
          reliably recognise.
        </p>
      </section>
    );
  }

  return (
    <section className="vocab" id="vocabulary">
      <h2 className="vocab__title">What Vox can recognise</h2>

      <p className="vocab__lead">
        Vox reads <strong>{verified.length} signs</strong> it has been measured
        on. Sign one of these and it will be transcribed; sign anything else and
        it will say <em>“gesture not recognised”</em> rather than guess.
      </p>

      {/* The numbers, plainly. A product that quietly narrows its scope is
          harder to trust than one that publishes it. */}
      <dl className="vocab__stats">
        <div>
          <dt>Recognised</dt>
          <dd>{verified.length} signs</dd>
        </div>
        <div>
          <dt>Right when it speaks</dt>
          <dd>{Math.round(quality.gatedPrecision * 100)}%</dd>
        </div>
        <div>
          <dt>Can sign back</dt>
          <dd>{signable ? `${signable.size} signs` : "—"}</dd>
        </div>
      </dl>

      <ul className="vocab__list">
        {verified.map((word) => (
          <li key={word} className="vocab__word">
            {word}
          </li>
        ))}
      </ul>

      <p className="vocab__note">
        These are the words that scored 80% or better on recordings the model
        never saw during training. The model knows{" "}
        {quality.classes} classes in total, but the rest have too few examples to
        be trusted — a median of two recordings each — so Vox does not offer
        them. Adding recordings widens this list automatically; nothing here is
        hard-coded.
      </p>

      <p className="vocab__note">
        Signing back is a different and much larger set: Vox can{" "}
        <strong>produce</strong> {signable ? signable.size : "many"} signs from
        the reference library, because playing a recorded sign needs no
        classifier. Typing or speaking a phrase works across that whole
        vocabulary.
      </p>
    </section>
  );
}

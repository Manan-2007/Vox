/**
 * Non-manual markers — the half of ISL grammar that is not on the hands.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS IS GRAMMAR AND NOT DECORATION
 * ---------------------------------------------------------------------------
 * In Indian Sign Language a raised brow is not an expression, it is a
 * morpheme. The same sequence of handshapes means different things depending on
 * the face carried over it:
 *
 *     YOU DOCTOR              neutral face   -> "You are a doctor."
 *     YOU DOCTOR              brows raised   -> "Are you a doctor?"
 *     YOU DOCTOR              brows furrowed -> "Which doctor are you?"
 *     HOSPITAL YOU GO   + head shake over GO -> "You are not going to hospital."
 *
 * A signer reading hands alone gets the words and loses the sentence type. That
 * is why the previous build's output was described as "grammatically correct but
 * flat": the word order was right and the clause type was invisible.
 *
 * ---------------------------------------------------------------------------
 * WHY IT IS SYNTHESISED RATHER THAN RECORDED
 * ---------------------------------------------------------------------------
 * The obvious alternative is to extract faces from the dictionary clips. It does
 * not work, for a reason that has nothing to do with tooling: those clips are
 * *citation forms*. Each one is a single word signed in isolation with a
 * deliberately neutral face, because the dictionary is teaching the handshape.
 * The non-manual marker belongs to the sentence, not the word, so it does not
 * exist in any recording of a word — it has to come from the syntax, which is
 * exactly where ./isl/grammar.ts already knows it.
 *
 * Scope, stated plainly: this covers clause-type marking (polar questions,
 * content questions, negation, topic) and the mouth being open on a greeting.
 * Mouth morphemes that distinguish minimal pairs, eye-gaze reference to points
 * in signing space, and role shift are not modelled.
 */

/**
 * One face pose. Every channel is normalized so the rig can scale it, and every
 * one is neutral at zero.
 */
export interface NonManual {
  /** -1 fully furrowed, +1 fully raised. */
  brow: number;
  /** 0-1, narrowing the eyes; carries doubt and, with a furrow, WH questions. */
  squint: number;
  /** 0-1, jaw opening. */
  mouthOpen: number;
  /** -1 pursed, +1 wide. A small positive reads as a smile. */
  mouthWide: number;
  /** Radians. Positive brings the chin toward the addressee. */
  headForward: number;
  /** Radians of roll, the questioning tilt. */
  headTilt: number;
  /** 0-1 amplitude of a repeated side-to-side shake — the negation marker. */
  headShake: number;
  /** 0-1 amplitude of a nod, which affirms and marks a topic boundary. */
  headNod: number;
}

export const NEUTRAL_FACE: NonManual = {
  brow: 0,
  squint: 0,
  mouthOpen: 0,
  mouthWide: 0,
  headForward: 0,
  headTilt: 0,
  headShake: 0,
  headNod: 0,
};

const CHANNELS = Object.keys(NEUTRAL_FACE) as (keyof NonManual)[];

export function face(partial: Partial<NonManual>): NonManual {
  return { ...NEUTRAL_FACE, ...partial };
}

/** Linear blend, used across the transition between two signs. */
export function blendFace(a: NonManual, b: NonManual, t: number): NonManual {
  const out = { ...NEUTRAL_FACE };
  for (const key of CHANNELS) out[key] = a[key] + (b[key] - a[key]) * t;
  return out;
}

export function scaleFace(a: NonManual, k: number): NonManual {
  const out = { ...NEUTRAL_FACE };
  for (const key of CHANNELS) out[key] = a[key] * k;
  return out;
}

/* ------------------------------------------------------- the marker set -- */

/**
 * Clause-type markers, as described in the ISL literature and as they are
 * actually produced. The magnitudes are deliberately restrained: a marker has to
 * be unmistakable without turning the figure into a cartoon, and an over-acted
 * brow raise reads as surprise rather than as a question particle.
 */
export const MARKERS = {
  /** Polar (yes/no) question: brows up, head forward, held over the clause. */
  polarQuestion: face({ brow: 0.85, headForward: 0.16, mouthWide: 0.1 }),
  /** Content (WH) question: brows down and drawn together, head tilted. */
  contentQuestion: face({ brow: -0.8, squint: 0.35, headForward: 0.12, headTilt: 0.1 }),
  /** Negation: the head shake is the marker; the brow only supports it. */
  negation: face({ brow: -0.45, headShake: 1, squint: 0.15 }),
  /** Topic marker: brows up with a small backward tilt, then released. */
  topic: face({ brow: 0.55, headTilt: -0.07 }),
  /** Affirmation, on YES and on a statement's final sign. */
  affirm: face({ headNod: 0.7, mouthWide: 0.12 }),
  /** Greetings and thanks read as cold without it. */
  warm: face({ mouthWide: 0.45, brow: 0.2 }),
  /** Pain, sickness, sorry: the face a signer actually makes. */
  discomfort: face({ brow: -0.5, squint: 0.5, mouthWide: -0.35 }),
  /** Emphasis on a large or intense sign. */
  emphasis: face({ brow: 0.25, mouthOpen: 0.3 }),
} as const;

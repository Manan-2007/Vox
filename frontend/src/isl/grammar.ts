/**
 * Translation between English and Indian Sign Language gloss, in both
 * directions.
 *
 * The previous version of this file did word-for-word substitution and dropped
 * a stopword list. That is not translation, and it produced output no signer
 * would accept: "hello thankyou pleased" in one direction, and "you name what"
 * read back to a hearing person as literally "you name what" in the other.
 *
 * ---------------------------------------------------------------------------
 * WHAT ISL GRAMMAR ACTUALLY DOES
 * ---------------------------------------------------------------------------
 * ISL is not English on the hands. The differences this file implements are the
 * ones that change whether a sentence is understood at all:
 *
 *   * Word order is subject-object-VERB. "I want water" is signed I WATER WANT.
 *   * There is no copula. "She is a doctor" is SHE DOCTOR — no sign for "is".
 *   * There are no articles. "the hospital" is HOSPITAL.
 *   * Time comes first and carries the tense. There is no signed past or future
 *     inflection: "I went home" is YESTERDAY I HOME GO, and "I will go home" is
 *     TOMORROW I HOME GO — or LATER, or the time you actually mean.
 *   * Question words come last. "What is your name?" is YOU NAME WHAT.
 *     "Where is the hospital?" is HOSPITAL WHERE. This is the single most
 *     recognisable feature of ISL syntax, and getting it backwards is what makes
 *     machine-glossed ISL read as broken.
 *   * Negation follows the verb it negates: "I don't understand" is
 *     I UNDERSTAND NOT.
 *
 * ---------------------------------------------------------------------------
 * NON-MANUAL GRAMMAR
 * ---------------------------------------------------------------------------
 * Clause type in ISL is carried on the FACE, not in the word order alone: a
 * raised brow makes a statement a yes/no question, a furrowed brow and a tilt
 * mark a content question, and negation is scoped by a head shake over the signs
 * it applies to. Producing the right hands with a blank face is producing half a
 * sentence, so this file also emits a marker per token — see ../avatar/nonManual
 * for what each one is and why it is synthesised here rather than recorded.
 *
 * ---------------------------------------------------------------------------
 * WHAT THIS STILL DOES NOT DO
 * ---------------------------------------------------------------------------
 * Mouth morphemes that distinguish minimal pairs, and the use of signing space
 * to set up a referent and point back at it, are absent in both directions.
 * Classifier predicates and verb agreement (directing GIVE from signer to
 * addressee rather than signing a neutral GIVE) are also out of scope: they need
 * a rig that can aim a sign, not a library of fixed recordings.
 */

import {
  face,
  MARKERS,
  NEUTRAL_FACE,
  type NonManual,
} from "../avatar/nonManual";
import {
  ENGLISH_BY_GLOSS,
  POS_BY_GLOSS,
  SURFACE_FORMS,
  type PartOfSpeech,
} from "./vocabulary.generated";

/* ------------------------------------------------------ English -> gloss -- */

/** Contractions expanded before anything else, so "don't" can become NOT. */
const CONTRACTIONS: Record<string, string> = {
  "i'm": "i am",
  "i've": "i have",
  "i'll": "i will",
  "i'd": "i would",
  "you're": "you are",
  "you've": "you have",
  "you'll": "you will",
  "he's": "he is",
  "she's": "she is",
  "it's": "it is",
  "we're": "we are",
  "we'll": "we will",
  "they're": "they are",
  "that's": "that is",
  "what's": "what is",
  "where's": "where is",
  "who's": "who is",
  "how's": "how is",
  "there's": "there is",
  "isn't": "is not",
  "aren't": "are not",
  "wasn't": "was not",
  "weren't": "were not",
  "don't": "do not",
  "doesn't": "does not",
  "didn't": "did not",
  "can't": "can not",
  "cannot": "can not",
  "couldn't": "could not",
  "won't": "will not",
  "wouldn't": "would not",
  "shouldn't": "should not",
  "haven't": "have not",
  "hasn't": "has not",
  "hadn't": "had not",
  "let's": "let us",
};

/** Words with no sign of their own that are dropped rather than fingerspelled. */
const FUNCTION_WORDS = new Set([
  "a", "an", "the",
  "is", "am", "are", "was", "were", "be", "been", "being",
  "do", "does", "did",
  "to", "of", "at", "in", "on", "for", "with", "from", "by", "as", "into",
  "and", "or", "but", "so", "then", "than", "that", "which",
  "there", "it", "its", "any",
  "um", "uh", "er", "just", "really", "very", "actually", "basically",
  // Intensifiers carry no sign of their own in ISL; intensity is non-manual
  // (a larger, sharper movement), which this pipeline cannot produce.
  "much", "too", "quite", "rather", "somewhat", "extremely",
  "please", // handled separately: kept, but never reordered
]);
// "please" is in the vocabulary and must not be dropped. Remove it from the
// function-word set; it is listed above only to document the near-miss.
FUNCTION_WORDS.delete("please");

/** Auxiliaries that only carry tense. The tense is extracted, the word dropped. */
const FUTURE_MARKERS = new Set(["will", "shall", "going", "gonna"]);
const PAST_MARKERS = new Set(["did", "was", "were", "had", "ago"]);

/** Irregular past forms whose base is what the vocabulary knows. */
const PAST_TENSE_BASES: Record<string, string> = {
  went: "go", came: "come", ate: "eat", drank: "drink", gave: "give",
  took: "take", saw: "see", said: "speak", told: "speak", knew: "know",
  thought: "think", brought: "bring", bought: "buy", found: "find",
  lost: "lose", sent: "send", slept: "sleep", woke: "wakeup", sat: "sit",
  stood: "stand", wrote: "write", read: "read", met: "meet", paid: "pay",
  felt: "feel", made: "have", got: "have", had: "have", wore: "wear",
  understood: "understand", forgot: "forget", ran: "go", left: "go",
};

export interface GlossToken {
  gloss: string;
  /** How to caption it — the English word this came from. */
  label: string;
  pos: PartOfSpeech;
  /** True when there is no sign in the library for this word. */
  missing: boolean;
  /** The face to carry over this sign. Never null — neutral is a value. */
  face: NonManual;
}

export type ClauseType = "statement" | "polar" | "content";

export interface GlossResult {
  tokens: GlossToken[];
  /** Words that have no ISL sign in the library, in the order they appeared. */
  unmatched: string[];
  /** The gloss as a signer would write it: "YOU NAME WHAT". */
  notation: string;
  /** True when the sentence was reordered as a question. */
  question: boolean;
  /** What kind of clause this is — the thing the face has to mark. */
  clause: ClauseType;
  /** True when a head shake scopes part of the phrase. */
  negated: boolean;
  /** Human-readable summary of the non-manual marking, for the UI. */
  markers: string[];
}

/**
 * Translate English into an ordered ISL gloss.
 *
 * `available` is the set of glosses the motion library can actually sign. A word
 * that glosses correctly but has no recording is still returned, marked
 * `missing`, because telling the user "there is no sign for 'ambulance' yet" is
 * far more useful than silently dropping the most important word in the
 * sentence.
 */
export function toGloss(text: string, available: ReadonlySet<string>): GlossResult {
  // Punctuation is stripped by normalizeEnglish, so the question mark has to be
  // read first. It is the only evidence a yes/no question gives: "you are a
  // doctor" and "are you a doctor" gloss to the same three signs, and the
  // difference between them lives entirely on the face.
  const askedAsQuestion = /\?\s*$/.test(text.trim());
  const words = normalizeEnglish(text);

  // Pass 1: words -> glosses, longest phrase first.
  interface Matched {
    gloss: string | null;
    label: string;
    pos: PartOfSpeech | null;
  }
  const matched: Matched[] = [];
  let tense: "past" | "future" | null = null;
  let negated = false;

  let i = 0;
  while (i < words.length) {
    const word = words[i];

    if (word === "not" || word === "no" && i > 0) {
      // "no" only negates mid-sentence; on its own it is the sign NO.
      negated = true;
      i += 1;
      continue;
    }
    if (FUTURE_MARKERS.has(word)) {
      tense ??= "future";
      i += 1;
      continue;
    }
    if (PAST_MARKERS.has(word) && !isContentWord(word)) {
      tense ??= "past";
      i += 1;
      continue;
    }

    const phrase = SURFACE_FORMS.find(
      (candidate) =>
        candidate.words.length <= words.length - i &&
        candidate.words.every((w, k) => words[i + k] === w),
    );
    if (phrase) {
      matched.push({
        gloss: phrase.gloss,
        label: phrase.surface,
        pos: POS_BY_GLOSS[phrase.gloss] ?? null,
      });
      i += phrase.words.length;
      continue;
    }

    const base = PAST_TENSE_BASES[word];
    if (base) {
      tense ??= "past";
      matched.push({ gloss: base, label: word, pos: POS_BY_GLOSS[base] ?? null });
      i += 1;
      continue;
    }

    // A word the curated vocabulary does not list, but which the motion library
    // happens to have a sign for. This is what makes the library extensible
    // without editing this file: run ml/add_words.py for a word and it becomes
    // signable immediately, because the gloss is just the word itself.
    if (available.has(word)) {
      matched.push({ gloss: word, label: word, pos: POS_BY_GLOSS[word] ?? null });
      i += 1;
      continue;
    }

    if (FUNCTION_WORDS.has(word)) {
      i += 1;
      continue;
    }

    // A real word with no sign. Kept so the UI can say so.
    matched.push({ gloss: null, label: word, pos: null });
    i += 1;
  }

  // Pass 2: reorder into ISL. Time first, question last, negation after the verb.
  const bucket = (entry: Matched): number => {
    // A modal follows the verb it modifies: "you help can", not "you can help".
    if (entry.gloss && MODALS.has(entry.gloss)) return 7;
    switch (entry.pos) {
      case "time":
        return 0;
      case "greeting":
      case "response":
        return 1;
      case "pronoun":
        return 2;
      case "question":
        return 8;
      case "verb":
        return 6;
      case "adjective":
      case "adverb":
        return 5;
      case "number":
      case "quantifier":
        return 3;
      default:
        return 4; // nouns and unknown words are the topic/object
    }
  };

  const ordered = matched
    .map((entry, index) => ({ entry, index, rank: bucket(entry) }))
    // Stable within a bucket: the original English order is preserved between
    // two nouns, which is what keeps "give mother medicine" from scrambling.
    .sort((a, b) => a.rank - b.rank || a.index - b.index)
    .map((item) => item.entry);

  const tokens: GlossToken[] = [];
  const unmatched: string[] = [];

  // A tense that was carried only by an auxiliary needs a time sign, or the
  // sentence loses its tense entirely.
  if (tense && !ordered.some((entry) => entry.pos === "time")) {
    const marker = tense === "past" ? "yesterday" : "later";
    tokens.push({
      gloss: marker,
      label: tense === "past" ? "(past)" : "(future)",
      pos: "time",
      missing: !available.has(marker),
      face: NEUTRAL_FACE,
    });
  }

  for (const entry of ordered) {
    if (!entry.gloss) {
      unmatched.push(entry.label);
      tokens.push({
        gloss: entry.label,
        label: entry.label,
        pos: "noun",
        missing: true,
        face: NEUTRAL_FACE,
      });
      continue;
    }
    tokens.push({
      gloss: entry.gloss,
      label: entry.label,
      pos: entry.pos ?? "noun",
      missing: !available.has(entry.gloss),
      face: NEUTRAL_FACE,
    });
  }

  // Negation goes after the verb — in practice, at the end of the clause but
  // before any question word.
  if (negated) {
    const questionAt = tokens.findIndex((token) => token.pos === "question");
    const negation: GlossToken = {
      gloss: "no",
      label: "not",
      pos: "response",
      missing: !available.has("no"),
      face: NEUTRAL_FACE,
    };
    if (questionAt >= 0) tokens.splice(questionAt, 0, negation);
    else tokens.push(negation);
  }

  const isContent = tokens.some((token) => token.pos === "question");
  const clause: ClauseType = isContent
    ? "content"
    : askedAsQuestion
      ? "polar"
      : "statement";
  const markers = markUp(tokens, clause, negated);

  return {
    tokens,
    unmatched,
    notation: tokens.map((token) => token.gloss.toUpperCase()).join(" "),
    question: isContent || clause === "polar",
    clause,
    negated,
    markers,
  };
}

/* ------------------------------------------------------- non-manual pass -- */

/**
 * Assign a face to every token, in place, and describe what was assigned.
 *
 * Scope is the whole point. A brow raise on a polar question is held over the
 * entire clause and peaks on its last sign; a WH marker is held over the clause
 * and peaks on the question word; a head shake covers the verb and everything
 * after it, which is what tells a reader that "GO" is what is being negated
 * rather than the subject. Marking only the negation sign itself — the obvious
 * implementation — produces a sentence a signer reads as "you go... no?".
 */
function markUp(
  tokens: GlossToken[],
  clause: ClauseType,
  negated: boolean,
): string[] {
  if (tokens.length === 0) return [];
  const notes: string[] = [];
  const faces = tokens.map(() => ({ ...NEUTRAL_FACE }));

  const apply = (index: number, marker: NonManual, weight: number) => {
    const target = faces[index];
    for (const key of Object.keys(NEUTRAL_FACE) as (keyof NonManual)[]) {
      const value = marker[key] * weight;
      // Strongest wins per channel, so a brow raise and a head shake coexist
      // but two brow instructions do not average into nothing.
      if (Math.abs(value) > Math.abs(target[key])) target[key] = value;
    }
  };

  if (clause === "content") {
    const at = tokens.findIndex((token) => token.pos === "question");
    tokens.forEach((_, index) => apply(index, MARKERS.contentQuestion, 0.6));
    if (at >= 0) apply(at, MARKERS.contentQuestion, 1);
    notes.push("brows furrowed — content question");
  } else if (clause === "polar") {
    tokens.forEach((_, index) => apply(index, MARKERS.polarQuestion, 0.7));
    apply(tokens.length - 1, MARKERS.polarQuestion, 1);
    notes.push("brows raised — yes/no question");
  } else {
    // A statement's topic — a leading time sign or pronoun — takes a brow raise
    // and a small tilt, released before the comment.
    const first = tokens[0];
    if (tokens.length > 1 && (first.pos === "time" || first.pos === "pronoun")) {
      apply(0, MARKERS.topic, 1);
      notes.push("brows raised on the topic");
    }
    if (!negated && tokens.length > 1) {
      apply(tokens.length - 1, MARKERS.affirm, 0.55);
    }
  }

  if (negated) {
    // Scope: from the verb (or, failing that, the negation sign) to the end,
    // stopping before a question word, which is never inside the negation.
    let from = tokens.findIndex((token) => token.pos === "verb");
    if (from < 0) from = tokens.findIndex((token) => token.gloss === "no");
    if (from < 0) from = Math.max(0, tokens.length - 1);
    for (let index = from; index < tokens.length; index += 1) {
      if (tokens[index].pos === "question") break;
      apply(index, MARKERS.negation, index === from ? 0.8 : 1);
    }
    notes.push("head shake over the negated part");
  }

  // Lexical faces: some signs are simply not made with a neutral one.
  tokens.forEach((token, index) => {
    if (WARM.has(token.gloss)) apply(index, MARKERS.warm, 1);
    if (DISCOMFORT.has(token.gloss)) apply(index, MARKERS.discomfort, 1);
    if (token.gloss === "yes") apply(index, MARKERS.affirm, 1);
    if (token.gloss === "no" && !negated) apply(index, MARKERS.negation, 0.9);
  });

  tokens.forEach((token, index) => {
    token.face = face(faces[index]);
  });
  return notes;
}

/** Signs that read as cold or rude on a neutral face. */
const WARM = new Set([
  "hello", "thankyou", "please", "welcome", "goodmorning", "goodnight",
  "goodbye", "happy", "goodafternoon", "goodevening", "nicetomeetyou",
]);

/** Signs whose meaning includes the face that goes with them. */
const DISCOMFORT = new Set([
  "pain", "sick", "sorry", "sad", "afraid", "tired", "headache", "fever",
  "hurt", "emergency", "accident", "problem", "difficult", "angry", "bad",
]);

function normalizeEnglish(text: string): string[] {
  let lowered = text.toLowerCase();
  for (const [contraction, expansion] of Object.entries(CONTRACTIONS)) {
    lowered = lowered.split(contraction).join(expansion);
  }
  return lowered
    .replace(/[^a-z0-9\s']/g, " ")
    .replace(/'/g, "")
    .split(/\s+/)
    .filter(Boolean);
}

/** "was"/"had" can be content words elsewhere; here they never are. */
function isContentWord(word: string): boolean {
  return SURFACE_FORMS.some((form) => form.words.length === 1 && form.words[0] === word);
}

/* ------------------------------------------------------ gloss -> English -- */

/**
 * Rendering a recognised gloss sequence back into English is the half that
 * decides whether a hearing person hears a sentence or a shopping list. It has
 * to undo everything the forward direction did: move the question word back to
 * the front, put the copula back, restore possessives that ISL expressed by
 * juxtaposition, and turn a leading time sign into tense on the verb.
 *
 * The sequence is parsed into slots first and only then rendered. An earlier
 * version spliced words out of an array as it went and produced sentences like
 * "What you are your name?" — the classic result of editing a sentence you have
 * not finished reading.
 */

const BE_FOR: Record<string, string> = {
  i: "am", you: "are", we: "are", they: "are",
  he: "is", she: "is", this: "is", that: "is",
};

const POSSESSIVE: Record<string, string> = {
  i: "my", you: "your", he: "his", she: "her", we: "our", they: "their",
};

const QUESTION_ENGLISH: Record<string, string> = {
  what: "what", where: "where", who: "who", when: "when", why: "why",
  how: "how", howmany: "how many", howmuch: "how much", which: "which",
};

/** Modals modify another verb rather than being the verb. */
const MODALS = new Set(["can", "cannot"]);

/**
 * Nouns that a bare pronoun in front of them possesses.
 *
 * ISL marks possession by juxtaposition: YOU NAME is "your name". But the same
 * two slots can also be a predicate — SHE DOCTOR is "she is a doctor", not "her
 * doctor". The difference is what kind of noun it is: things you *have* (a name,
 * a mother, a stomach, a phone) versus things you *are* (a doctor, a teacher, a
 * student). Only the first list takes the possessive reading.
 */
const POSSESSABLE = new Set([
  "name", "mother", "father", "brother", "sister", "son", "daughter", "wife",
  "husband", "family", "friend", "home", "stomach", "phone", "bag", "key",
  "money", "book", "car", "work", "price", "ticket", "medicine", "clothes",
  "room", "hearingaid", "child", "baby", "people",
]);

/**
 * Kin, body parts and one's own name: nouns that keep the possessive reading
 * even when the sentence has a verb, because they are almost never used without
 * a possessor. "I MOTHER SEE" is "I see my mother"; "I HOME GO" is just "I go
 * home", which is why HOME is possessable but not inalienable.
 */
const INALIENABLE = new Set([
  "name", "mother", "father", "brother", "sister", "son", "daughter", "wife",
  "husband", "family", "stomach",
]);

/** Places that read as a destination after a motion verb: "go to the market". */
const PLACES = new Set([
  "hospital", "school", "college", "office", "shop", "market", "bank",
  "restaurant", "toilet", "kitchen", "room", "station", "city", "village",
  "temple", "home", "india", "here", "there",
]);
const MOTION_VERBS = new Set(["go", "come"]);

/** Mass and abstract nouns that read wrong with an article. */
const NO_ARTICLE = new Set([
  "water", "food", "money", "milk", "tea", "rice", "bread", "time", "help",
  "pain", "blood", "work", "home", "school", "india", "name", "sugar", "salt",
  "fruit", "vegetable", "clothes", "people", "family", "emergency", "fire",
  "signlanguage", "here", "there", "morning", "afternoon", "evening", "night",
  "medicine", "tea", "milk", "bread", "advice", "transport",
]);

/**
 * Verbs whose object is a person and which need "to" in English.
 * ISL directs the verb at the addressee instead of using a preposition.
 */
const NEEDS_TO = new Set(["listen", "speak", "give", "send", "show", "answer"]);

/** Irregular plurals, for "how many ___". */
const PLURAL_OF: Record<string, string> = {
  child: "children", man: "men", woman: "women", person: "people",
  people: "people", money: "money", water: "water", family: "families",
  baby: "babies", city: "cities", key: "keys",
};

type Tense = "past" | "present" | "future";

interface Parsed {
  time: string | null;
  question: string | null;
  negated: boolean;
  modal: string | null;
  subject: string | null;
  verb: string | null;
  objects: string[];
  /** Greetings and one-word responses, which are not sentences. */
  interjections: string[];
}

function parse(glosses: readonly string[]): Parsed {
  const pos = (gloss: string): PartOfSpeech => POS_BY_GLOSS[gloss] ?? "noun";
  const out: Parsed = {
    time: null, question: null, negated: false, modal: null,
    subject: null, verb: null, objects: [], interjections: [],
  };

  const rest: string[] = [];
  for (const gloss of glosses) {
    if (!gloss) continue;
    if (pos(gloss) === "question" && !out.question) {
      out.question = gloss;
      continue;
    }
    if (MODALS.has(gloss)) {
      out.modal = gloss;
      continue;
    }
    if (pos(gloss) === "time" && !out.time && rest.length === 0) {
      out.time = gloss;
      continue;
    }
    rest.push(gloss);
  }

  // A trailing NO negates; a NO on its own is the answer "no".
  if (rest.length > 1 && rest[rest.length - 1] === "no") {
    out.negated = true;
    rest.pop();
  }
  if (out.modal === "cannot") out.negated = true;

  // Greetings and responses are lifted out and rendered as themselves.
  while (rest.length && (pos(rest[0]) === "greeting" || pos(rest[0]) === "response")) {
    out.interjections.push(rest.shift()!);
  }

  if (rest.length && pos(rest[0]) === "pronoun") out.subject = rest.shift()!;

  const verbAt = rest.findIndex((gloss) => pos(gloss) === "verb");
  if (verbAt >= 0) out.verb = rest.splice(verbAt, 1)[0];

  out.objects = rest;
  return out;
}

export function toEnglish(glosses: readonly string[]): string {
  const words = glosses.filter(Boolean);
  if (words.length === 0) return "";

  const p = parse(words);
  const english = (gloss: string): string => ENGLISH_BY_GLOSS[gloss] ?? gloss;
  const pos = (gloss: string): PartOfSpeech => POS_BY_GLOSS[gloss] ?? "noun";

  const tense: Tense =
    p.time === "yesterday" ? "past"
      : p.time === "tomorrow" || p.time === "later" ? "future"
        : "present";

  // "Please, help." reads as a plea for punctuation. Softeners lead without one.
  const SOFTENERS = new Set(["please", "sorry"]);
  const leadWords = p.interjections.map(english);
  const softOnly = p.interjections.every((gloss) => SOFTENERS.has(gloss));
  const lead = leadWords.join(", ");

  // Nothing but greetings: "Hello.", "Thank you.", "Yes."
  if (!p.subject && !p.verb && p.objects.length === 0 && !p.question) {
    return lead ? capitalize(finish(lead)) : "";
  }

  /* ------------------------------------------------------ noun phrases --- */
  // A possessable noun right after the subject pronoun belongs to it, and the
  // pronoun is then not a separate subject: YOU NAME -> "your name".
  let subject = p.subject;
  const objects = [...p.objects];
  let possessedPhrase: string | null = null;
  const possessiveApplies =
    subject !== null &&
    objects.length > 0 &&
    POSSESSIVE[subject] !== undefined &&
    (p.verb ? INALIENABLE.has(objects[0]) : POSSESSABLE.has(objects[0]));
  if (possessiveApplies && subject) {
    possessedPhrase = `${POSSESSIVE[subject]} ${english(objects[0])}`;
    objects.shift();
    // Only a verbless sentence loses its subject to the possessive: in
    // "I MOTHER SEE" the subject is still I.
    if (!p.verb) subject = null;
  }

  const definite = p.question !== null;
  const known = (gloss: string) => POS_BY_GLOSS[gloss] !== undefined;
  const nounPhrase = (gloss: string, plural = false): string => {
    const word = plural ? pluralize(english(gloss)) : english(gloss);
    // An unrecognised gloss is passed through untouched: it is a word the
    // recogniser produced that this grammar knows nothing about, and dressing it
    // up with an article would only make the guess louder.
    if (!known(gloss) || pos(gloss) !== "noun" || NO_ARTICLE.has(gloss) || plural) {
      return word;
    }
    return definite ? `the ${word}` : `${/^[aeiou]/.test(word) ? "an" : "a"} ${word}`;
  };

  // "go hospital" is "go to the hospital"; a destination needs its preposition
  // back, which ISL never had.
  const destination = p.verb !== null && MOTION_VERBS.has(p.verb);
  const objectPhrase = (gloss: string, plural = false): string => {
    if (destination && PLACES.has(gloss)) {
      return NO_ARTICLE.has(gloss) ? english(gloss) : `to the ${english(gloss)}`;
    }
    if (p.verb && NEEDS_TO.has(p.verb) && pos(gloss) === "pronoun") {
      return `to ${objectPronoun(english(gloss))}`;
    }
    if (pos(gloss) === "pronoun") return objectPronoun(english(gloss));
    return nounPhrase(gloss, plural);
  };

  const objectText = (plural = false) =>
    [possessedPhrase, ...objects.map((gloss) => objectPhrase(gloss, plural))]
      .filter(Boolean)
      .join(" ");

  /* ------------------------------------------------------------ questions -- */
  if (p.question) {
    const wh = QUESTION_ENGLISH[p.question] ?? english(p.question);
    // "how many"/"how much" quantify the object, so it comes straight after.
    const counting = p.question === "howmany" || p.question === "howmuch";

    if (p.verb) {
      const bare = english(p.verb);
      const auxiliary = p.modal
        ? modalWord(p.modal)
        : doSupport(subject, tense);
      const parts = counting
        ? [wh, objectText(p.question === "howmany"), auxiliary,
           subject ? english(subject) : "you", bare]
        : [wh, auxiliary, subject ? english(subject) : "you", bare, objectText()];
      return capitalize(parts.filter(Boolean).join(" ")) + "?";
    }

    // Verbless question: WH + be + the thing.
    const complement = objectText(p.question === "howmany") ||
      (subject ? english(subject) : "");
    const usedSubject = !objectText() && subject;
    const be = beWord(usedSubject ? subject : null, complement, tense);
    return capitalize(
      [wh, be, usedSubject ? english(subject!) : complement, p.negated ? "not" : ""]
        .filter(Boolean)
        .join(" "),
    ) + "?";
  }

  /* ------------------------------------------------------------ statements -- */
  let core: string;
  if (p.verb) {
    const verb = p.modal
      ? `${p.negated ? "cannot" : modalWord(p.modal)} ${english(p.verb)}`
      : conjugate(english(p.verb), subject, tense, p.negated);
    core = [subject ? english(subject) : null, verb, objectText()]
      .filter(Boolean)
      .join(" ");
  } else {
    // No verb: English needs a copula. The subject is the pronoun if there is
    // one, otherwise the first noun — "MOTHER SICK" is "Mother is sick."
    let topic = subject ? english(subject) : possessedPhrase;
    let complementParts = objects;
    if (!topic && objects.length) {
      const [first, ...others] = objects;
      // The topic slot takes no article: "Mother is sick", not "A mother is sick".
      topic = english(first);
      complementParts = others;
      possessedPhrase = null;
    }
    const complement = [
      topic === possessedPhrase ? null : possessedPhrase,
      ...complementParts.map((gloss) => nounPhrase(gloss)),
    ]
      .filter(Boolean)
      .join(" ");
    // "X PAIN" is not "X is pain" — English makes it a verb.
    if (complement === "pain" && topic) {
      const hurts = tense === "past" ? "hurt" : "hurts";
      core = [topic, p.negated ? "does not hurt" : hurts].join(" ");
    } else {
      const be = beWord(subject, complement, tense);
      core = [topic, be, p.negated ? "not" : "", complement]
        .filter(Boolean)
        .join(" ");
    }
  }

  const timeWord = p.time ? english(p.time) : "";
  const sentence = [lead ? (softOnly ? lead : `${lead},`) : "", core, timeWord]
    .filter(Boolean)
    .join(" ");
  return capitalize(finish(sentence));
}

function beWord(subject: string | null, _complement: string, tense: Tense): string {
  const present = subject ? BE_FOR[subject] ?? "is" : "is";
  if (tense === "present") return present;
  if (tense === "future") return "will be";
  return present === "are" ? "were" : "was";
}

function doSupport(subject: string | null, tense: Tense): string {
  if (tense === "past") return "did";
  if (tense === "future") return "will";
  return subject === "he" || subject === "she" || subject === "this" || subject === "that"
    ? "does"
    : "do";
}

/** Both CAN and CANNOT render as "can"; CANNOT also sets `negated`. */
function modalWord(_modal: string): string {
  return "can";
}

function conjugate(
  verb: string,
  subject: string | null,
  tense: Tense,
  negated: boolean,
): string {
  if (tense === "future") return negated ? `will not ${verb}` : `will ${verb}`;
  if (tense === "past") {
    if (negated) return `did not ${verb}`;
    return PAST_OF[verb] ?? (verb.endsWith("e") ? `${verb}d` : `${verb}ed`);
  }
  const thirdPerson =
    subject === "he" || subject === "she" || subject === "this" || subject === "that";
  if (negated) return thirdPerson ? `does not ${verb}` : `do not ${verb}`;
  if (!thirdPerson) return verb;
  if (/(s|x|z|ch|sh)$/.test(verb)) return `${verb}es`;
  if (/[^aeiou]y$/.test(verb)) return `${verb.slice(0, -1)}ies`;
  return `${verb}s`;
}

const PAST_OF: Record<string, string> = {
  go: "went", come: "came", eat: "ate", drink: "drank", give: "gave",
  take: "took", see: "saw", speak: "spoke", know: "knew", think: "thought",
  bring: "brought", buy: "bought", find: "found", lose: "lost", send: "sent",
  sleep: "slept", sit: "sat", stand: "stood", write: "wrote", read: "read",
  meet: "met", pay: "paid", have: "had", wear: "wore", understand: "understood",
  forget: "forgot", "wake up": "woke up",
};

/** "I" and "he" are subjects; as objects English wants "me" and "him". */
function objectPronoun(word: string): string {
  const map: Record<string, string> = {
    i: "me", he: "him", she: "her", we: "us", they: "them", you: "you",
  };
  return map[word] ?? word;
}

function pluralize(word: string): string {
  const irregular = PLURAL_OF[word];
  if (irregular) return irregular;
  if (/(s|x|z|ch|sh)$/.test(word)) return `${word}es`;
  if (/[^aeiou]y$/.test(word)) return `${word.slice(0, -1)}ies`;
  return `${word}s`;
}

function capitalize(text: string): string {
  const trimmed = text.replace(/\s+/g, " ").trim();
  return trimmed.charAt(0).toUpperCase() + trimmed.slice(1);
}

function finish(text: string): string {
  const trimmed = text.replace(/\s+/g, " ").trim();
  return /[.!?]$/.test(trimmed) ? trimmed : `${trimmed}.`;
}

/**
 * English text -> ISL gloss tokens (P10).
 *
 * Deliberately simple, per the roadmap: lowercase, strip punctuation, drop
 * filler words and articles, fold a few synonyms, then split into tokens the
 * clip library can (or cannot) serve. Real ISL grammar reordering is far out of
 * scope — this produces a word-for-word gloss of the content words.
 */

const STOPWORDS = new Set([
  "a", "an", "the",
  "is", "am", "are", "was", "were", "be", "been", "being",
  "do", "does", "did",
  "will", "would", "shall", "should", "can", "could", "may", "might",
  "to", "of", "at", "in", "on", "for", "with", "and", "or", "so", "that",
  "this", "these", "those", "there", "it", "its",
  "um", "uh", "er", "like", "just", "really", "very",
]);

/**
 * Multi-word signs, matched BEFORE single words and longest-first. ISL renders
 * "good morning" as one sign, not "good" + "morning", so a word-by-word gloss
 * would be wrong (and would queue clips that do not exist).
 */
const PHRASES: [string[], string][] = [
  [["how", "are", "you"], "howareyou"],
  [["good", "morning"], "goodmorning"],
  [["thank", "you"], "thankyou"],
  [["thanks", "a", "lot"], "thankyou"],
  [["nice", "to", "meet", "you"], "pleased"],
  [["how", "do", "you", "do"], "howareyou"],
];

/** Spoken variants folded onto clip vocabulary words. */
const SYNONYMS: Record<string, string> = {
  hi: "hello",
  hey: "hello",
  hallo: "hello",
  thanks: "thankyou",
  thank: "thankyou",
  ty: "thankyou",
  ok: "alright",
  okay: "alright",
  fine: "alright",
  good: "alright",
  great: "alright",
  glad: "pleased",
  happy: "pleased",
  nice: "pleased",
  morning: "goodmorning",
};

export interface GlossToken {
  word: string;
  /** Whether the clip library has a video for this word. */
  matched: boolean;
}

export interface GlossResult {
  tokens: GlossToken[];
  matched: string[];
  unmatched: string[];
}

export function gloss(text: string, available: ReadonlySet<string>): GlossResult {
  const tokens: GlossToken[] = [];

  const words = text
    .toLowerCase()
    .replace(/[^a-z\s']/g, " ")
    .replace(/'/g, "")
    .split(/\s+/)
    .filter(Boolean);

  // Longest phrases first, so "how are you" wins over "you".
  const phrases = [...PHRASES].sort((a, b) => b[0].length - a[0].length);

  let i = 0;
  while (i < words.length) {
    const phrase = phrases.find(([seq]) =>
      seq.every((w, k) => words[i + k] === w),
    );
    if (phrase) {
      tokens.push({ word: phrase[1], matched: available.has(phrase[1]) });
      i += phrase[0].length;
      continue;
    }
    const raw = words[i];
    i += 1;
    if (STOPWORDS.has(raw)) continue;
    const word = SYNONYMS[raw] ?? raw;
    tokens.push({ word, matched: available.has(word) });
  }

  return {
    tokens,
    matched: tokens.filter((t) => t.matched).map((t) => t.word),
    unmatched: tokens.filter((t) => !t.matched).map((t) => t.word),
  };
}

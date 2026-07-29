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

/** Spoken variants folded onto clip vocabulary words. */
const SYNONYMS: Record<string, string> = {
  hi: "hello",
  hey: "hello",
  thank: "thanks",
  thankyou: "thanks",
  ok: "yes",
  okay: "yes",
  yeah: "yes",
  yep: "yes",
  nope: "no",
  food: "eat",
  eating: "eat",
  goodbye: "bye",
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

  for (const raw of words) {
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

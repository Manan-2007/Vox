"""Generate the frontend's copy of the vocabulary from ml/vocabulary.py.

The word list has to exist in both languages: Python resolves it against the
dictionary and trains on it, TypeScript glosses English into it and renders it
back. Maintaining two hand-written copies of 250 words and 570 surface forms
would guarantee they drift, and the failure mode of that drift is silent — a
word the avatar can sign but the gloss engine will never produce.

So there is one source, ml/vocabulary.py, and this writes the other:

    python ml/export_vocabulary.py

Run it after editing the vocabulary. The generated file is committed, so the
frontend builds without a Python step.
"""

from __future__ import annotations

import sys
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ML_DIR))
from vocabulary import VOCABULARY  # noqa: E402

OUT = ML_DIR.parent / "frontend" / "src" / "isl" / "vocabulary.generated.ts"

HEADER = '''/**
 * GENERATED FILE — do not edit.
 *
 * Source: ml/vocabulary.py
 * Regenerate: python ml/export_vocabulary.py
 *
 * The ISL vocabulary shared by the whole product: the glosses the recogniser can
 * predict, the English words that map onto each one, and the part of speech the
 * grammar engine in ./grammar.ts needs to order a sentence.
 */

export type PartOfSpeech =
  | "pronoun"
  | "noun"
  | "verb"
  | "adjective"
  | "adverb"
  | "time"
  | "question"
  | "number"
  | "response"
  | "greeting"
  | "quantifier";

export interface VocabularyEntry {
  gloss: string;
  /** Every English form that maps to this sign; the first is the preferred one. */
  surfaces: string[];
  pos: PartOfSpeech;
}

export const VOCABULARY: VocabularyEntry[] = [
'''

FOOTER = '''];

export const POS_BY_GLOSS: Record<string, PartOfSpeech> = Object.fromEntries(
  VOCABULARY.map((entry) => [entry.gloss, entry.pos]),
);

export const ENGLISH_BY_GLOSS: Record<string, string> = Object.fromEntries(
  VOCABULARY.map((entry) => [entry.gloss, entry.surfaces[0]]),
);

/**
 * English surface form -> gloss, longest first.
 *
 * Order matters: "how are you" has to be tried before "how", or the greeting is
 * lost and the sentence becomes a question about nothing. Callers walk this list
 * in order and take the first phrase that matches at the current position.
 */
export const SURFACE_FORMS: { surface: string; words: string[]; gloss: string }[] =
  VOCABULARY.flatMap((entry) =>
    entry.surfaces.map((surface) => ({
      surface,
      words: surface.split(" "),
      gloss: entry.gloss,
    })),
  ).sort((a, b) => b.words.length - a.words.length);
'''


def main() -> None:
    lines = [HEADER]
    for gloss, surfaces, pos in VOCABULARY:
        forms = ", ".join(f'"{s}"' for s in surfaces)
        lines.append(f'  {{ gloss: "{gloss}", surfaces: [{forms}], pos: "{pos}" }},\n')
    lines.append(FOOTER)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(lines), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ML_DIR.parent)} — {len(VOCABULARY)} glosses")


if __name__ == "__main__":
    main()

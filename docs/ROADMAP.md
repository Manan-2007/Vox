# Vox — the four goals, honestly assessed

The stated ambition:

1. A full two-way interpreter — a Deaf signer and a hearing speaker holding a
   natural conversation.
2. Recognition that generalizes across signers, lighting, and regional variation.
3. Continuous signing — whole sentences signed fluidly, not word by word.
4. Facial expression and body pose as part of meaning, the way real ISL uses them.

Note on goal 3: *recognising* continuous signing and *producing* it are separate
problems, and only the first is unsolved here. Playback already renders a phrase
as one continuous animation — sign, hold, then an eased transitional movement
into the next sign — because a reader segments the stream on exactly those
transitions. Concatenating recordings end to end, which is what the first
version did, produces one continuous blur with no word boundaries in it.

Nothing below is marked done unless it is measured. Numbers come from
`ml/models/metrics.json`, produced by `python ml/evaluate.py` on a test split
held out by source video.

---

## 1. Two-way interpreter — **built, and now actually a translator**

Working end to end:

- signer → landmarks → BiLSTM → gated word → **ISL grammar → English sentence**
  → speech
- speaker → speech or typing → **English → ISL gloss, reordered** → 3D signer

The grammar step is what changed. Word-for-word substitution produced
"you name what" in one direction and read it back as "you name what" in the
other. `frontend/src/isl/grammar.ts` now implements SOV order, copula and
article dropping, time-first tense, question-word-final, and post-verbal
negation — in both directions. Verified on 20 sentences each way.

| remaining gap | cost |
|---|---|
| Turn-taking is a pause timeout | Medium — needs a "signer is signing" detector, which is the same problem as goal 3. The motion-energy signal in `tracking.ts` is the raw material. |
| No repair ("sorry, what?") | Medium — needs confidence-aware dialogue, not just a threshold. |
| No classifier predicates or verb agreement | High — needs a rig that can *aim* a sign, not a library of fixed recordings. |

---

## 2. Generalization — **measured, and the honest answer is "partly"**

Over 242 classes, on recordings never trained on:

| metric | value |
|---|---|
| top-1 | 38.2% |
| top-3 | 45.1% |
| **correct when it speaks** (85% gate) | **77.2%**, on 39.5% of samples |
| chance baseline | 0.4% |
| words verified at 80%+ | 37 |
| words never measurable | 118 |

The gated number is the one that describes the product, because the backend
stays silent below its confidence bar. Staying silent is the correct failure
mode: a wrong word makes a hearing person act on something the signer never
said.

**Why the raw number is low, precisely.** The dictionary provides one signer per
word. 118 of 242 words have a single recording, so nothing can be held back to
test them and the model has one example to learn from. Accuracy by data volume
makes the relationship plain:

| samples per word | accuracy |
|---|---|
| 1–2 | 30.2% |
| 3–5 | 24.4% |
| 6+ | 45.3% |

**The fix is data, not architecture.** In order of value:

1. **Record yourself.** `python ml/collect.py <word>`, 30-40 samples, mixed in
   and retrained. This is the only change that directly fixes recognition for
   *your* hands, *your* camera and *your* room.
2. **More INCLUDE categories.** [Zenodo 4010759](https://zenodo.org/records/4010759)
   has 21 signers per word for 263 words. It is 57 GB and throttled to roughly
   300 KB/s, so a category is an hour or two — but it is the only source of
   genuine signer diversity that exists for ISL at this scale.
3. **CISLR** ([IIT-K/CISLR](https://huggingface.co/datasets/IIT-K/CISLR)) has
   4,765 words but is access-gated and averages 1.5 videos per word — good for
   avatar coverage, poor for training.

---

## 3. Continuous signing — **not started, and genuinely hard**

Vox recognizes *isolated* signs: a rolling two-second window, one label. Fluent
signing has no gaps to segment on, signs blend into each other, and the boundary
between two signs is not observable from landmarks alone.

Cheapest first:

1. **CTC over the landmark stream** — learns alignment instead of requiring it.
   Needs sentence-level data; INCLUDE and the dictionary are both word-level.
   [ISLTranslate](https://aclanthology.org/2023.findings-acl.665.pdf) (31k ISL
   sentence pairs) or ISL-CSLTR are the candidates.
2. **Transformer encoder + CTC** — better long-range modelling, more data still.
3. **Sign spotting** — find known signs inside continuous video without
   segmenting everything. Degrades gracefully, and is the realistic next step.

The `rest` rejection class and the motion-energy signal are the groundwork: both
are ways of asking "is a sign happening right now", which is the first half of
segmentation.

**Cost: weeks, and a sentence-level ISL corpus.** This remains the single
biggest gap between Vox and an interpreter.

---

## 4. Face and body as meaning — **body done, face done in one direction**

- **Body: done.** Shoulders, elbows and nose are in the recogniser's frame
  contract, and normalization anchors on the shoulders. This is what made
  recognition work at all. The avatar carries a wider 13-point block with depth,
  so the figure is a body rather than a flat diagram — see `docs/3D-AVATAR.md`.

- **Face, production: done.** The avatar has a face and uses it grammatically.
  `frontend/src/isl/grammar.ts` decides the clause type and
  `frontend/src/avatar/nonManual.ts` turns it into brows, lids, mouth and head
  movement, scoped to the right signs:

  | clause | marker | scope |
  |---|---|---|
  | polar question | brows raised, head forward | whole clause, peaking on the last sign |
  | content question | brows furrowed, squint, head tilt | whole clause, peaking on the WH sign |
  | negation | head shake | the verb and everything after it, stopping before a WH sign |
  | topic | brows raised, small tilt back | the leading time or pronoun |

  Scope is the part that matters. Marking only the negation sign itself — the
  obvious implementation — produces a sentence a signer reads as "you go… no?".

  The markers are **synthesised from the syntax rather than extracted from the
  dictionary clips**, and that is not a shortcut. Those clips are citation
  forms: one word signed in isolation with a deliberately neutral face. The
  marker belongs to the sentence, not the word, so it is not in any recording of
  a word.

- **Face, recognition: still absent.** Vox cannot *read* a signer's brows, so a
  signed yes/no question still arrives as a statement.

  Adding it is mechanically straightforward and scientifically not. MediaPipe
  `FaceLandmarker` gives 52 blendshape coefficients, which is a compact and
  well-suited representation. That is a contract change and a retrain.

  **The blocker is labels.** The dictionary labels *words*, not facial grammar.
  A model given face features but no facial-grammar labels learns nothing new
  from them. Worth doing only alongside a corpus annotated for non-manual
  markers.

**Still not modelled in either direction:** mouth morphemes that distinguish
minimal pairs, use of signing space to set up a referent and point back at it,
and role shift.

---

## If you do one thing next

Record yourself signing the twenty words you would actually use, and retrain.
A few hours, no new research, and it moves the number that matters more than any
architecture change available.

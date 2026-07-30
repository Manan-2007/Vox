# Vox — the four goals, honestly assessed

The stated ambition:

1. A full two-way interpreter — a Deaf signer and a hearing speaker holding a
   natural conversation.
2. Recognition that generalizes across signers, lighting, and regional variation.
3. Continuous signing — whole sentences signed fluidly, not word by word.
4. Facial expression and body pose as part of meaning, the way real ISL uses them.

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

## 4. Face and body as meaning — **body done, face not**

- **Body: done.** Shoulders, elbows and nose are in the frame contract, and
  normalization anchors on the shoulders. This is what made recognition work at
  all.
- **Face: absent.** ISL uses eyebrows, mouth morphemes, head tilt and gaze
  grammatically — negation, questions and intensity live there. Vox is blind to
  all of it, and the avatar cannot produce any of it.

Adding it is *mechanically* straightforward and *scientifically* not. MediaPipe
`FaceLandmarker` gives 52 blendshape coefficients, which is a compact and
well-suited representation. That is a contract change and a retrain.

**The blocker is labels.** The dictionary labels *words*, not facial grammar. A
model given face features but no facial-grammar labels learns nothing new from
them. Worth doing only alongside a corpus annotated for non-manual markers.

---

## If you do one thing next

Record yourself signing the twenty words you would actually use, and retrain.
A few hours, no new research, and it moves the number that matters more than any
architecture change available.

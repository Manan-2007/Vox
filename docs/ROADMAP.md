# Vox — the four goals, honestly assessed

The stated ambition:

1. A full two-way interpreter — a Deaf or hard-of-hearing signer and a hearing
   speaker holding a natural conversation.
2. Recognition that generalizes across signers, lighting, and regional ISL
   variation.
3. Continuous signing — whole sentences signed fluidly, not word by word.
4. Facial expression and body pose as part of meaning, the way real ISL uses
   them.

Below: where each one actually stands, and what closing it costs. Nothing here
is marked done unless it is measured.

---

## 1. Two-way interpreter — **mostly built**

Working end to end today:

- signer → landmarks → LSTM → gated word → sentence → speech (optional)
- speaker → speech recognition → ISL gloss → clip queue + 3D motion

What is missing for it to feel *natural*:

| gap | cost |
|---|---|
| Vocabulary of 6 signs | Low — each INCLUDE category adds ~20 words with 21 signers. Bounded, mechanical work. |
| Turn-taking is manual (pause timeout / button) | Medium — needs a "signer is signing" vs "signer stopped" detector, which is the same segmentation problem as goal 3. |
| No repair ("sorry, what?") | Medium — needs confidence-aware dialogue, not just a threshold. |

**Verdict: the architecture is right; it is a vocabulary and turn-taking
problem, not a redesign.**

---

## 2. Generalization — **partly solved, and measured**

This is the one where the numbers moved most. Same code, same evaluation
(split by *source video*, so no signer appears on both sides):

| training data | unseen-signer accuracy |
|---|---|
| 2–8 YouTube dictionary videos per word | 27.9% |
| 21 INCLUDE signers per word | **100%** (6 signs) |

What that number does **not** cover:

- **Lighting / background.** INCLUDE was recorded in varied conditions, which
  helps, but every signer is filmed front-on at similar distance. A dim room
  or strong backlight is untested.
- **Regional variation.** INCLUDE uses one set of forms. A signer using a
  different regional variant of "thank you" is, to this model, a different
  sign. This is a *data* problem: it needs samples of each variant, labelled
  as the same word.
- **Population size.** 21 signers is small. It is enough to stop the model
  memorising one person; it is not enough to claim it works for everyone.

**Next concrete step:** `python ml/collect.py <word>` and mix your own
recordings in. Personal data is the cheapest large accuracy win available,
and it directly fixes "it does not recognize *my* signing".

---

## 3. Continuous signing — **not started, and genuinely hard**

Today Vox recognizes *isolated* signs: a rolling 2-second window, one label.
Fluent signing has no gaps to segment on, signs blend into each other
(coarticulation), and the boundary between two signs is not observable from
landmarks alone.

The honest options, cheapest first:

1. **CTC over the landmark stream.** Replace the softmax-per-window with a
   sequence model trained with Connectionist Temporal Classification, which
   learns alignment instead of requiring it. Needs *sentence-level* data —
   INCLUDE is word-level, so this needs a different dataset
   (e.g. [ISLVT](https://data.mendeley.com/datasets/98mzk82wbb/1) for ISL
   sentences, or How2Sign / PHOENIX for other languages).
2. **Transformer encoder + CTC.** Better long-range modelling, more data
   still.
3. **Sign spotting** — detect known signs inside continuous video without
   segmenting everything. A useful middle step, and it degrades gracefully.

**Cost: weeks, and a sentence-level ISL corpus we do not currently have.**
This is a research project, not a feature. It is the single biggest gap
between Vox and a real interpreter, and pretending otherwise would be
dishonest.

The existing transition-artifact problem (documented in `backend/main.py`) is
a symptom of exactly this: a sliding window straddling two signs produces a
confident wrong answer, because the model was never taught what "between
signs" looks like.

---

## 4. Face and body as meaning — **body done, face not**

- **Body: done.** The frame contract carries shoulders, elbows and nose, and
  normalization anchors on the shoulders. This is what made recognition work
  at all (see `ml/normalize.py`).
- **Face: not captured.** ISL uses eyebrows, mouth morphemes, head tilt and
  gaze grammatically — negation, questions, and intensity live there. Vox is
  blind to all of it.

Adding it is *mechanically* straightforward and *scientifically* not:

- MediaPipe `FaceLandmarker` gives 52 blendshape coefficients (browser and
  Python), which is a compact, well-suited representation — far better than
  468 raw points.
- That means a contract change to ~193 floats, a worker change, and a
  retrain.
- **The blocker is labels.** INCLUDE labels *words*, not facial grammar. A
  model given face features but no facial-grammar labels learns nothing new
  from them. To use the face you need data annotated for non-manual markers.

**Verdict: cheap to plumb, useless without the right labels.** Worth doing
only alongside a dataset that annotates non-manual features.

---

## Reference projects, and what is actually reusable

| project | what it is | usable here? |
|---|---|---|
| [SignAvatars](https://github.com/ZhengdiYu/SignAvatars) | 70K 3D holistic motion sequences, SMPL-X mesh annotations (ECCV 2024) | **No, for now.** No ISL subset, request-gated, ships no RGB video, and SMPL-X cannot be redistributed — a fresh clone would not work. Full assessment in [3D-AVATAR.md](3D-AVATAR.md). |
| [Sign-Kit](https://github.com/spectre900/Sign-Kit-An-Avatar-based-ISL-Toolkit) | Web toolkit, speech → ISL via a rigged 3D avatar (three.js) | **Direction, not code.** It is *production* only (no recognition). Confirms a rigged avatar beats video clips for the speech→sign half: no per-word licensing, any camera angle. The natural successor to our clip player. |
| [Voice2sign](https://github.com/vinaygowdan06/Voice2sign), [ISL-Detection](https://github.com/MaitreeVaria/Indian-Sign-Language-Detection), [ISL-Recognition](https://github.com/gadmin7/Indian-Sign-Language-Recognition) | Student ISL projects: CNN on cropped hand images, or MediaPipe + classifier on **static alphabet** signs | **No.** Static-alphabet CNNs solve an easier problem (no motion, no signing space) and do not transfer to word-level dynamic signs. Vox's landmark+LSTM approach is already the stronger design. |
| [INCLUDE](https://zenodo.org/records/4010759) | 4287 ISL videos, 263 words, 7+ signers, CC-BY-4.0 | **Yes — already in use.** This is the highest-leverage asset found, and the reason recognition works at all. 254 words remain unused. |

The pattern across the ISL repos surveyed: nearly all do **static alphabet
recognition from images**. Vox is doing dynamic word-level recognition from
body-anchored landmark sequences, which is a harder and more useful problem —
and the reason the reference implementations were not worth copying.

---

## If you do one thing next

**Record yourself.** 30–40 samples per word with `ml/collect.py`, mixed into
the INCLUDE data, then retrain. It is a few hours of work, needs no new
research, and it is the only change that directly fixes recognition for *your*
hands, *your* camera, and *your* lighting.

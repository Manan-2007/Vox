# The 3D signer — how it is built, and why it is built that way

## What ships today

A procedural, anatomically constrained upper-body figure in three.js, driven by
landmark motion. Two files:

| file | owns |
|---|---|
| `frontend/src/avatar/skeleton.ts` | where the body IS — depth recovery, bone constraints, IK, smoothing, the rest pose |
| `frontend/src/avatar/rig.ts` | what the body is MADE OF — torso, arms, hands, head, face |

The split is not tidiness. The previous version mixed the two, so a change to
how an arm was placed meant editing the code that drew a fingernail.

One rig serves both directions:

- **your signing** — frames from the camera Web Worker, mirrored
- **signing to you** — a queue of signs from `frontend/public/signs/`

Same object, same geometry, so a learner comparing their own hands to the
reference is looking at one thing rather than relating two.

---

## The four defects this replaced

The earlier rig drew a line from each tracked point to the next. That is a
scatter plot, not a body, and it failed in four visible ways:

1. **No depth.** Every body joint was pinned to `z = 0` while the hands carried
   real metric depth. The arms were a flat cut-out — and, because the torso is a
   solid standing forward of that plane, the arms were drawn *inside the chest*
   and appeared to emerge somewhere near the neck.
2. **Rubber bones.** A limb whose ends are two independently tracked points is
   as long as the tracker says it is. MediaPipe misplaces an elbow constantly
   under foreshortening, so the upper arm stretched by up to a third within a
   single two-second sign.
3. **Amputation.** The forearm was drawn only when the *hand* landmarker had a
   hand. It loses one about a tenth of the time; the arm then ended at the elbow.
4. **No idle.** Tracking loss fed the raw values straight through, so an empty
   room produced whatever the pose model hallucinated, at full speed, undamped.

---

## How depth is recovered

MediaPipe's pose `z` is a weak estimate: good enough to say which side of the
body a joint is on, not good enough to place it. So it supplies only the
*direction*, and the magnitude comes from the bone:

```
dz    = clamp(z_child - z_parent, ±sqrt(L² - d²))
scale = sqrt(L² - dz²) / d
child = parent + (dxy · scale, dz)
```

where `d` is the parent-to-child distance *as projected on screen* and `L` is
the bone's true length. The result is exactly `L` long whatever the tracker
said, leans the way the tracker leaned, and degrades to a flat-but-correct arm
when there is no depth hint at all. Screen position — the part MediaPipe is
actually good at — is preserved up to that one scale factor.

When an elbow is not tracked at all, it is solved by two-bone IK from the
shoulder and the hand with a pole vector pointing down, out and slightly back —
which is where a human elbow goes, and which stops the arm collapsing into a
straight line through the chest.

**Bone lengths are constants, not measurements.** An earlier version learned
them from the longest projection seen so far, on the sound theory that a bone
parallel to the image plane projects at its true length. It measures well and
animates badly: the estimate climbs as the arm straightens, so the arm grows
during the sign. Everything here is already in shoulder-width units, and a
tracked upper arm measures 0.79–0.82 of one against the 0.80 the anatomy table
predicts — so the constant is both steadier and about as accurate.

## Staying out of the chest

An arm is kept in front of the torso solid, in **depth only** and **forward
only**. Both restrictions are load-bearing:

- Not a radial push along the torso's ellipse normal. At shoulder height the
  chest is wider than the shoulder joints are apart, so an arm hanging naturally
  at the signer's side is *inside* the ellipse, and a radial escape splays it
  outward — away from where the tracker actually saw it.
- Not "to the nearer surface". That puts a discontinuity at `z = 0`, which is
  exactly where a lowering arm passes through, and the joint then jumps the
  whole depth of the chest in one frame.

The lift and the length clamp are one projection rather than two steps, because
push-then-clamp does not converge: the clamp scales the joint back toward the
parent and undoes part of the push, so the pair oscillates and the number of
iterations that ran becomes visible as jitter.

## The rest pose

There is always a coherent pose to be in. When measurement stops, every joint
relaxes toward a standing figure over about half a second, driven by critically
damped springs — critically damped rather than exponential because an
exponential filter slow enough to kill tracker jitter also visibly lags a fast
sign. Losing tracking is a transition, not a cliff.

The figure is dimmed rather than hidden while it idles. Hiding it makes the
stage blank and unblank every time a shoulder is occluded, and a blink is more
alarming than a person standing still.

---

## Verifying it

`frontend/src/avatar/__checks__/skeleton.check.ts` runs the solver over every
frame of the whole library and asserts the invariants the rig exists to
guarantee. This matters because the rig is the hardest part of the app to review
by eye: a figure can look roughly right while stretching a bone by 30% or
burying a forearm in its own chest for a few frames.

```bash
cd frontend
npx rolldown src/avatar/__checks__/skeleton.check.ts -f esm \
  -o /tmp/check.mjs --platform node
node /tmp/check.mjs
```

Current output over 239 signs / 7,189 frames:

| invariant | before | after |
|---|---|---|
| bone length spread within one sign | 34.7% | **0.00%** |
| joints inside the torso solid | 13.0% | **0.07%** |
| frames with an arm missing | — | **0%** |
| frames with at least one hand | 90% | **100%** |
| largest single-frame move while relaxing | 0.41 units | **0.044** |
| head detached from the neck | — | **0%** |

---

## The face

`frontend/src/avatar/nonManual.ts`. In ISL a raised brow is not an expression,
it is a morpheme: the same handshapes mean different things under a different
face. `YOU DOCTOR` with brows raised is "Are you a doctor?"; with brows furrowed
and a head tilt it is "Which doctor are you?"; with a head shake over the verb,
the verb is negated.

The markers are **synthesised from the syntax**, not extracted from the
dictionary clips, and that is not a shortcut. Those clips are *citation forms* —
one word signed in isolation with a deliberately neutral face, because the
dictionary is teaching the handshape. The marker belongs to the sentence, not
the word, so it does not exist in any recording of a word. `isl/grammar.ts`
already knows the clause type, and that is where the marker comes from.

Scope is stated in the UI rather than implied: the gloss line names the marker
in play ("brows raised — yes/no question"), so a viewer can tell the grammar
from the avatar having a mood.

What is modelled: polar questions, content questions, negation scope, topic
marking, and a small lexical set (greetings warm, pain/sorry/sick uncomfortable).
What is not: mouth morphemes that distinguish minimal pairs, eye-gaze reference
to points in signing space, role shift.

---

## What is still not here

- **No skinned mesh.** The figure is solids posed per frame, not a rigged body
  with weighted deformation. Joints are rounded rather than continuous.
- **No finger-level twist.** The hands are built from MediaPipe's 21 metric
  world landmarks, which carry real shape and orientation but no rotation about
  a bone's own axis.
- **The torso does not yaw.** Signers in the source footage face the camera; a
  real body-orientation estimate needs a cue the pose block does not carry
  reliably, and guessing one makes the figure swivel at random.
- **The lower body is out of frame on purpose** — the camera frames chest-up,
  the way sign reference footage is framed.

### On SMPL-X and SignAvatars

[SignAvatars](https://github.com/ZhengdiYu/SignAvatars) (ECCV 2024) is a
large-scale 3D sign holistic motion dataset with SMPL-X mesh annotations
(~70K sequences, 8.34M 3D annotations). It would replace the top two items above
with a rigged, skinned body and full MANO hand articulation.

Concrete blockers, unchanged:

1. **Language mismatch.** None of its subsets are ISL, and it ships no RGB
   video. For Vox's vocabulary it provides no training labels at all — its value
   here is the avatar and the motion prior, not recognition data.
2. **SMPL-X licensing.** The body model is a separate MPI registration and
   cannot be redistributed, so a fresh clone would not just work.
3. **Runtime weight.** Shipping the body model (tens of MB) plus browser
   skinning is a different performance budget from posing a few hundred solids.
4. **Retargeting.** MediaPipe landmarks → SMPL-X pose parameters is an
   inverse-kinematics fit, not a mapping. Done badly it looks broken.

**Recommendation: not now.** It would improve how signs are *displayed*, not how
well they are *recognized*, and recognition is the weak link. For ISL
recognition the higher-leverage dataset remains
**[INCLUDE](https://zenodo.org/records/4010759)** — real ISL, openly licensed,
directly usable by `ml/extract.py`.

# 3D holistic motion in Vox — what's built, and SignAvatars

## What ships today

`frontend/src/components/SignAvatar3D.tsx` renders an upper-body **joint
skeleton** in three.js — both hands (21 landmarks each), plus shoulders,
elbows and head — driven by the same 141-float frames the recognizer consumes.
It appears in two places:

- **Camera panel** — your own signing, mirrored, live from the Web Worker.
- **Speech → ISL panel** — the *reference* motion for the clip playing,
  replayed from `frontend/public/clips/motion.json` (produced by
  `ml/clip_motion.py`, which runs the clips back through MediaPipe at 15 fps).

Both share one renderer, so the "what I'm doing" and "what it should look
like" views are geometrically comparable.

### Honest limits of this avatar

- It is a **skeleton, not a mesh** — joints and bones, no skin, no body shape.
- **Depth is weak.** MediaPipe's `z` is a rough per-landmark estimate, not
  metric depth. It is smoothed and scaled to ~35% so the view reads as
  "3D-ish" rather than pretending to be a true 3D reconstruction. The slow
  orbit is what actually communicates dimensionality.
- **No face, no finger-level rotations** beyond landmark positions.

---

## SignAvatars — what it is, and what adopting it would take

[SignAvatars](https://github.com/ZhengdiYu/SignAvatars) (ECCV 2024) is the
first large-scale **3D sign language holistic motion dataset with mesh
annotations**: ~70K motion sequences and 8.34M 3D annotations, expressed as
**SMPL-X** whole-body parameters (with a MANO hands variant). Subsets cover
ASL, GSL, HamNoSys-notated signs, and word-level English.

This is a genuinely different class of representation from what Vox has:

| | Vox today | SignAvatars |
|---|---|---|
| Representation | 2.5D landmark positions | SMPL-X pose/shape parameters |
| Output | joint skeleton | rigged, skinned body mesh |
| Hands | 21 landmarks/hand | full MANO articulation |
| Face | none | FLAME expression params |
| Source | MediaPipe, live | fitted to video offline |

### What it would unlock

1. **A real avatar for speech → sign.** Instead of playing a recorded video
   clip, drive a rigged SMPL-X character from motion parameters — consistent
   appearance, any camera angle, no per-word video licensing.
2. **Motion priors.** Their learned prior could clean up jittery MediaPipe
   output, or fill occluded frames.
3. **Sign language production (SLP)** — generating novel sign motion from
   text, which recorded clips can never do.

### What it would actually cost

Concrete blockers, in order:

1. **Access + language mismatch.** The data is request-form gated
   (non-commercial research) and ships **no RGB video** — you must obtain
   source videos separately from How2Sign / RWTH-PHOENIX / WLASL. Critically,
   **none of its subsets are Indian Sign Language.** For Vox's ISL vocabulary
   it provides no training labels at all; its value here is the *avatar and
   motion prior*, not recognition data.
2. **SMPL-X body model licensing.** SMPL-X itself is a separate registration
   (MPI). Redistributing the model files with Vox is not permitted, so a
   fresh clone could not just work — exactly the failure mode we removed.
3. **Runtime weight.** Skinning SMPL-X in the browser means shipping the body
   model (tens of MB) plus a skinning implementation, and running it per
   frame. That is a different performance budget from drawing ~47 line
   segments.
4. **A retargeting layer.** MediaPipe landmarks → SMPL-X pose parameters is
   an inverse-kinematics fit, not a mapping. Doing it well is its own
   project; doing it badly produces an avatar that looks broken.

### Recommendation

**Not now, and not for accuracy.** SignAvatars would improve how signs are
*displayed*, not how well they are *recognized* — and recognition is the weak
link. It also cannot be shipped in a self-contained repo because of the SMPL-X
licence.

The path if you want it later:

1. Keep the current skeleton avatar as the fallback renderer.
2. Request SignAvatars + SMPL-X access; prototype **offline** (Python +
   their visualization tools) rendering one ISL word.
3. Retarget MediaPipe → SMPL-X with a small IK fit, evaluated on the clips
   we already have.
4. Only then consider a browser runtime (three.js SMPL-X skinning), behind a
   feature flag, with the skeleton as the default.

For ISL recognition specifically, the higher-leverage dataset is
**[INCLUDE](https://zenodo.org/records/4010759)** (CC-BY-4.0, 4287 videos,
263 ISL words, 7 signers) — real ISL, openly licensed, and directly usable by
`ml/extract.py`.

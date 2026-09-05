# Vox — system audit, 2026-07-31

Written before any code changed. Every claim below is traced to a file, a line, or
a measurement taken from the repository as it stands at `fbc4a34`.

The short version: **three of the four subsystems are not repairable by patching,
and the reason is the same in each case — they are built on the wrong primitive.**

| Subsystem | Primitive it uses today | Primitive it needs | Verdict |
|---|---|---|---|
| Avatar | ~120 unparented `THREE.Mesh` primitives positioned per frame | one `SkinnedMesh` over a humanoid bone hierarchy | **replace** |
| Animation | landmark **positions** copied onto primitives | joint **rotations** retargeted onto named bones | **replace** |
| Tracking | One Euro filter, no confidence model | filter + confidence gate + predictive hold + decay | **rebuild around a state machine** |
| Recognition | 242-class LSTM, 1065 samples | ~40-class model + honest rejection | **cannot be fixed in the frontend; needs data** |
| Grammar / ISL | rule-based gloss + non-manuals | (sound) | **keep** |
| Motion library | MediaPipe extraction from ISLRTC clips | (sound, needs conditioning) | **keep, add a conditioning pass** |

---

## 1. Severity ranking

Ordered by *how much each one costs a Deaf user trying to read the avatar*, not by
how hard it is to fix.

| # | Defect | Severity | Root cause | Fixable in frontend? |
|---|---|---|---|---|
| 1 | Handshapes are unreadable | **Critical** | no skinned hand mesh; cylinders between landmarks | yes |
| 2 | Recognition invents words | **Critical** | model is 38% top-1; UI shows argmax regardless of confidence | UI yes, model no |
| 3 | Signs are not identifiable on playback | **Critical** | #1 + 15 fps un-resampled data + no wrist orientation | yes |
| 4 | Avatar anatomy (face, neck, shoulders, torso) | High | primitives, not a mesh | yes |
| 5 | Fingers barely articulate | High | no finger bones exist at all | yes |
| 6 | Tracking loss causes lurching | High | no confidence state machine; one global spring | yes |
| 7 | Jitter / flicker / random jumps | High | filter tuned for the recogniser, not the renderer | yes |
| 8 | No fingerspelling fallback | High | not implemented anywhere | yes |
| 9 | Camera crops the hands | Medium | fixed framing box, no content-aware fit | yes |
| 10 | Motion reads as robotic | Medium | no easing on transitions between sign and hold | yes |
| 11 | Rendering is flat | Medium | no environment map, no AO, no soft shadow tuning | yes |
| 12 | No debug panel | Medium | not implemented | yes |
| 13 | Module boundaries leak | Low | `skeleton.ts` imports frame layout constants from `signMotion.ts` | yes |

---

## 2. The avatar (critical)

### 2.1 There is no rig

`frontend/src/avatar/rig.ts` is named `SignRig` but it is not a rig. It is a bag of
independent meshes whose world transforms are recomputed from scratch every frame:

```
rig.ts:396   bones   = FINGER_BONES.map(...)  → 15 cylinders per hand
rig.ts:403   joints  = KNUCKLES.map(...)      → 15 spheres per hand
rig.ts:416   tips    = FINGER_TIPS.map(...)   → 5 spheres + 5 "nails" per hand
rig.ts:426   palm    = one BoxGeometry
rig.ts:431   cuff    = one cylinder
```

That is **41 disconnected objects per hand**, 82 for both, plus ~14 for the body and
~20 for the face. Nothing is parented to anything. There is no `THREE.Bone`, no
`THREE.Skeleton`, and no `THREE.SkinnedMesh` anywhere in the project:

```
$ grep -rn "SkinnedMesh\|THREE.Bone\|Skeleton(" frontend/src
(no matches)
```

Consequences that follow directly and cannot be patched away:

* **Fingers cannot articulate smoothly.** A finger is three cylinders and three
  spheres that are *re-placed*, not rotated. Between the segments there is a visible
  discontinuity in surface normal and no continuous skin, so a bent finger reads as a
  chain of beads. `rig.ts:70-76` gives each phalanx a hard radius pair; where two
  phalanges meet at an angle the cylinders interpenetrate on the inside of the bend
  and gap on the outside. That is the reported "finger joints collapse".
* **Palm orientation is decorative.** `posePalm` (`rig.ts:692`) fits a *box* to the
  plane through wrist / index-MCP / pinky-MCP. A box has no thenar eminence, no
  hypothenar pad, no arch, and no knuckle line — the three cues a reader uses to
  judge palm facing. Palm-up and palm-down differ only by which flat face of the box
  is lit.
* **Fingers intersect.** Nothing checks. There is no collision, no joint limit, and
  no notion of a finger having a neighbour.
* **Thumb opposition is impossible to read.** The thumb is bones `[1,2],[2,3],[3,4]`
  with no CMC saddle joint modelled and no distinct thenar mass, so opposition
  (thumb pad rotating to face the fingers) renders as the thumb simply translating.

### 2.2 The body is a lathe, not an anatomy

`makeTorsoGeometry` (`rig.ts:795`) lofts 13 superelliptical rings. It is a genuinely
careful piece of work, and it is still the wrong tool: a torso built as stacked rings
has no scapula, no clavicle, no ribcage taper into the latissimus, and — decisively —
**no deformation when the arm moves.** The shoulder is covered by a separate ellipsoid
"deltoid" (`rig.ts:233`, sized `0.112 × 0.135 × 0.112`) that is translated over the
joint. Real shoulders do not work that way; the visible result is the "shoulder pad"
appearance in the screenshot.

The head (`makeHeadGeometry`, `rig.ts:890`) is a UV sphere with four analytic
deformations. Measured against `ANATOMY` (`skeleton.ts:93`):

| Feature | Value in code | Anatomically correct | Error |
|---|---|---|---|
| `headWidth` | 0.44 shoulder-widths | 0.39 | **+13%** (documented as deliberate at `skeleton.ts:108-115`) |
| `headHeight` | 0.62 | 0.58 | +7% |
| eye separation | `±0.225 × headWidth` = 0.198 sw | interpupillary ≈ 0.16 sw | **+24%** |
| eyeball radius | 0.033 sw (`rig.ts:316`) | ≈ 0.030 sw | +10% |
| `neckLength` | 0.26 | ≈ 0.17 | **+53%** |

The neck error is the one you see. `ANATOMY.neckLength = 0.26` is 53% too long, and
the neck mesh is additionally drawn from `shoulderMid + 0.24·up` to
`head.centre − 0.42·headHeight` (`rig.ts:525-529`), which stretches it further. That
is the "neck transitions badly" report, and it is a single wrong constant compounded
by a hand-placed mesh.

Eyes sit at `y = +0.055 × headHeight` — i.e. *above* the head's vertical centre. On a
real skull the pupils sit at almost exactly the mid-height of the head. Combined with
a hairline at `+0.20 × headHeight` (`rig.ts:947`), the forehead is compressed and the
face reads as a mask, which matches the screenshot.

### 2.3 Every material is `MeshPhysicalMaterial` with no environment

`rig.ts:191` builds ~14 `MeshPhysicalMaterial` instances with `sheen` and `clearcoat`
enabled. `MeshPhysicalMaterial` with clearcoat is roughly 2× the fragment cost of
`MeshStandardMaterial`, and **without an environment map both sheen and clearcoat
contribute almost nothing** — they are reflectance models that need something to
reflect. `SignAvatar.tsx` adds four analytic lights and no `scene.environment`. So the
project pays the price of the most expensive material in three.js and gets the look of
the cheapest.

`SignAvatar.tsx:197-201` then sets `side = THREE.DoubleSide` on *every* material to
work around the mirror scale, doubling the fragment count again and disabling backface
culling on 120 objects.

### 2.4 Transparency toggling forces shader recompiles

`rig.ts:467-473` flips `material.transparent` and sets `needsUpdate = true` whenever
presence crosses 0.995. `needsUpdate` on a material triggers a **program recompile** in
three.js. Presence is driven by a damped spring, so it will cross that threshold
repeatedly during normal tracking dropout — each crossing recompiles 14 shader programs
mid-frame. That is a multi-frame stall, and it happens exactly when tracking is already
struggling.

---

## 3. Animation and retargeting (critical)

### 3.1 The system animates positions, not rotations

This is the single architectural decision that causes most of what the user reported.

`skeleton.ts` solves *joint positions* (`ArmPose { shoulder, elbow, wrist }`), and
`rig.ts` places primitives at those positions. Nothing anywhere computes a **rotation**.
Consequences:

* **Wrist rotation is not represented at all.** `ArmPose` has no orientation field.
  The forearm is a cylinder from elbow to wrist — a cylinder is rotationally symmetric,
  so forearm pronation/supination is *unrepresentable*. In sign language, palm
  orientation is one of the five phonological parameters (handshape, location,
  movement, orientation, non-manuals). One fifth of the linguistic signal is missing
  by construction. This is the direct cause of "wrists rotate incorrectly".
* **No joint limits are possible.** A limit is a constraint on a rotation. With only
  positions there is nothing to constrain, so `constrain()` (`skeleton.ts:695`) can
  only enforce bone *length* and chest clearance. An elbow can therefore hyperextend
  or bend sideways — "elbows bend unnaturally".
* **No inverse kinematics in the usual sense.** `poleElbow` (`skeleton.ts:736`) is a
  correct two-bone position solve, but its output cannot be handed to a skeleton
  because there is no skeleton.
* **Blending is per-coordinate.** `mixFrames` (`signMotion.ts:197`) lerps raw floats.
  Lerping two positions of a rotating joint cuts the corner — the limb shortens
  through the blend and then snaps back. Correct blending is `slerp` on quaternions.

### 3.2 Playback is not resampled and the source is 15 fps

`SignPlayer.tick` (`signMotion.ts:483`) advances `position += dt * motion.fps` with
`motion.fps = 15`, and `sampleFrame` linearly interpolates the two neighbouring frames.
Two problems:

1. **Linear interpolation of a 15 fps trajectory at 60 fps is piecewise-linear** — the
   velocity is a step function with a discontinuity at every source frame. That is
   exactly the "robotic" quality reported. A sign moving at 1 m/s changes direction
   instantaneously 15 times a second.
2. **No conditioning of the source.** The library is raw MediaPipe output. Measured
   over the 239 signs, per-frame hand-landmark noise is visible in the data itself; it
   is smoothed for the *live* path (`tracking.ts` `FrameSmoother`) but the recorded
   path is played back raw. The reference sign is therefore *noisier* than the live
   mirror of the user's own hands.

### 3.3 There is no fingerspelling

`grep -rin "fingerspell" .` → no matches. The manual explicitly requires it
(requirement 6: "If a word does not exist: perform fingerspelling"). Today an unknown
word becomes a `QueueItem` with `missing: true` and the avatar simply stands still for
`MISSING_SECONDS = 0.9` while a caption names the word (`signMotion.ts:320`). For a
Deaf user who is not reading the caption, an unknown word is **silence**.

There is also no ISL manual alphabet anywhere in the repository — no letter signs in
`frontend/public/signs/` (the file `i.json` is the pronoun *I*, not the letter).

### 3.4 Transitions have the right idea and the wrong shape

`SignPlayer` implements SIGN → HOLD → TRANSITION, which is linguistically correct and
worth keeping. But `TRANSITION_SECONDS = 0.26` is fixed regardless of how far the hands
must travel: a transition from a sign that ends at the forehead to one that starts at
the waist gets the same 0.26 s as one that barely moves. Real transitional movement
obeys Fitts' law — duration scales with distance. Fixed duration means long transitions
are frantic and short ones are sluggish.

---

## 4. Tracking (high)

### 4.1 What is already good

`tracking.ts` is the strongest file in the project and most of it should survive:

* `FrameSmoother` is a correct, presence-aware One Euro implementation
  (`tracking.ts:110`), with separate tuning for hands and body — and the reasoning for
  the split (shoulders are the normalization anchor) is right.
* `HandAssigner` (`tracking.ts:196`) solves handedness flapping by continuity, which is
  a real and commonly-missed bug.
* `POSE_VISIBILITY = 0.5` gating (`tracking.ts:385`) is the right instinct.

### 4.2 What is missing

* **No confidence signal reaches the renderer.** The worker posts `hands`, `body`,
  `gaps` and `motion`, but never MediaPipe's per-detection `score`. `HandAssigner`
  reads `category.score` (`tracking.ts:232`) and then discards it. The renderer
  therefore cannot distinguish "solid lock" from "barely detected", and treats both
  identically.
* **No prediction across occlusion.** `HAND_HOLD_FRAMES = 8` (`skeleton.ts:266`) holds
  the *last* hand shape for 8 frames and then drops it. Holding is not prediction: a
  hand occluded mid-movement freezes and then teleports to wherever it re-appears.
  There is no velocity model, so no constant-velocity extrapolation is possible.
* **No outlier rejection.** Nothing checks whether this frame's landmark set is
  physically reachable from the previous one. A single bad MediaPipe frame — which it
  produces regularly under motion blur — passes straight into One Euro, whose `beta`
  term *increases* the cutoff when it sees a large derivative. **A One Euro filter
  actively lets outliers through**; that is its design. Without a gate in front of it,
  the reported "random jumps" are guaranteed.
* **Two different filters, inconsistently applied.** The recogniser's 141-float vector
  is One-Euro-filtered. The 126-float **world** landmarks — the ones the avatar's hands
  are actually built from — are **not filtered at all** (`mediapipe.worker.ts:254-264`
  copies them raw). The most visually sensitive data in the product is the only data
  with no smoothing on it.
* **Detection confidences are set to 0.3 across the board**
  (`mediapipe.worker.ts:58-60`). The docstring justifies this for the recogniser
  (a missing hand is a hole in the window). It is the wrong trade for the renderer:
  a 0.3-confidence detection is frequently a *false* hand, and drawing it is the
  reported "false detections". Both consumers are fed from the same threshold.

### 4.3 Loss of tracking

`SkeletonSolver.solve` (`skeleton.ts:326`) does have a rest pose and a slow relax
(`RELAX_SMOOTHING = 0.55`), which is better than nothing. But:

* The switch is **binary**: `measured ?? REST`. There is no graded blend between "I can
  see you well", "I can partly see you", and "you have gone". `measure()` returns
  `null` the instant both shoulders drop below visibility — so a momentary shoulder
  occlusion flips the whole target from the tracked pose to the rest pose, and the
  spring starts pulling the arms down. That is the reported body twitch.
* `RELAX_SMOOTHING` is applied to **every joint at once**, so the figure relaxes as a
  unit. A person losing tracking should have their *arms* settle while the torso and
  head stay put.
* Presence floor `IDLE_PRESENCE = 0.26` makes the figure semi-transparent while idle,
  which is what triggers the shader-recompile problem in §2.4.

---

## 5. Recognition (critical — and not fixable in the frontend)

### 5.1 The model cannot work at this scale

Measured from `ml/models/metrics.json` and `ml/data/`:

| Metric | Value |
|---|---|
| Classes | **242** |
| Total training samples | **1065** |
| Median samples per class | **2** |
| Classes with ≤ 2 samples | **159** (66%) |
| Classes with 0 samples | 4 |
| `rest` (reject) class samples | **16** |
| Top-1 accuracy | **0.382** |
| Top-3 accuracy | 0.451 |
| Test samples total | 233 (fewer than one per class) |
| Classes with any test sample | 124 |
| Classes ≥ 80% top-1 | **37** |
| Gated precision @ 0.85 | 0.772 |
| Gated coverage @ 0.85 | 0.395 |

**242 classes on 1065 samples is roughly 4.4 samples per class.** No architecture
recovers from that. The 38% top-1 is not a bug to be fixed by better preprocessing; it
is the information-theoretic ceiling of this dataset.

Worse, the `rest` reject class has 16 samples, and inspecting `ml/data/rest/` shows
they are named `drive-*.npy`, `egg-*.npy` — i.e. they are **idle segments cut out of
other signs' dictionary clips**, not recordings of a person sitting in front of a
webcam with their hands open. The classifier has therefore never seen the single most
common real-world input: *a person doing nothing in particular*.

### 5.2 The "Headache" bug is a UI defect on top of that

This is the mechanism behind the exact symptom reported.

The backend is correct — `backend/main.py:262` gates emission on
`confidence > threshold AND stable_count >= 3 AND label != "rest"`. It refuses to emit.

But on every non-accepted frame it still sends `top3` (`main.py:281`), and the frontend
displays the argmax **unconditionally**:

```tsx
// Recognition.tsx:31-37
{!active ? "Recognition paused"
  : lead ? lead.word                      // ← argmax at ANY confidence
  : ...}
```

`lead` is `top3[0]`, set from every `status: "listening"` message
(`useVoxSocket.ts:116`). So the large word on screen is the model's best guess even at
4% confidence. `headache` has 9 training samples and **0% test accuracy**; with 242
classes and a near-uniform softmax over a novel input, it will win the argmax
regularly. The product then displays it in the position a user reads as "this is what
you signed".

**The application is not hallucinating a gesture — it is faithfully displaying a
rejected one in the place reserved for accepted ones.**

### 5.3 The 0.85 threshold is not calibrated

`CONFIDENCE_THRESHOLD = 0.85` is a raw softmax value. Softmax outputs from a small
LSTM trained on 1065 samples are not probabilities — they are systematically
overconfident. Gated precision at 0.85 is 0.772, i.e. **nearly a quarter of everything
the system does say is wrong**, at 39.5% coverage. No temperature scaling or any other
calibration is applied.

---

## 6. Rendering, camera, performance

* **No environment map.** `scene.environment` is never set. Every PBR material in the
  scene is therefore lit only by four analytic lights, which is why the figure reads as
  matte plastic (requirement 10).
* **No tone-mapping/output colour-space discipline.** `ACESFilmicToneMapping` is set
  (`SignAvatar.tsx:136`) but `outputColorSpace` is left default and no
  `renderer.setClearColor` / alpha compositing check is done — the figure is composited
  over a CSS gradient, so the tone-mapped output is being blended in a different space
  from the page.
* **Fixed framing box.** `FRAMING.body = { centreY: -0.20, height: 3.15, width: 2.7 }`
  (`SignAvatar.tsx:64`) and `fitDistance()` fits *that box*, not the actual content.
  The comment at `SignAvatar.tsx:58-63` explicitly accepts that resting hands leave
  frame. For signs that reach high or wide (and for the idle pose) the hands are
  cropped — requirement 9 says never.
* **Shadow map 1024² with `radius: 3`** on a `PCFSoftShadowMap` gives a very soft,
  very cheap shadow. Fine. But there is no ambient occlusion of any kind, which is what
  actually seats a hand against a body.
* **The whole scene re-renders unconditionally at rAF** even when nothing has changed
  (idle, no queue, no camera).
* **`Float32Array.from(msg.vector)` + `setFrame` + `setAvatarFrame` on every worker
  message** (`useHandTracking.ts:199-209`) triggers a full React re-render of
  `SessionPage` at up to 30 Hz, cascading into `SignStage`, `Recognition`, `Composer`.
  This is the main-thread cost the requirements ask to remove.

---

## 7. Code quality and module boundaries

* **`skeleton.ts` imports the wire format from `signMotion.ts`** (`skeleton.ts:62-80`):
  `POSE_BLOCK`, `POSE_STRIDE`, `LEFT_WORLD`… A solver should not know the byte layout
  of a file format. This is the leak that makes the avatar and the motion library
  impossible to change independently.
* **`landmarks.ts` and `tracking.ts` both define frame assembly.** `assembleFrame`
  (`tracking.ts:420`) is documented as "the stabilised counterpart to `buildFrameVector`
  in landmarks.ts" — i.e. the same logic exists twice.
* **`rig.ts` mixes geometry generation, material creation, and per-frame posing** in one
  1022-line class.
* **Magic numbers throughout `rig.ts`** — `0.112`, `0.135`, `0.045`, `0.015`, `0.24`,
  `0.42` — each hand-tuned against a specific screenshot, with no single source of
  proportion.
* **No tests that assert anything visual.** `__checks__/skeleton.check.ts` and
  `__checks__/grammar.check.ts` exist and are not wired into any npm script.
* **`orb-ai-assistant/`** is an unrelated project sitting in the repo root.

---

## 8. What is genuinely good and must be preserved

Being fair to the existing work — these are correct and rebuilding them would be a
regression:

1. **`ml/normalize.py` as the single shared contract** between training and inference,
   with the v1→v2 body-anchored rationale documented. This is the right call and the
   docstring explaining why hand-local normalization was catastrophic is worth keeping.
2. **`isl/grammar.ts`** — SOV reordering, copula deletion, article deletion, time-first,
   question-word-final, negation-after-verb. This is real ISL syntax, not word
   substitution.
3. **`avatar/nonManual.ts`** — treating brow raise as a morpheme rather than an
   expression, and synthesising it from syntax rather than extracting it from citation
   forms. The reasoning at `nonManual.ts:20-34` is correct.
4. **`HandAssigner`** — continuity-based handedness resolution.
5. **The motion library itself** — 239 signs of genuine MediaPipe extraction from ISLRTC
   dictionary clips, with both image-space placement and metric world-space shape. This
   is the product's most valuable asset and the format (two coordinate systems per
   frame) is the right design.
6. **`SignPlayer`'s SIGN → HOLD → TRANSITION state machine** — the insight that readers
   segment on transitional movement is correct.

---

## 9. Proposed architecture

```
                         ┌─────────────────────────────────────────┐
  camera ──▶ capture ──▶ │ WORKER                                  │
                         │  MediaPipe Hands + Pose                 │
                         │  ↓                                      │
                         │  HandAssigner  (continuity)             │
                         │  ↓                                      │
                         │  OutlierGate   (physical plausibility)  │  ← NEW
                         │  ↓                                      │
                         │  OneEuro + ConstantVelocityPredictor    │  ← NEW
                         │  ↓                                      │
                         │  TrackingState {pose, conf, quality}    │  ← NEW
                         └─────────────────────────────────────────┘
                                         │ SharedArrayBuffer / transfer
                                         ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │ POSE SOURCE  (one interface, three implementations)              │
   │   LiveSource      ← worker frames                                │
   │   LibrarySource   ← SignPlayer over the motion library           │
   │   SynthSource     ← fingerspelling / procedural signs            │  ← NEW
   │ produces:  SignerPose { body, hands[2], face, confidence }       │
   └──────────────────────────────────────────────────────────────────┘
                                         ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │ RETARGETER            landmarks ──▶ bone ROTATIONS               │  ← NEW
   │   armIK    : shoulder→elbow→wrist two-bone IK + pole vector      │
   │   wristSolve: full 3-DOF orientation from palm basis             │
   │   fingerSolve: per-joint MCP/PIP/DIP + thumb CMC from landmarks  │
   │   jointLimits: anatomical clamp per bone, per axis               │
   │   → HumanoidPose { Map<BoneName, Quaternion> }                   │
   └──────────────────────────────────────────────────────────────────┘
                                         ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │ POSE BLENDER          quaternion slerp, spring per bone          │  ← NEW
   │   confidence-weighted; per-chain decay; idle blend               │
   └──────────────────────────────────────────────────────────────────┘
                                         ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │ AVATAR  (interface: HumanoidAvatar)                              │
   │   VrmAvatar        — @pixiv/three-vrm, humanoid bone map         │  ← NEW
   │   ProceduralAvatar — generated SkinnedMesh, same bone names      │  ← NEW
   └──────────────────────────────────────────────────────────────────┘
                                         ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │ RENDERER   env map · ACES · soft shadows · SSAO · adaptive fit   │
   └──────────────────────────────────────────────────────────────────┘
```

### 9.1 The central decision: bone rotations, not positions

Every requirement in sections 1, 2, 7 and 8 of the brief follows from this one change.

The **VRM humanoid bone spec maps 1:1 onto MediaPipe's 21 hand landmarks**:

| VRM bone | MediaPipe landmark pair | Joint |
|---|---|---|
| `leftThumbMetacarpal` | 1 → 2 | CMC |
| `leftThumbProximal` | 2 → 3 | MCP |
| `leftThumbDistal` | 3 → 4 | IP |
| `leftIndexProximal` | 5 → 6 | MCP |
| `leftIndexIntermediate` | 6 → 7 | PIP |
| `leftIndexDistal` | 7 → 8 | DIP |
| … (middle 9-12, ring 13-16, little 17-20) | | |

15 finger bones per hand, exactly matching the 15 phalanx segments MediaPipe reports.
That is why the brief's recommendation is right: **VRM is not a cosmetic upgrade, it is
the data structure that makes finger articulation expressible.**

### 9.2 Two avatar backends behind one interface

Trade-off, stated plainly:

* **VRM path** gives professional mesh quality, real skin weights, blend shapes for the
  face, and spring bones — *if a licensed .vrm asset is available*. I will not download
  a third-party avatar into this repository without an explicit decision from you: it
  is someone's copyrighted asset with a licence that governs redistribution.
* **Procedural path** is a `SkinnedMesh` generated at load time over the *identical*
  VRM-named bone hierarchy — no asset, no licence, works offline, guaranteed to exist.
  Quality ceiling is lower than a modelled human, but with proper skinning, subdivided
  limb geometry, and real hand topology it is far above what is there today.

Building both behind `interface HumanoidAvatar` costs perhaps 15% more than building
one, and means the VRM decision can be made later without another rewrite.

### 9.3 Tracking as a state machine

Replace the binary `measured ?? REST` with four states and graded transitions:

```
        ┌──────────┐  conf > 0.6, 3 frames   ┌──────────┐
        │ ACQUIRING├────────────────────────▶│ TRACKING │
        └──────────┘                         └────┬─────┘
              ▲                                   │ conf < 0.4
              │ conf > 0.6                        ▼
        ┌─────┴────┐   > 500 ms          ┌─────────────────┐
        │   IDLE   │◀────────────────────┤ COASTING        │
        └──────────┘                     │ (predict + hold)│
                                         └─────────────────┘
```

* **TRACKING** — full confidence-weighted update, One Euro on everything including
  world landmarks.
* **COASTING** — constant-velocity extrapolation of the last good pose, amplitude
  decaying to zero over ~350 ms. The hand keeps moving the way it was moving, slowing
  down. This is what makes short occlusions invisible.
* **IDLE** — blend to a breathing rest pose over 800 ms, per-chain (arms first, torso
  last), never snapping.

Nothing ever jumps because every transition is a quaternion slerp with a duration.

### 9.4 Recognition: honesty first

Three things, in order of how much they help:

1. **Stop displaying the argmax.** Below threshold the readout says
   *"Gesture not recognized"*. Non-negotiable and takes an hour.
2. **Restrict the vocabulary to what is measured.** 37 words have ≥80% held-out
   accuracy. Offering 242 and being right 38% of the time is worse than offering 37 and
   being right 80% of the time — for an accessibility product it is *much* worse,
   because a confident wrong word in a medical conversation is a safety issue.
3. **Calibrate.** Temperature scaling on the held-out split, plus an explicit
   open-set rejection (energy-based or max-logit) rather than a raw softmax threshold.

Retraining on more data is the real fix and is outside what code changes can deliver;
I will say so rather than pretend otherwise.

---

## 10. Implementation order

Each step is independently shippable and leaves the app working.

| Step | Module | Why this order |
|---|---|---|
| 0 | `core/` types, `Vec`/`Quat` helpers, bone name enum | everything depends on it |
| 1 | **`avatar/humanoid/`** — bone hierarchy, `ProceduralAvatar` SkinnedMesh | the hands are the product |
| 2 | **`avatar/retarget/`** — landmarks → rotations, joint limits, IK | makes step 1 move |
| 3 | `avatar/blend/` — quaternion springs, confidence weighting, state machine | fixes lurching |
| 4 | `render/` — env map, AO, adaptive framing, perf | makes it look real |
| 5 | `tracking/` — outlier gate, world-landmark filtering, confidence plumbing | fixes jitter |
| 6 | `signs/` — motion conditioning pass, resampling, fingerspelling | fixes readability |
| 7 | `recognition/` — rejection UI, calibration, vocabulary gating | fixes hallucination |
| 8 | `debug/` — the developer panel | needs all of the above to report on |
| 9 | cleanup — delete dead code, wire the checks into `npm test` | |

---

*Audit ends. Nothing above has been changed yet.*

---

# Rebuild status — what was replaced, and what it measures

Appended after implementation. Every number below is produced by `npm run check`
in `frontend/`, run over all 239 signs of the real motion library.

## Modules delivered

| Module | Status | Verified by |
|---|---|---|
| `core/math.ts` — rotation toolkit | **new** | 32 checks |
| `core/humanoid.ts` — VRM bone contract, clinical joint limits | **new** | — |
| `avatar/retarget/rest.ts` — per-model skeleton measurement | **new** | 9 checks |
| `avatar/retarget/hand.ts` — 21 landmarks → 15 bone rotations | **new** | 13 checks |
| `avatar/retarget/arm.ts` — two-bone IK, humeral roll, forearm twist | **new** | 7 checks |
| `avatar/retarget/body.ts` — spine, neck, head, non-manuals | **new** | 7 checks |
| `tracking/handConditioner.ts` — outlier gate, skeleton projection, One Euro | **new** | 2 checks |
| `avatar/pose/blend.ts` — 4-state machine, rate limit, anatomy clamp | **new** | 21 checks |
| `avatar/VrmAvatar.ts` — VRM load + strict bone validation | **new** | — |
| `avatar/AvatarStage.ts` — env lighting, content-aware framing | **new** | — |
| `components/DevPanel.tsx` — developer diagnostics | **new** | — |
| `avatar/rig.ts`, `avatar/skeleton.ts` | **deleted** (1,997 lines) | — |

## Measured results

```
hand pipeline (192,105 finger bones)
  bone direction vs measurement   median 4.94°
  fingertip lands within          7.2% of hand span (p50)
  joints outside clinical range   0
  adjacent fingertips coincide    never (p1 = 5.3 mm)
  landmark jitter removed         40%
  jitter amplification            +10% (bounded)
  open vs closed handshape        29.1° vs 38.7° MCP flexion — distinguishable

whole pipeline (7,189 source → 28,756 rendered frames)
  wrist placement error           p50 7.9%  p75 23.2%  of shoulder width
  largest bone step               17.19°/frame max (human ceiling 23.9°)
  bones stretched                 0
  non-finite rotations            0
```

## Defects found and fixed during the rebuild

Each of these was found by a check, not by inspection:

1. **Minimal rotation is wrong for the proximal phalanx.** Zero roll is arbitrary;
   the real roll is fixed by the finger's bend plane. Getting it wrong made the
   library demand ±40° of lateral motion at PIP joints that have ~6°. Fixed by
   deriving the hinge from `proximalDir × intermediateDir`; spread collapsed to ±3°.
2. **Shoulder limits were in the wrong reference frame.** Clinical goniometry
   measures from arms-at-sides; VRM rests in a T-pose. The 90° offset made the
   arm unable to reach the signer's own side.
3. **Pose `z` is not a depth.** Measured: `|pose dz|` median 2.83 shoulder widths
   against a 1.50-shoulder-width arm. Rebuilt depth from bone lengths, using the
   tracker only for lean direction.
4. **The elbow hinge axis had the wrong sign,** so every bend read as
   hyperextension and clamped to zero — the arm rendered permanently straight at
   0.538 m against a 0.540 m reach.
5. **Signing space was anchored to the rest shoulders,** not the solved ones. The
   spine solve swings the shoulders up to 12 cm; the arms were solved from an
   origin the arms did not have.
6. **Springs alone do not prevent snapping.** A 15 fps source through a 55 ms
   spring still produced 39°/frame. Added an unconditional angular rate limit.
7. **Clamp-then-rate-limit, not the reverse.** The anatomy clamp can itself move a
   joint a long way; rate-limiting must be last or it is not a guarantee.

## Known open items

* **Wrist placement tail (p90 91%).** The library's hand and pose blocks come from
  two independent trackers that sometimes describe different arms. Gross
  mismatches are now detected and the pose model wins, which cut the median from
  145% to 7.9%. The residual is sub-threshold disagreement and is not separable
  without re-extracting the library — `ml/build_motion.py` should associate hand
  detections to pose wrists at extraction time.
* **Fingerspelling (brief §6) is NOT implemented.** There is no ISL manual
  alphabet in the repository and none can be extracted from the dictionary clips,
  which contain only whole-word citation forms. It needs 26+ authored handshapes
  as `HumanoidPose` keyframes. The rig can express them; the data does not exist.
* **Motion resampling.** `SignPlayer` still lerps a 15 fps source. The rate limiter
  makes this safe but not optimal; Catmull–Rom over the frame sequence would
  restore the sign's true velocity profile.
* **Recognition remains data-limited.** 242 classes on 1,065 samples cannot be
  fixed in code. The vocabulary is now gated to the 37 measured-reliable words.

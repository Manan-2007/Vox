# Vox — demo script & shot list

Vocabulary: `bye come eat hello help please sorry thanks`. The model and all
eight clips ship with the repo — nothing to supply.

**The one trick that makes the demo land:** before signing a word live, type it
into the Speech → ISL panel and watch its clip — then copy that exact form.
The model was trained on those clips.

## Setup (before recording / presenting)

1. `./start.sh` — wait for "Backend connected" in the top bar.
2. Even, front-facing light; plain background; sit ~1 m from the camera.
3. Settings → threshold 85%; auto-speak on. Close the drawer.
4. Do one throwaway sign to confirm the skeleton overlay tracks.

## Script (~90 seconds)

| # | Action | What to say / point at |
|---|--------|------------------------|
| 1 | Landing page, 5 s | "Vox — a two-way ISL interpreter. Everything runs locally." |
| 2 | Start a session | Point at the skeleton overlay and FPS chip: "hand tracking in a Web Worker, 15 FPS." |
| 3 | Sign `hello` (copy the clip's form), hold ~2 s | Word appears; point at the confidence bar crossing the 85% marker. |
| 4 | Sign `please`, then `help` | Sentence builds in the transcript bubble. |
| 5 | Pause ~4 s | Sentence commits and is **spoken aloud** — "signed · spoken aloud" tag. |
| 6 | Other person clicks 🎤 and says "hello, thanks for your help" | Bubble lands on the right as "heard"; real ISL clips for hello/thanks/help play in sequence. |
| 7 | Sign `bye` → Speak sentence | Close the loop. |
| 8 | Pull the network cable / kill the backend, 5 s | Badge flips to "Backend closed…", then auto-reconnects when restarted. |

If a wrong word lands mid-demo: click **Undo word**, keep going — mention the
transition-artifact limitation out loud; honesty plays better than hiding it.

## Shot list (recorded video)

1. **Wide** — both people + screen visible, one full exchange (steps 3–7).
2. **Screen capture** — the same exchange, full-screen UI.
3. **Close-up** — hands + skeleton overlay while signing (left panel).
4. **Cutaway** — `train.py` output: confusion matrix PNG + val accuracy.
5. **Cutaway** — `ml/collect.py` recording a sample (shows how data was made).
6. End card: repo layout + "no video leaves the machine" line.

Record 2 and 3 in the same take (screen-record while filming over-shoulder);
sync in the edit. Keep the recognition segments unedited — no cuts between
sign and word appearing, or it reads as faked.

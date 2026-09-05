# Put your signer here

    signer.vrm

Vox ships without a humanoid on purpose: a VRM is someone's copyrighted asset
with terms governing redistribution, so the choice of signer is yours.

## Requirements

* VRM 0.x or VRM 1.0 (a plain .glb will be refused — it has no humanoid map)
* A complete humanoid skeleton
* **All fifteen finger bones per hand.** This is the one that gets missed. Many
  models export with a simplified hand, and such a model loads perfectly and then
  signs nothing — the hands stay flat while everything else moves. Vox refuses
  those at load and names the missing bones, rather than pretending to work.

## Pointing somewhere else

    VITE_VOX_AVATAR=/avatar/my-other-signer.vrm npm run dev --prefix frontend

## Checking a model before committing to it

Load it and open the developer panel (Ctrl/Cmd + Alt + D, or the ⚙ on the stage).
If the model was accepted, the panel reports its triangle count and the tracking
state of all six channels. If it was refused, the stage names the missing bones.

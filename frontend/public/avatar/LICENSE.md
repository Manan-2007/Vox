# example-polydancer.vrm  (NOT the default signer)

**Polydancer**, by PolygonalMind — from the 100 Avatars collection.

    Licence      CC0 1.0 Universal (public domain dedication)
    Commercial   allowed
    Source       https://arweave.net/jPOg-G0MPH55ZQmamFhT9f8cHn-hjeAQ0mRO5gWeKMQ
    Registry     https://github.com/ToxSam/open-source-avatars
    Format       VRM 0.0, 1.2 MB, 1 mesh, 52 humanoid bones

CC0 is asserted in the file's own VRM metadata (`licenseName: CC0`,
`commercialUssageName: Allow`), not only by the registry that hosts it — which is
why this one was chosen over the VRoid sample avatars, whose licensing is
contradictory between Pixiv's own FAQ and their VRoid Hub listing.

Verified with:

    node scripts/inspect-vrm.mjs public/avatar/signer.vrm

    FINGER BONES  30 / 30
    ✓ USABLE

Replacing it is a file swap — run the inspector on any candidate first. A model
with an incomplete hand rig loads perfectly and signs nothing, so Vox refuses
those at load rather than showing flat hands.


## Why this is not `signer.vrm`

It is technically perfect — VRM 0.0, 30/30 finger bones, CC0 asserted in the
file's own metadata — and aesthetically wrong for this product. It is a stylised
character from an NFT art collection with a grinning skull for a face. The brief
for Vox is that the signer "should feel like a trained interpreter", and a
grotesque cartoon undermines the credibility of a tool Deaf users are asked to
rely on in medical conversations.

It is kept because it is genuinely useful as a TEST FIXTURE: it exercises the
VRM 0.x path, including the `rotateVRM0` reference-space handling that a VRM 1.0
model would not. To load it:

    VITE_VOX_AVATAR=/avatar/example-polydancer.vrm npm run dev --prefix frontend

The default is the built-in generated signer, which is plain and neutral. Replace
it by putting a better humanoid at `signer.vrm` — run the inspector first.

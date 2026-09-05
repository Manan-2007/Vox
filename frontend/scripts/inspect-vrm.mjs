/**
 * Tell me whether this .vrm can actually sign.
 *
 *     node scripts/inspect-vrm.mjs public/avatar/signer.vrm
 *
 * A VRM that is missing its finger bones LOADS PERFECTLY and signs nothing —
 * the hands stay flat while everything else moves. That is the worst failure
 * mode this product has, because it looks like a subtle animation bug and is
 * actually total loss of meaning. The app refuses such models at runtime; this
 * script answers the same question before you commit to a download.
 *
 * It reads the glTF JSON chunk out of the binary container directly rather than
 * going through a loader, so it needs no browser, no three.js and no GPU.
 */

import { readFileSync } from "node:fs";

/* The 15 finger bones per hand that MediaPipe's 21 landmarks map onto, plus the
   arm and spine bones the rig drives. Names are VRM 1.0's; 0.x uses the same
   ones except for the thumb, which is handled below. */
const FINGERS = ["Thumb", "Index", "Middle", "Ring", "Little"];
const SEGMENTS = {
  Thumb: ["Metacarpal", "Proximal", "Distal"],
  other: ["Proximal", "Intermediate", "Distal"],
};

/**
 * VRM 0.x named the thumb's segments differently, and a 0.x model is otherwise
 * perfectly usable — three-vrm migrates it on load. Accepting both spellings
 * here keeps the script from rejecting a model the app would happily take.
 */
const THUMB_ALIASES = {
  leftThumbMetacarpal: "leftThumbProximal",
  leftThumbProximal: "leftThumbIntermediate",
  leftThumbDistal: "leftThumbDistal",
  rightThumbMetacarpal: "rightThumbProximal",
  rightThumbProximal: "rightThumbIntermediate",
  rightThumbDistal: "rightThumbDistal",
};

function requiredBones() {
  const bones = [
    "hips", "spine", "head", "neck",
    "leftUpperArm", "leftLowerArm", "leftHand",
    "rightUpperArm", "rightLowerArm", "rightHand",
  ];
  for (const side of ["left", "right"]) {
    for (const finger of FINGERS) {
      const segments = finger === "Thumb" ? SEGMENTS.Thumb : SEGMENTS.other;
      for (const segment of segments) bones.push(`${side}${finger}${segment}`);
    }
  }
  return bones;
}

/** Pull the JSON chunk out of a binary glTF container. */
function readGltfJson(path) {
  const buffer = readFileSync(path);
  if (buffer.length < 12) throw new Error("file is too small to be a GLB");
  const magic = buffer.readUInt32LE(0);
  if (magic !== 0x46546c67) {
    throw new Error("not a binary glTF — the magic bytes are not 'glTF'");
  }
  let offset = 12;
  while (offset + 8 <= buffer.length) {
    const chunkLength = buffer.readUInt32LE(offset);
    const chunkType = buffer.readUInt32LE(offset + 4);
    const start = offset + 8;
    if (chunkType === 0x4e4f534a) {
      return JSON.parse(buffer.subarray(start, start + chunkLength).toString("utf8"));
    }
    offset = start + chunkLength;
  }
  throw new Error("no JSON chunk found");
}

/** The humanoid bone map, from whichever VRM version this is. */
function readHumanoid(gltf) {
  const extensions = gltf.extensions ?? {};

  if (extensions.VRMC_vrm) {
    const bones = extensions.VRMC_vrm.humanoid?.humanBones ?? {};
    return {
      version: `VRM ${extensions.VRMC_vrm.specVersion ?? "1.0"}`,
      present: new Set(Object.keys(bones)),
      meta: extensions.VRMC_vrm.meta ?? {},
    };
  }

  if (extensions.VRM) {
    const list = extensions.VRM.humanoid?.humanBones ?? [];
    return {
      version: `VRM ${extensions.VRM.specVersion ?? "0.x"}`,
      present: new Set(list.map((entry) => entry.bone).filter(Boolean)),
      meta: extensions.VRM.meta ?? {},
      legacy: true,
    };
  }

  throw new Error("this glTF carries no VRM extension — it is not a VRM");
}

/* ---------------------------------------------------------------- main --- */

const path = process.argv[2];
if (!path) {
  console.error("usage: node scripts/inspect-vrm.mjs <file.vrm>");
  process.exit(2);
}

let gltf;
let humanoid;
try {
  gltf = readGltfJson(path);
  humanoid = readHumanoid(gltf);
} catch (error) {
  console.error(`✗ ${path}\n  ${error.message}`);
  process.exit(1);
}

const has = (bone) => {
  if (humanoid.present.has(bone)) return true;
  // A 0.x model may spell the thumb the older way.
  const alias = THUMB_ALIASES[bone];
  return alias ? humanoid.present.has(alias) : false;
};

const required = requiredBones();
const missing = required.filter((bone) => !has(bone));
const fingerBones = required.filter((b) => FINGERS.some((f) => b.includes(f)));
const fingersPresent = fingerBones.filter(has).length;

const meta = humanoid.meta ?? {};
const title = meta.name ?? meta.title ?? "(untitled)";
const author = meta.authors?.join(", ") ?? meta.author ?? "(unknown)";
const licence =
  meta.licenseUrl ?? meta.licenseName ?? meta.otherLicenseUrl ?? "(unstated)";

console.log(`${path}`);
console.log(`  format        ${humanoid.version}`);
console.log(`  name          ${title}`);
console.log(`  author        ${author}`);
console.log(`  licence       ${licence}`);
if (meta.commercialUsage ?? meta.commercialUssageName) {
  console.log(`  commercial    ${meta.commercialUsage ?? meta.commercialUssageName}`);
}
if (meta.allowRedistribution !== undefined) {
  console.log(`  redistribute  ${meta.allowRedistribution}`);
}
console.log(`  humanoid      ${humanoid.present.size} bones mapped`);
console.log(`  FINGER BONES  ${fingersPresent} / ${fingerBones.length}`);
console.log(`  meshes        ${gltf.meshes?.length ?? 0}`);

if (missing.length > 0) {
  console.log(`\n✗ UNUSABLE — missing ${missing.length} bone(s) the rig needs:`);
  console.log(`  ${missing.join("\n  ")}`);
  if (missing.some((b) => FINGERS.some((f) => b.includes(f)))) {
    console.log(
      "\n  This model has an incomplete hand rig, so it cannot form handshapes.\n" +
        "  Vox will refuse it at load rather than sign with flat hands.",
    );
  }
  process.exit(1);
}

console.log("\n✓ USABLE — every bone the rig drives is present, including all 30 finger bones.");

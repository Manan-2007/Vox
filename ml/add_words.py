"""Add signs to the library, one command, any word in the ISL dictionary.

    python ml/add_words.py ambulance fever "thank you" umbrella

Resolves each word against the cached ISLRTC dictionary index, downloads the
clips, extracts the motion, and writes the sign files the avatar plays. The
frontend picks new signs up with no code change: the gloss engine treats any
word the library has as signable (see `toGloss` in frontend/src/isl/grammar.ts).

This exists because "which words does it know" should be a question with a
command as its answer, not a hard-coded list. The curated vocabulary in
ml/vocabulary.py is the *starting* set — the words the product was designed
around, and the ones the recogniser is trained on. This adds to what the avatar
can *produce*, which is a much cheaper thing to extend: producing a sign needs
one recording, recognising it needs many.

Recognition is deliberately NOT extended here. A word added this way can be
signed to you; for Vox to understand it back from you, it needs training data —
record your own with `python ml/collect.py <word>` and retrain.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ML_DIR))
from fetch_dictionary import (  # noqa: E402
    MAX_CLIPS_PER_GLOSS, build_index, channel_search, load_extra,
    match_score, save_extra,
)

VIDEOS_DIR = ML_DIR / "videos"
SIGNS_DIR = ML_DIR.parent / "frontend" / "public" / "signs"


def gloss_for(word: str) -> str:
    """The library key for a word: lowercase, letters and digits only."""
    return "".join(c for c in word.lower() if c.isalnum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("words", nargs="+", help="English words or phrases")
    parser.add_argument("--videos-dir", type=Path, default=VIDEOS_DIR)
    parser.add_argument("--no-search", action="store_true",
                        help="do not fall back to searching the ISLRTC channel")
    args = parser.parse_args()

    index = build_index() + load_extra()
    print(f"dictionary index: {len(index)} entries")

    resolved: dict[str, list[str]] = {}
    missing: list[str] = []

    for word in args.words:
        gloss = gloss_for(word)
        if not gloss:
            continue
        hits = [(v, t) for v, t in index if match_score(word, t)]

        if not hits and not args.no_search:
            print(f"  searching the channel for {word!r}…")
            found = channel_search([word])
            hits = [(v, t) for v, t in found if match_score(word, t)]
            if hits:
                save_extra(hits)

        if not hits:
            missing.append(word)
            print(f"  {word:<20} no dictionary entry")
            continue

        ids = list(dict.fromkeys(v for v, _ in hits))[:MAX_CLIPS_PER_GLOSS]
        resolved[gloss] = ids
        print(f"  {word:<20} -> {gloss}  ({len(ids)} clip(s))")

    if not resolved:
        sys.exit("\nNothing to add.")

    args.videos_dir.mkdir(parents=True, exist_ok=True)
    print(f"\ndownloading {sum(len(v) for v in resolved.values())} clip(s)")
    for gloss, ids in resolved.items():
        for vid in ids:
            target = args.videos_dir / f"{gloss}__{vid}.mp4"
            if target.exists() and target.stat().st_size > 1024:
                continue
            subprocess.run(
                [sys.executable, "-m", "yt_dlp", "-q", "--no-warnings",
                 "--js-runtimes", "node", "--remote-components", "ejs:github",
                 "-f", "mp4[height<=480]/best[height<=480]/best",
                 "-o", str(args.videos_dir / f"{gloss}__{vid}.%(ext)s"),
                 f"https://www.youtube.com/watch?v={vid}"],
                capture_output=True, text=True,
            )

    print("\nextracting motion")
    result = subprocess.run(
        [sys.executable, str(ML_DIR / "build_motion.py"),
         "--videos-dir", str(args.videos_dir), "--only", *resolved],
        text=True,
    )
    if result.returncode != 0:
        sys.exit("motion extraction failed")

    # build_motion only writes glosses it knows about; anything added here that
    # is not in vocabulary.py needs its manifest entry merging in by hand.
    merge_manifest(resolved)

    print(f"\nAdded {len(resolved)} sign(s). They are signable immediately — "
          f"reload the app.")
    if missing:
        print(f"No dictionary entry for: {', '.join(missing)}")


def merge_manifest(resolved: dict[str, list[str]]) -> None:
    """Ensure every newly extracted sign appears in the manifest."""
    manifest_path = SIGNS_DIR / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    signs = manifest.setdefault("signs", {})

    changed = False
    for gloss in resolved:
        sign_path = SIGNS_DIR / f"{gloss}.json"
        if not sign_path.exists() or gloss in signs:
            continue
        sign = json.loads(sign_path.read_text())
        frames = sign.get("frames", [])
        signs[gloss] = {
            "english": sign.get("english", gloss),
            "pos": sign.get("pos", "noun"),
            "frames": len(frames),
            "seconds": round(len(frames) / max(1, sign.get("fps", 15)), 2),
            "twoHanded": False,
            "quality": 0.0,
        }
        changed = True

    if changed:
        manifest["signs"] = dict(sorted(signs.items()))
        manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

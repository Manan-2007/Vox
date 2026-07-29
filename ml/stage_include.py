"""Stage INCLUDE dataset videos for ml/extract.py.

INCLUDE ships as `<Category>/<NN>. <Word>/<MVI_xxxx>.MOV`. extract.py wants
`<label>__<id>.<ext>` in one directory, so this makes symlinks with the label
derived from the folder name.

    python ml/stage_include.py --include-dir <extracted>/Greetings --out videos/

INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition (CC-BY-4.0)
https://zenodo.org/records/4010759
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--include-dir", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--only", nargs="*", help="only these labels")
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)


def to_label(folder: str) -> str:
    """'49. How are you' -> 'howareyou'"""
    name = re.sub(r"^\s*\d+\.\s*", "", folder).strip().lower()
    return re.sub(r"[^a-z]", "", name)


staged: dict[str, int] = {}
for word_dir in sorted(p for p in args.include_dir.iterdir() if p.is_dir()):
    label = to_label(word_dir.name)
    if args.only and label not in args.only:
        continue
    for video in sorted(word_dir.iterdir()):
        if video.suffix.lower() not in {".mov", ".mp4", ".avi"}:
            continue
        link = args.out / f"{label}__{video.stem}{video.suffix.lower()}"
        if not link.exists():
            link.symlink_to(video.resolve())
        staged[label] = staged.get(label, 0) + 1

for label, count in sorted(staged.items()):
    print(f"  {label:<16} {count:>3} videos")
print(f"\nStaged into {args.out}. Next: python ml/extract.py --videos-dir {args.out}")

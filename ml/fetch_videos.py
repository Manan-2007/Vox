"""Download the source sign videos listed in ml/videos.txt.

    python ml/fetch_videos.py --out videos/
    python ml/extract.py --videos-dir videos/

Requires yt-dlp (pip install yt-dlp) and Node.js for YouTube's JS challenges.
The clips are ISL dictionary videos published by ISLRTC, Goa Board of
Education, and DEAF TV on YouTube; they are downloaded for local training and
demo playback only.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

failures = []
for line in (ML_DIR / "videos.txt").read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    label, vid = line.split()
    target = args.out / f"{label}__{vid}.mp4"
    if target.exists():
        print(f"  have {target.name}")
        continue
    print(f"  downloading {label} ({vid})")
    result = subprocess.run([
        sys.executable, "-m", "yt_dlp", "-q",
        "--js-runtimes", "node", "--remote-components", "ejs:github",
        "-f", "mp4[height<=480]/best[height<=480]/best",
        "-o", str(args.out / f"{label}__{vid}.%(ext)s"),
        f"https://www.youtube.com/watch?v={vid}",
    ])
    if result.returncode != 0:
        failures.append(f"{label} {vid}")

if failures:
    print("\nFailed:", ", ".join(failures))
    sys.exit(1)
print("\nAll videos present. Next: python ml/extract.py --videos-dir", args.out)

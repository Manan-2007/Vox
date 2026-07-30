"""Resolve ml/vocabulary.py against the official ISLRTC ISL dictionary.

The Indian Sign Language Research and Training Centre (ISLRTC, an autonomous
body under the Department of Empowerment of Persons with Disabilities,
Government of India) publishes its ISL dictionary as a public YouTube
playlist — one short clip per word, one signer, plain grey background, front-on
framing at a consistent distance. That is close to ideal input for landmark
extraction, and it is the *standard* form of each sign, which matters more than
any other property when the point is to teach and to be understood.

Two stages, so a bad match costs nothing:

    python ml/fetch_dictionary.py --plan          # resolve only, print a report
    python ml/fetch_dictionary.py --download      # fetch what the plan resolved

The plan is written to ml/dictionary_plan.json so the download is reproducible
and reviewable. Clips land in ml/videos/<gloss>__<videoid>.mp4, which is exactly
the layout ml/extract.py expects.

Why not INCLUDE for these words: INCLUDE (Zenodo 4010759) is a better *training*
corpus — 21 signers per word — but it is a lexicon of nouns and adjectives. It
contains no "eat", no "help", no "please", no "water", no question words. You
cannot hold a conversation in it. The dictionary covers those; the trade-off is
one signer per clip, which is handled by augmentation in ml/preprocess.py and by
the "teach a sign" flow for words a given user needs to be recognised.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ML_DIR))
from vocabulary import GLOSSES, search_terms  # noqa: E402

# The ISLRTC dictionary is published as several playlists. "Everyday Terms" is
# by far the largest (~4,100 entries) and carries most of the vocabulary; the
# others fill real gaps — "doctor", "hospital" and "ambulance" are medical
# terms, and the digits live in their own playlist.
PLAYLISTS = {
    "everyday": "PLFjydPMg4Dapq9vcdmGyHs8uJhiqMgUrX",
    "medical": "PLFjydPMg4DarE4-ra8BWjZ5UNJZZnPzVz",
    "numbers": "PLFjydPMg4DarnMdkwFwOQz__g-Sj25ISO",
    "academic": "PLFjydPMg4Dao754g-aMpjuqBcZQlEnqdV",
    "wellbeing": "PLFjydPMg4Dar-oytwy9SSJhirsLc0Zd6X",
    "basic_course": "PLFjydPMg4DaqDBkYtoBhE34D7RMRQUG6x",
}
CHANNEL = "https://www.youtube.com/@islrtc"
INDEX_PATH = ML_DIR / "dictionary_index.tsv"
EXTRA_PATH = ML_DIR / "dictionary_index_search.tsv"
PLAN_PATH = ML_DIR / "dictionary_plan.json"
DEFAULT_VIDEOS_DIR = ML_DIR / "videos"

#: Never take more than this many clips for one gloss. Variants ("sign 1",
#: "sign 2") are genuinely different forms and all are useful, but a word with
#: nine matches would otherwise dominate the training set.
MAX_CLIPS_PER_GLOSS = 4


# --------------------------------------------------------------- the index --

def build_index() -> list[tuple[str, str]]:
    """(video_id, title) for every dictionary entry, cached on disk."""
    if INDEX_PATH.exists():
        rows = []
        for line in INDEX_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            vid, _, title = line.partition("\t")
            rows.append((vid, title))
        return rows

    rows: list[tuple[str, str]] = []
    for name, playlist in PLAYLISTS.items():
        print(f"indexing playlist {name} ({playlist}) — this takes a minute")
        result = subprocess.run(
            [sys.executable, "-m", "yt_dlp", "--flat-playlist",
             "--print", "%(id)s\t%(title)s",
             "--js-runtimes", "node", "--remote-components", "ejs:github",
             f"https://www.youtube.com/playlist?list={playlist}"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            sys.exit(f"could not index playlist {name}:\n{result.stderr[-800:]}")
        for line in result.stdout.splitlines():
            vid, _, title = line.partition("\t")
            if vid and title and title != "NA":
                rows.append((vid, title))

    INDEX_PATH.write_text(
        "".join(f"{v}\t{t}\n" for v, t in rows), encoding="utf-8"
    )
    print(f"wrote {len(rows)} entries to {INDEX_PATH.name}")
    return rows


# -------------------------------------------------------------- matching ----

#: Trailing qualifiers the dictionary adds to a title. Stripping them lets
#: "pain (sign 1)" match the term "pain" while still being a distinct clip.
QUALIFIER = re.compile(
    r"\s*[\(\[]?\s*(sign\s*\d+|meaning|example|for\s+\w+|verb|noun|number|"
    r"competition[^)]*|ethnicity)\s*[\)\]]?\s*$",
    re.IGNORECASE,
)


def normalize(text: str) -> str:
    """Lowercase, strip qualifiers and punctuation, collapse whitespace."""
    text = text.strip()
    # A title may carry more than one qualifier: "why (meaning) (sign 2)".
    for _ in range(3):
        stripped = QUALIFIER.sub("", text)
        if stripped == text:
            break
        text = stripped
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def title_variants(title: str) -> list[str]:
    """A dictionary title may list synonyms: 'road, street' -> both.

    The dictionary names a sign by every English word it covers, comma
    separated: "want, desire, wish for", "hard, difficult", "add, more". Each
    part is a legitimate name for the same clip, so each is matchable.

    Splitting happens BEFORE normalization, because normalization strips the
    commas. Only split when every part is short: "Take a long time at
    something, take hours at something" is one idiom, not two words, and
    splitting it would let it match far too eagerly.
    """
    # Strip the trailing qualifier first so "again, repeat (sign 1)" splits into
    # "again" and "repeat" rather than "again" and "repeat (sign 1)".
    stripped = title.strip()
    for _ in range(3):
        shorter = QUALIFIER.sub("", stripped)
        if shorter == stripped:
            break
        stripped = shorter

    base = normalize(stripped)
    if not base:
        return []
    parts = [normalize(p) for p in stripped.split(",")]
    parts = [p for p in parts if p]
    if len(parts) > 1 and all(len(p.split()) <= 3 for p in parts):
        return list(dict.fromkeys([base, *parts]))
    return [base]


def match_score(term: str, title: str) -> int:
    """How well a dictionary title serves as a clip for `term`. 0 = unusable.

    Deliberately strict: a wrong clip is worse than a missing one, because it
    teaches the model the wrong sign and shows the user the wrong hands. Only
    exact matches (after stripping the dictionary's own qualifiers) count, so
    "help" never matches "helpline" and "no" never matches "No pain, no gain".
    """
    term_n = normalize(term)
    if not term_n:
        return 0
    variants = title_variants(title)
    if term_n in variants:
        # An unqualified title is the canonical clip; a "(sign 2)" is a variant.
        return 100 if normalize(title) == term_n and "(" not in title else 90
    return 0


def channel_search(terms: list[str], per_term: int = 8) -> list[tuple[str, str]]:
    """Search ISLRTC's own channel. Used only for glosses the playlists miss.

    Several of the most ordinary words — "hello", "help", "doctor", "hospital",
    "family" — are published on the channel but are not members of any of the
    dictionary playlists, so a playlist-only index misses exactly the vocabulary
    the product needs most. Searching the channel (not YouTube at large) keeps
    every clip from the same official source and the same recording setup.
    """
    found: list[tuple[str, str]] = []
    for term in terms:
        query = re.sub(r"[^a-z0-9 ]", "", term.lower()).replace(" ", "+")
        if not query:
            continue
        result = subprocess.run(
            [sys.executable, "-m", "yt_dlp", "--flat-playlist",
             "--print", "%(id)s\t%(title)s", "--playlist-end", str(per_term),
             "--js-runtimes", "node", "--remote-components", "ejs:github",
             f"{CHANNEL}/search?query={query}"],
            capture_output=True, text=True,
        )
        for line in result.stdout.splitlines():
            vid, _, title = line.partition("\t")
            # Playlists come back from search too; they have no usable clip.
            if vid.startswith("PL") or not vid or not title or title == "NA":
                continue
            found.append((vid, title))
    return found


def load_extra() -> list[tuple[str, str]]:
    if not EXTRA_PATH.exists():
        return []
    rows = []
    for line in EXTRA_PATH.read_text(encoding="utf-8").splitlines():
        vid, _, title = line.partition("\t")
        if vid and title:
            rows.append((vid, title))
    return rows


def save_extra(rows: list[tuple[str, str]]) -> None:
    merged = dict(load_extra())
    merged.update(dict(rows))
    EXTRA_PATH.write_text(
        "".join(f"{v}\t{t}\n" for v, t in sorted(merged.items())), encoding="utf-8"
    )


def resolve(index: list[tuple[str, str]]) -> tuple[dict, list[str]]:
    """gloss -> [{id, title}] for every gloss we can serve, plus the misses."""
    by_gloss: dict[str, list[dict]] = {}
    missing: list[str] = []

    for gloss in GLOSSES:
        candidates: list[tuple[int, int, str, str]] = []
        for rank, term in enumerate(search_terms(gloss)):
            for vid, title in index:
                score = match_score(term, title)
                if score:
                    # earlier search terms win ties: rank ascending
                    candidates.append((-score, rank, vid, title))
        if not candidates:
            missing.append(gloss)
            continue
        candidates.sort()
        seen: set[str] = set()
        picked: list[dict] = []
        for _, _, vid, title in candidates:
            if vid in seen:
                continue
            seen.add(vid)
            picked.append({"id": vid, "title": title})
            if len(picked) >= MAX_CLIPS_PER_GLOSS:
                break
        by_gloss[gloss] = picked

    return by_gloss, missing


# -------------------------------------------------------------- download ----

def download(plan: dict, out_dir: Path, limit: int | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (gloss, clip["id"])
        for gloss, clips in plan["glosses"].items()
        for clip in clips
    ]
    if limit:
        jobs = jobs[:limit]

    have = 0
    failed: list[str] = []
    for i, (gloss, vid) in enumerate(jobs, 1):
        target = out_dir / f"{gloss}__{vid}.mp4"
        if target.exists() and target.stat().st_size > 1024:
            have += 1
            continue
        result = subprocess.run(
            [sys.executable, "-m", "yt_dlp", "-q", "--no-warnings",
             "--js-runtimes", "node", "--remote-components", "ejs:github",
             "-f", "mp4[height<=480]/best[height<=480]/best",
             "-o", str(out_dir / f"{gloss}__{vid}.%(ext)s"),
             f"https://www.youtube.com/watch?v={vid}"],
            capture_output=True, text=True,
        )
        if result.returncode != 0 or not target.exists():
            failed.append(f"{gloss} ({vid})")
        if i % 25 == 0 or i == len(jobs):
            print(f"  {i}/{len(jobs)} clips  ({have} already present, "
                  f"{len(failed)} failed)", flush=True)

    print(f"\n{len(jobs) - len(failed)}/{len(jobs)} clips in {out_dir}")
    if failed:
        print(f"failed ({len(failed)}): {', '.join(failed[:20])}"
              + (" …" if len(failed) > 20 else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--plan", action="store_true",
                        help="resolve the vocabulary and write the plan only")
    parser.add_argument("--download", action="store_true",
                        help="download the clips the plan resolved")
    parser.add_argument("--videos-dir", type=Path, default=DEFAULT_VIDEOS_DIR)
    parser.add_argument("--limit", type=int, default=None,
                        help="download at most this many clips (for a smoke test)")
    parser.add_argument("--no-search", action="store_true",
                        help="skip the channel-search pass for unresolved glosses")
    args = parser.parse_args()
    if not args.plan and not args.download:
        args.plan = True

    if args.plan:
        index = build_index()
        index += load_extra()
        print(f"index: {len(index)} dictionary entries")
        by_gloss, missing = resolve(index)

        # Second pass: ask the channel directly about whatever the playlists did
        # not cover, cache the hits, and resolve again.
        if missing and not args.no_search:
            print(f"searching the channel for {len(missing)} unresolved glosses")
            hits: list[tuple[str, str]] = []
            for i, gloss in enumerate(missing, 1):
                rows = channel_search(search_terms(gloss))
                keep = [(v, t) for v, t in rows
                        if any(match_score(term, t)
                               for term in search_terms(gloss))]
                hits += keep
                print(f"  {i:>3}/{len(missing)} {gloss:<14} "
                      f"{len(keep)} match(es) of {len(rows)} result(s)", flush=True)
            if hits:
                save_extra(hits)
                index += hits
                by_gloss, missing = resolve(index)

        clips = sum(len(v) for v in by_gloss.values())
        PLAN_PATH.write_text(json.dumps({
            "_source": "ISLRTC official Indian Sign Language dictionary "
                       "(youtube.com/@islrtc) — Government of India",
            "glosses": by_gloss,
            "missing": missing,
        }, indent=1), encoding="utf-8")

        counts = defaultdict(int)
        for v in by_gloss.values():
            counts[len(v)] += 1
        print(f"\nresolved {len(by_gloss)}/{len(GLOSSES)} glosses "
              f"-> {clips} clips")
        for n in sorted(counts):
            print(f"  {counts[n]:>3} glosses with {n} clip(s)")
        if missing:
            print(f"\nno dictionary entry for {len(missing)}:")
            print("  " + ", ".join(missing))
        print(f"\nplan -> {PLAN_PATH.name}. Next: "
              f"python ml/fetch_dictionary.py --download")

    if args.download:
        if not PLAN_PATH.exists():
            sys.exit("no plan — run with --plan first")
        download(json.loads(PLAN_PATH.read_text()), args.videos_dir, args.limit)
        print(f"Next: python ml/extract.py --videos-dir {args.videos_dir}")


if __name__ == "__main__":
    main()

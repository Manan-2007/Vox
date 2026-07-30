"""Turn collected .npy samples into a normalized, split training set.

Reads every sample under ml/data/<label>/*.npy, applies the shared
`normalize_frame` to each of its 30 frames, and writes to ml/data/processed/:

    X_train.npy    (n_train, 30, 126) float32
    y_train.npy    (n_train,)         int64
    X_val.npy      (n_val,   30, 126) float32
    y_val.npy      (n_val,)           int64
    label_map.json {"labels": [...], "label_to_index": {...}}

Label indices come from sorting the label names, so the mapping is stable across
runs and across machines. The backend must read label_map.json rather than
re-deriving the order — a re-sorted map silently relabels every prediction.

Usage:
    python ml/preprocess.py
    python ml/preprocess.py --val-split 0.25 --seed 7
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize import FEATURE_DIM, SEQUENCE_LENGTH, normalize_sequence  # noqa: E402

ML_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ML_DIR / "data"
PROCESSED_DIRNAME = "processed"


def discover_labels(data_dir: Path) -> list[str]:
    """Label directories, sorted — this ordering defines the integer indices."""
    if not data_dir.exists():
        sys.exit(f"No data directory at {data_dir}. Run ml/collect.py first.")
    labels = sorted(
        d.name
        for d in data_dir.iterdir()
        if d.is_dir() and d.name != PROCESSED_DIRNAME and not d.name.startswith(".")
    )
    if not labels:
        sys.exit(f"No label directories under {data_dir}. Run ml/collect.py first.")
    return labels


def source_video(path: Path) -> str:
    """The source video a sample came from — `<video>_<window>.npy`."""
    return path.stem.rsplit("_", 1)[0]


def load_label(label_dir: Path) -> tuple[list[np.ndarray], list[str], list[str], int]:
    """Load and normalize every sample for one label.

    Returns (sequences, groups, skipped_messages, n_empty); `groups` is the
    source video per sample, and n_empty counts samples in which no hand was
    ever detected.
    """
    sequences: list[np.ndarray] = []
    groups: list[str] = []
    skipped: list[str] = []
    n_empty = 0

    for path in sorted(label_dir.glob("*.npy")):
        try:
            raw = np.load(path)
        except Exception as exc:
            skipped.append(f"{path.name}: unreadable ({exc})")
            continue

        if raw.shape != (SEQUENCE_LENGTH, FEATURE_DIM):
            skipped.append(
                f"{path.name}: shape {raw.shape}, expected "
                f"({SEQUENCE_LENGTH}, {FEATURE_DIM})"
            )
            continue

        if not raw.any():
            n_empty += 1  # kept, but flagged: no hand was ever visible

        sequences.append(normalize_sequence(raw))
        groups.append(source_video(path))

    return sequences, groups, skipped, n_empty


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess ISL landmark samples.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--val-split", type=float, default=0.2, help="default 0.2")
    parser.add_argument(
        "--test-split", type=float, default=0.2,
        help="share of source videos held back and never trained or tuned on "
             "(default 0.2; 0 disables)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0.0 < args.val_split < 1.0:
        sys.exit("--val-split must be between 0 and 1 (exclusive).")
    if not 0.0 <= args.test_split < 1.0:
        sys.exit("--test-split must be at least 0 and below 1.")

    labels = discover_labels(args.data_dir)
    out_dir = args.data_dir / PROCESSED_DIRNAME

    print(f"Reading from {args.data_dir}")
    print(f"Found {len(labels)} label(s): {', '.join(labels)}\n")

    sequences: list[np.ndarray] = []
    y_list: list[int] = []
    group_list: list[str] = []
    all_skipped: list[str] = []
    empty_by_label: dict[str, int] = {}

    for index, label in enumerate(labels):
        seqs, groups, skipped, n_empty = load_label(args.data_dir / label)
        sequences.extend(seqs)
        y_list.extend([index] * len(seqs))
        group_list.extend(f"{label}/{g}" for g in groups)
        all_skipped.extend(f"{label}/{m}" for m in skipped)
        if n_empty:
            empty_by_label[label] = n_empty
        n_videos = len(set(groups))
        print(f"  [{index}] {label:<20} {len(seqs):>4} sample(s) from {n_videos} video(s)")

    if all_skipped:
        print(f"\nSkipped {len(all_skipped)} malformed file(s):")
        for message in all_skipped[:10]:
            print(f"  - {message}")
        if len(all_skipped) > 10:
            print(f"  ... and {len(all_skipped) - 10} more")

    if empty_by_label:
        print("\nWarning: samples with no hand detected in any frame (all zeros):")
        for label, count in empty_by_label.items():
            print(f"  - {label}: {count}")
        print("  These train the model on nothing. Consider deleting and re-recording.")

    if not sequences:
        sys.exit("\nNo usable samples found.")

    X = np.stack(sequences).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)

    groups = np.asarray(group_list)
    counts = Counter(y.tolist())

    if len(labels) < 2:
        print("\nWarning: only one label — a classifier needs at least two.")

    # ------------------------------------------------------------------
    # Split by SOURCE VIDEO, never by window.
    #
    # Windows from one video overlap heavily, so a random split puts near
    # duplicates of the same frames on both sides and reports accuracy that
    # is really memorisation. Holding out whole videos makes validation
    # answer the question that matters: does this transfer to a signer the
    # model has never seen?
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Three splits, not two.
    #
    # Validation picks the checkpoint and decides when to stop, which makes its
    # accuracy an optimistic estimate of itself. The test split is held back from
    # both, so it is the only number that answers "does this work on a recording
    # nobody tuned against". A class with too few source videos to spare one for
    # each split gives up its test share first and its validation share second —
    # training data is the scarcer resource.
    # ------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    val_mask = np.zeros(len(y), bool)
    test_mask = np.zeros(len(y), bool)
    single_video = []
    untested = []

    for index, label in enumerate(labels):
        rows = np.flatnonzero(y == index)
        label_videos = sorted(set(groups[rows]))
        if len(label_videos) < 2:
            single_video.append(label)
            continue

        shuffled = list(rng.permutation(label_videos))
        # At least one video always stays in training.
        spare = len(shuffled) - 1

        n_test = min(spare, max(1, int(round(len(shuffled) * args.test_split))))
        if args.test_split == 0:
            n_test = 0
        test_videos = shuffled[:n_test]
        spare -= n_test

        n_val = min(spare, max(1, int(round(len(shuffled) * args.val_split))))
        val_videos = shuffled[n_test : n_test + n_val]

        if test_videos:
            test_mask |= np.isin(groups, test_videos)
        else:
            untested.append(label)
        if val_videos:
            val_mask |= np.isin(groups, val_videos)

    if single_video:
        print(
            f"\nOnly one source recording for {len(single_video)} label(s): "
            + ", ".join(single_video[:20])
            + (" …" if len(single_video) > 20 else "")
            + "\n  All their samples go to training. Nothing can be held out, so no"
            "\n  measurement covers these words — they are trainable but unverified."
            "\n  Record your own with: python ml/collect.py <word>"
        )
    if untested:
        print(
            f"\n{len(untested)} label(s) have too few recordings to spare one for the"
            " test split;\n  they are validated but not tested."
        )

    X_train, y_train = X[~val_mask & ~test_mask], y[~val_mask & ~test_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    if len(X_val) == 0:
        sys.exit("\nNo validation samples — every class has a single source video.")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "X_val.npy", X_val)
    np.save(out_dir / "y_val.npy", y_val)
    np.save(out_dir / "X_test.npy", X_test)
    np.save(out_dir / "y_test.npy", y_test)
    label_map = {
        "labels": labels,
        "label_to_index": {label: i for i, label in enumerate(labels)},
    }
    (out_dir / "label_map.json").write_text(json.dumps(label_map, indent=2) + "\n")

    train_counts = Counter(y_train.tolist())
    val_counts = Counter(y_val.tolist())
    test_counts = Counter(y_test.tolist())

    print(f"\nWrote to {out_dir}")
    print(f"  X_train {X_train.shape}  y_train {y_train.shape}")
    print(f"  X_val   {X_val.shape}  y_val   {y_val.shape}")
    print(f"  X_test  {X_test.shape}  y_test  {y_test.shape}")
    print(f"  label_map.json ({len(labels)} labels)")

    # With a 200-word vocabulary a full per-class table is unreadable; print the
    # distribution and only the classes that are actually short of data.
    print("\nPer-class sample counts")
    thin = [labels[i] for i in range(len(labels)) if counts[i] < 8]
    print(f"  total {len(y)}  train {len(y_train)}  val {len(y_val)}  test {len(y_test)}")
    print(f"  median samples per class: {int(np.median([counts[i] for i in range(len(labels))]))}")
    print(f"  fewest: {min(counts[i] for i in range(len(labels)))}, "
          f"most: {max(counts[i] for i in range(len(labels)))}")
    if thin:
        print(f"  {len(thin)} class(es) under 8 samples: "
              + ", ".join(thin[:20]) + (" …" if len(thin) > 20 else ""))

    no_val = [labels[i] for i in range(len(labels)) if val_counts[i] == 0]
    no_test = [labels[i] for i in range(len(labels)) if test_counts[i] == 0]
    print(
        f"\nCoverage: {len(labels) - len(no_val)}/{len(labels)} labels have validation"
        f" samples, {len(labels) - len(no_test)}/{len(labels)} have test samples."
    )
    print(
        "  Accuracy is only meaningful for labels with test samples. The rest are\n"
        "  trained and served but unmeasured — the UI marks them as such."
    )

    # Written next to the splits so the app can tell verified words from
    # unverified ones without re-deriving it from the data directory.
    coverage = {
        label: {
            "samples": counts[i],
            "train": train_counts[i],
            "val": val_counts[i],
            "test": test_counts[i],
            "verified": test_counts[i] > 0,
        }
        for i, label in enumerate(labels)
    }
    (out_dir / "coverage.json").write_text(json.dumps(coverage, indent=1) + "\n")


if __name__ == "__main__":
    main()

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
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0.0 < args.val_split < 1.0:
        sys.exit("--val-split must be between 0 and 1 (exclusive).")

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
    rng = np.random.default_rng(args.seed)
    val_mask = np.zeros(len(y), bool)
    single_video = []

    for index, label in enumerate(labels):
        rows = np.flatnonzero(y == index)
        label_videos = sorted(set(groups[rows]))
        if len(label_videos) < 2:
            single_video.append(label)
            continue
        n_val = max(1, int(round(len(label_videos) * args.val_split)))
        n_val = min(n_val, len(label_videos) - 1)  # always keep one for training
        chosen = rng.choice(label_videos, size=n_val, replace=False)
        val_mask |= np.isin(groups, chosen)

    if single_video:
        print(
            "\nWarning: only one source video for: " + ", ".join(single_video) +
            "\n  Those samples all go to training — validation cannot measure them."
            "\n  Add another video (or your own recordings) for a real score."
        )

    X_train, y_train = X[~val_mask], y[~val_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    if len(X_val) == 0:
        sys.exit("\nNo validation samples — every class has a single source video.")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "X_val.npy", X_val)
    np.save(out_dir / "y_val.npy", y_val)
    label_map = {
        "labels": labels,
        "label_to_index": {label: i for i, label in enumerate(labels)},
    }
    (out_dir / "label_map.json").write_text(json.dumps(label_map, indent=2) + "\n")

    train_counts = Counter(y_train.tolist())
    val_counts = Counter(y_val.tolist())

    held_out = sorted(set(groups[val_mask]))
    print(f"\nHeld-out videos (validation): {', '.join(held_out)}")
    print(f"\nWrote to {out_dir}")
    print(f"  X_train {X_train.shape}  y_train {y_train.shape}")
    print(f"  X_val   {X_val.shape}  y_val   {y_val.shape}")
    print(f"  label_map.json ({len(labels)} labels)")

    print("\nPer-class sample counts")
    print(f"  {'idx':<5}{'label':<20}{'total':>7}{'train':>7}{'val':>7}")
    for index, label in enumerate(labels):
        print(
            f"  {index:<5}{label:<20}{counts[index]:>7}"
            f"{train_counts[index]:>7}{val_counts[index]:>7}"
        )
    print(f"  {'':<5}{'TOTAL':<20}{len(y):>7}{len(y_train):>7}{len(y_val):>7}")

    missing = [labels[i] for i in range(len(labels)) if val_counts[i] == 0]
    if missing:
        print(
            f"\nWarning: no validation samples for: {', '.join(missing)}. "
            "Validation accuracy will not reflect these classes."
        )


if __name__ == "__main__":
    main()

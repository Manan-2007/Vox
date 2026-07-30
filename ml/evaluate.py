"""Measure the recogniser honestly, and write what the app should claim.

A single top-1 accuracy over 242 classes is close to meaningless for this
product, because the product does not behave like a top-1 classifier. The
backend refuses to emit anything below a confidence threshold and requires the
same class to win several frames in a row, so the question that matters is not
"how often is the top guess right" but "when it does speak, is it right, and
which words can it be trusted on".

This writes ml/models/metrics.json, which the app reads to mark words as
verified or unverified. Run it after ml/train.py:

    python ml/evaluate.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ML_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ML_DIR))

DEFAULT_PROCESSED = ML_DIR / "data" / "processed"
DEFAULT_MODELS = ML_DIR / "models"
#: Confidence the backend requires before it will emit a word.
GATE = 0.85


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--gate", type=float, default=GATE)
    args = parser.parse_args()

    for name in ("X_test.npy", "y_test.npy"):
        if not (args.processed_dir / name).exists():
            sys.exit(f"missing {name} — run ml/preprocess.py")

    import keras
    import layers  # noqa: F401  (registers AttentionPooling)

    X_test = np.load(args.processed_dir / "X_test.npy")
    y_test = np.load(args.processed_dir / "y_test.npy")
    labels = json.loads((args.models_dir / "label_map.json").read_text())["labels"]
    coverage = {}
    coverage_path = args.processed_dir / "coverage.json"
    if coverage_path.exists():
        coverage = json.loads(coverage_path.read_text())

    model = keras.models.load_model(args.models_dir / "vox_lstm.keras")
    probs = model.predict(X_test, verbose=0)
    ranked = np.argsort(probs, axis=1)[:, ::-1]

    top1 = float((ranked[:, 0] == y_test).mean())
    top3 = float(np.mean([y in row for y, row in zip(y_test, ranked[:, :3])]))
    top5 = float(np.mean([y in row for y, row in zip(y_test, ranked[:, :5])]))

    # What the product actually does: speak only above the gate.
    confidence = probs[np.arange(len(y_test)), ranked[:, 0]]
    spoke = confidence >= args.gate
    precision = (
        float((ranked[spoke, 0] == y_test[spoke]).mean()) if spoke.any() else 0.0
    )
    coverage_rate = float(spoke.mean())

    print("=" * 66)
    print("  HELD-OUT TEST — recordings never trained on or tuned against")
    print("=" * 66)
    print(f"  classes                 : {len(labels)} "
          f"({len(set(y_test.tolist()))} present in the test set)")
    print(f"  test samples            : {len(y_test)}")
    print(f"  chance baseline         : {1 / len(labels):.1%}")
    print()
    print(f"  top-1 accuracy          : {top1:.1%}")
    print(f"  top-3 accuracy          : {top3:.1%}")
    print(f"  top-5 accuracy          : {top5:.1%}")
    print()
    print(f"  With the backend's {args.gate:.0%} confidence gate:")
    print(f"    speaks on             : {coverage_rate:.1%} of samples")
    print(f"    correct when it speaks: {precision:.1%}")
    print("    (the rest it stays silent on, which is the intended behaviour —")
    print("     a wrong word is far worse than no word)")

    # Per class, and grouped by how much data that class had.
    per_class: dict[str, dict] = {}
    by_recordings: dict[int, list[bool]] = defaultdict(list)
    for index, label in enumerate(labels):
        rows = y_test == index
        if not rows.any():
            per_class[label] = {"support": 0, "top1": None, "verified": False}
            continue
        correct = float((ranked[rows, 0] == index).mean())
        per_class[label] = {
            "support": int(rows.sum()),
            "top1": round(correct, 3),
            "verified": True,
        }
        recordings = coverage.get(label, {}).get("samples", 0)
        bucket = 1 if recordings <= 2 else 2 if recordings <= 5 else 3
        by_recordings[bucket] += (ranked[rows, 0] == index).tolist()

    print("\n  Accuracy by how many samples the word had:")
    names = {1: "1-2 samples", 2: "3-5 samples", 3: "6+ samples"}
    for bucket in sorted(by_recordings):
        results = by_recordings[bucket]
        print(f"    {names[bucket]:<14} {np.mean(results):>6.1%}  "
              f"({len(results)} test samples)")

    reliable = sorted(
        label for label, stats in per_class.items()
        if stats["verified"] and (stats["top1"] or 0) >= 0.8
    )
    print(f"\n  {len(reliable)} word(s) at 80%+ on held-out recordings:")
    print("    " + ", ".join(reliable) if reliable else "    none")

    metrics = {
        "classes": len(labels),
        "test_samples": int(len(y_test)),
        "classes_measured": len(set(y_test.tolist())),
        "top1": round(top1, 4),
        "top3": round(top3, 4),
        "top5": round(top5, 4),
        "gate": args.gate,
        "gated_precision": round(precision, 4),
        "gated_coverage": round(coverage_rate, 4),
        "reliable": reliable,
        "per_class": per_class,
    }
    out = args.models_dir / "metrics.json"
    out.write_text(json.dumps(metrics, indent=1) + "\n")
    print(f"\n  wrote {out}")

    # The frontend reads this to mark words the recogniser can be trusted on.
    web = ML_DIR.parent / "frontend" / "public" / "signs" / "recognition.json"
    if web.parent.exists():
        web.write_text(json.dumps({
            "classes": len(labels),
            "top1": round(top1, 4),
            "top3": round(top3, 4),
            "gatedPrecision": round(precision, 4),
            "reliable": reliable,
            "measured": sorted(
                label for label, stats in per_class.items() if stats["verified"]
            ),
        }, indent=1) + "\n")
        print(f"  wrote {web}")


if __name__ == "__main__":
    main()

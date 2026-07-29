"""Train the LSTM sign classifier on the preprocessed landmark sequences.

Reads ml/data/processed/ (written by ml/preprocess.py) and writes to ml/models/:

    vox_lstm.keras          best model by val_loss
    label_map.json          copied from processed/ — travels WITH the model
    confusion_matrix.png    validation confusion matrix

The label map is copied next to the model on purpose. Label indices come from
sorting label names, so collecting a new sign renumbers the classes; a model
loaded against a freshly regenerated map would silently mispredict. The backend
must read ml/models/label_map.json, not ml/data/processed/label_map.json.

Note on the Masking layer: mask_value=0.0 drops a timestep only when all 126
features are exactly zero — i.e. frames where MediaPipe saw no hand at all. A
one-handed sign keeps its 63 zeros and is still learned from, and a normalized
hand has its wrist at exactly (0,0,0) without zeroing the frame. So masking
removes dead frames only, which is what we want.

Usage:
    python ml/train.py
    python ml/train.py --epochs 300 --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: never try to open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    classification_report,
    confusion_matrix,
)

ML_DIR = Path(__file__).resolve().parent
DEFAULT_PROCESSED_DIR = ML_DIR / "data" / "processed"
DEFAULT_MODELS_DIR = ML_DIR / "models"

SEQUENCE_LENGTH = 30
FEATURE_DIM = 126


def load_processed(processed_dir: Path):
    """Load X/y splits and the label map, or exit with a pointer to preprocess."""
    required = [
        "X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy", "label_map.json",
    ]
    missing = [name for name in required if not (processed_dir / name).exists()]
    if missing:
        sys.exit(
            f"Missing {', '.join(missing)} in {processed_dir}.\n"
            "Run: python ml/preprocess.py"
        )

    X_train = np.load(processed_dir / "X_train.npy")
    y_train = np.load(processed_dir / "y_train.npy")
    X_val = np.load(processed_dir / "X_val.npy")
    y_val = np.load(processed_dir / "y_val.npy")
    label_map = json.loads((processed_dir / "label_map.json").read_text())
    labels = label_map["labels"]

    for name, X in (("X_train", X_train), ("X_val", X_val)):
        if X.ndim != 3 or X.shape[1:] != (SEQUENCE_LENGTH, FEATURE_DIM):
            sys.exit(
                f"{name} has shape {X.shape}, expected "
                f"(n, {SEQUENCE_LENGTH}, {FEATURE_DIM}). Re-run ml/preprocess.py."
            )

    seen = set(np.concatenate([y_train, y_val]).tolist())
    if not seen <= set(range(len(labels))):
        sys.exit("Label indices in y fall outside label_map.json. Re-run preprocess.")

    return X_train, y_train, X_val, y_val, labels


def augment_sequence(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One augmented copy of a normalized (30, 126) sequence.

    Skeleton-space augmentation on normalized landmarks: small in-plane
    rotation, uniform scale jitter, Gaussian noise, and temporal resampling
    (speed variation). The literature reports sizeable accuracy gains from
    exactly these transforms on landmark data.

    Zero stays zero: absent-hand blocks and empty frames are never touched,
    otherwise noise would "unmask" them and corrupt both the Masking layer's
    view and the absent-hand convention.
    """
    out = seq.copy()

    # temporal: resample to a random speed (0.8x..1.2x), back to 30 frames
    if rng.random() < 0.5:
        speed = rng.uniform(0.8, 1.2)
        src = np.clip(np.arange(30) * speed, 0, 29)
        lo = np.floor(src).astype(int)
        hi = np.minimum(lo + 1, 29)
        frac = (src - lo)[:, None]
        resampled = out[lo] * (1 - frac) + out[hi] * frac
        # a frame interpolated between an empty and a non-empty frame is
        # neither — keep hard emptiness from the nearer source frame
        empty = ~seq.any(axis=1)
        nearest = np.where(frac[:, 0] < 0.5, lo, hi)
        resampled[empty[nearest]] = 0.0
        out = resampled.astype(np.float32)

    theta = rng.uniform(-13, 13) * np.pi / 180.0  # in-plane rotation
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    scale = rng.uniform(0.9, 1.1)
    noise_sd = 0.01

    for block in range(2):
        lo = block * 63
        pts = out[:, lo : lo + 63].reshape(-1, 21, 3)
        present = pts.any(axis=(1, 2))  # per-frame: is this hand there?
        if not present.any():
            continue
        x = pts[present, :, 0].copy()
        y = pts[present, :, 1].copy()
        pts[present, :, 0] = (cos_t * x - sin_t * y) * scale
        pts[present, :, 1] = (sin_t * x + cos_t * y) * scale
        pts[present, :, 2] *= scale
        pts[present] += rng.normal(0, noise_sd, pts[present].shape)
        out[:, lo : lo + 63] = pts.reshape(-1, 63)

    return out.astype(np.float32)


def build_model(num_classes: int, keras):
    """Masking -> LSTM(64, seq) -> LSTM(128) -> Dense(64) -> Dropout -> softmax."""
    return keras.Sequential(
        [
            keras.layers.Input(shape=(SEQUENCE_LENGTH, FEATURE_DIM)),
            keras.layers.Masking(mask_value=0.0),
            keras.layers.LSTM(64, return_sequences=True),
            keras.layers.LSTM(128),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name="vox_lstm",
    )


def save_confusion_matrix(cm: np.ndarray, labels: list[str], accuracy: float, path: Path):
    """Write an annotated confusion matrix PNG. Colour is row-normalized (recall)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        row_totals = cm.sum(axis=1, keepdims=True)
        shaded = np.divide(cm, row_totals, out=np.zeros_like(cm, float), where=row_totals > 0)

    size = max(5.0, 0.75 * len(labels) + 2.5)
    fig, ax = plt.subplots(figsize=(size, size * 0.88))
    im = ax.imshow(shaded, cmap="Blues", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Vox validation confusion matrix\nval accuracy {accuracy:.1%}")

    for i in range(len(labels)):
        for j in range(len(labels)):
            if row_totals[i, 0] == 0:
                continue
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center", fontsize=9,
                color="white" if shaded[i, j] > 0.55 else "#222222",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, label="share of true class")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_confusion_matrix(cm: np.ndarray, labels: list[str]) -> None:
    """Same matrix as the PNG, in the terminal."""
    width = max(max((len(l) for l in labels), default=5), 5)
    header = " " * (width + 2) + "".join(f"{i:>5}" for i in range(len(labels)))
    print(f"\n  rows = true, cols = predicted (by index)\n{header}")
    for i, label in enumerate(labels):
        row = "".join(f"{v:>5}" for v in cm[i])
        print(f"  {label:<{width}}{row}   [{i}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Vox LSTM classifier.")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--augment", type=int, default=2, metavar="N",
        help="augmented copies per training sample (0 disables; default 2)",
    )
    args = parser.parse_args()

    X_train, y_train, X_val, y_val, labels = load_processed(args.processed_dir)
    num_classes = len(labels)

    if args.augment > 0:
        rng = np.random.default_rng(args.seed)
        copies = [X_train]
        for _ in range(args.augment):
            copies.append(np.stack([augment_sequence(s, rng) for s in X_train]))
        X_train = np.concatenate(copies)
        y_train = np.tile(y_train, args.augment + 1)
        # The validation set is never augmented — it must stay real.
        print(f"augmentation x{args.augment}: train grows to {len(X_train)} samples")

    import keras  # imported after arg parsing so --help stays fast

    keras.utils.set_random_seed(args.seed)

    args.models_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.models_dir / "vox_lstm.keras"

    print(f"train {X_train.shape}   val {X_val.shape}   {num_classes} classes")
    print(f"labels: {', '.join(f'{i}={l}' for i, l in enumerate(labels))}\n")

    model = build_model(num_classes, keras)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",  # y is integer indices
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience, restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            model_path, monitor="val_loss", save_best_only=True, verbose=0,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    # The label map travels with the model — see the module docstring.
    shutil.copyfile(args.processed_dir / "label_map.json", args.models_dir / "label_map.json")

    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    y_pred = model.predict(X_val, verbose=0).argmax(axis=1)

    epochs_run = len(history.history["loss"])
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1

    print("\n" + "=" * 62)
    print("  RESULTS")
    print("=" * 62)
    print(f"  epochs run       : {epochs_run} (best val_loss at epoch {best_epoch})")
    print(f"  val loss         : {val_loss:.4f}")
    print(f"  VAL ACCURACY     : {val_accuracy:.4f}  ({val_accuracy:.1%})")
    print(f"  chance baseline  : {1 / num_classes:.1%}")
    print(f"  val samples      : {len(y_val)}")

    print("\n  Classification report")
    print(
        classification_report(
            y_val, y_pred,
            labels=list(range(num_classes)), target_names=labels,
            zero_division=0, digits=3,
        )
    )

    cm = confusion_matrix(y_val, y_pred, labels=list(range(num_classes)))
    print_confusion_matrix(cm, labels)

    cm_path = args.models_dir / "confusion_matrix.png"
    save_confusion_matrix(cm, labels, val_accuracy, cm_path)

    print(f"\n  model      -> {model_path}")
    print(f"  label map  -> {args.models_dir / 'label_map.json'}")
    print(f"  confusion  -> {cm_path}")
    print(
        "\n  Note: the validation set drives early stopping and checkpointing, so\n"
        "  this accuracy is optimistic. Judge the model live via the webcam, and\n"
        "  hold out a separate test set before trusting the number."
    )


if __name__ == "__main__":
    main()

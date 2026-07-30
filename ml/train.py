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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize import FEATURE_DIM, SEQUENCE_LENGTH  # noqa: E402  (the contract)


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


def mirror_sequence(seq: np.ndarray) -> np.ndarray:
    """Left-right mirror image of a normalized sequence.

    This is the highest-value augmentation available, and it is not a trick: a
    left-handed signer produces the mirror image of the same sign, and it means
    the same thing. A model trained only on right-dominant recordings treats a
    left-handed signer as a different language.

    Mirroring in shoulder-anchored space is a negation of x about the body
    centre, plus a swap of the two hand blocks — the mirror of the left hand *is*
    the right hand, so leaving the blocks in place would produce a signer whose
    hands are on the wrong sides of their body.
    """
    out = seq.copy()

    # Swap the hand blocks. Absent blocks (all zeros) swap correctly too.
    left = out[:, 0:63].copy()
    out[:, 0:63] = out[:, 63:126]
    out[:, 63:126] = left

    # Negate x for every point, in every block that is present. Zero must stay
    # zero: -0.0 is falsy in numpy comparisons but it is a real value in the
    # array, and an absent block must remain exactly zeros.
    for block_start, block_len in ((0, 63), (63, 63), (126, 15)):
        col = slice(block_start, block_start + block_len)
        present = out[:, col].any(axis=1)
        if not present.any():
            continue
        for offset in range(block_start, block_start + block_len, 3):
            out[present, offset] = -out[present, offset]

    # The pose block's own left/right points must swap as well: L shoulder
    # becomes R shoulder, L elbow becomes R elbow. Nose stays put.
    pose = out[:, 126:141].reshape(len(out), 5, 3)
    pose[:, [1, 2]] = pose[:, [2, 1]]
    pose[:, [3, 4]] = pose[:, [4, 3]]
    out[:, 126:141] = pose.reshape(len(out), 15)

    return out.astype(np.float32)


def drop_frames(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomly blank a hand block in a few frames, imitating tracker dropout.

    Live tracking loses a hand for a frame or two several times a minute. If the
    model has only ever seen complete sequences, those gaps are out-of-
    distribution input at exactly the moment a sign is being made. Training
    through them is what makes the live system tolerate its own tracker.
    """
    out = seq.copy()
    for block in range(2):
        col = slice(block * 63, block * 63 + 63)
        if not out[:, col].any():
            continue
        # Up to 10% of frames, in short runs rather than scattered singletons —
        # that is how real dropout arrives.
        n_runs = rng.integers(0, 3)
        for _ in range(n_runs):
            start = int(rng.integers(0, SEQUENCE_LENGTH - 1))
            length = int(rng.integers(1, 3))
            out[start : start + length, col] = 0.0
    return out.astype(np.float32)


def augment_sequence(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One augmented copy of a normalized (SEQUENCE_LENGTH, 141) sequence.

    Runs in shoulder-anchored space, where the origin is the body centre, so a
    rotation/scale about the origin is a coherent whole-body transform. Hands
    AND the pose block are transformed together — rotating the hands while
    leaving the anchor fixed would teach the model impossible geometry.

    Zero stays zero: absent-hand blocks and hand-free frames are never touched,
    otherwise noise would "unmask" them and break both the Masking layer and
    the absent-hand convention.
    """
    out = seq.copy()

    # temporal: resample to a random speed (0.8x..1.2x), back to 30 frames
    if rng.random() < 0.5:
        speed = rng.uniform(0.8, 1.2)
        src = np.clip(np.arange(SEQUENCE_LENGTH) * speed, 0, SEQUENCE_LENGTH - 1)
        lo = np.floor(src).astype(int)
        hi = np.minimum(lo + 1, SEQUENCE_LENGTH - 1)
        frac = (src - lo)[:, None]
        resampled = out[lo] * (1 - frac) + out[hi] * frac
        # a frame interpolated between an empty and a non-empty hand block is
        # neither — keep hard emptiness from the nearer source frame
        nearest = np.where(frac[:, 0] < 0.5, lo, hi)
        for block in range(2):
            col = slice(block * 63, block * 63 + 63)
            empty = ~seq[:, col].any(axis=1)
            resampled[empty[nearest], col] = 0.0
        out = resampled.astype(np.float32)

    theta = rng.uniform(-12, 12) * np.pi / 180.0   # whole-body in-plane rotation
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    scale = rng.uniform(0.9, 1.1)                  # signer size / camera distance
    shift = rng.normal(0, 0.05, 2)                 # framing offset
    noise_sd = 0.01

    # every 3-float point in the frame: 42 hand points then 5 pose points
    for start in range(0, FEATURE_DIM, 3):
        block_start = (start // 63) * 63 if start < 126 else 126
        col = slice(block_start, block_start + (63 if start < 126 else 15))
        present = out[:, col].any(axis=1)          # per frame: is this block there?
        if not present.any():
            continue
        x = out[present, start].copy()
        y = out[present, start + 1].copy()
        out[present, start] = (cos_t * x - sin_t * y) * scale + shift[0]
        out[present, start + 1] = (sin_t * x + cos_t * y) * scale + shift[1]
        out[present, start + 2] *= scale
        out[present, start : start + 3] += rng.normal(0, noise_sd, (present.sum(), 3))

    if rng.random() < 0.4:
        out = drop_frames(out, rng)

    return out.astype(np.float32)


def build_model(num_classes: int, keras):
    """Bidirectional LSTM with attention pooling.

    Three changes from the original stacked unidirectional LSTM, each for a
    reason that shows up at this vocabulary size:

    BIDIRECTIONAL. A sign's identity often depends on where it *ends* — two signs
    can share an opening and diverge, and a forward-only pass has to commit
    before it sees the difference. Reading the window in both directions lets
    the early frames be interpreted in light of the late ones.

    ATTENTION POOLING instead of taking the last timestep. The final frame of a
    30-frame window is frequently the least informative one: the window slides
    over a live stream, so it may land after the sign has finished. Learning
    which frames matter beats always trusting the last one.

    LAYER NORM + DROPOUT throughout. At ~20 samples per class this model would
    otherwise memorise; the regularisation is doing as much work as the
    architecture.

    The input contract is unchanged: (SEQUENCE_LENGTH, FEATURE_DIM), masked on
    all-zero frames, so the backend and the browser need no changes.
    """
    inputs = keras.layers.Input(shape=(SEQUENCE_LENGTH, FEATURE_DIM))
    masked = keras.layers.Masking(mask_value=0.0)(inputs)

    x = keras.layers.Bidirectional(
        keras.layers.LSTM(96, return_sequences=True, dropout=0.2)
    )(masked)
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(96, return_sequences=True, dropout=0.2)
    )(x)
    x = keras.layers.LayerNormalization()(x)

    # Attention pooling. The Masking layer's mask propagates here, so padded
    # frames get no weight — see ml/layers.py for why this needs a custom layer
    # rather than Dense + Softmax.
    from layers import AttentionPooling  # noqa: PLC0415  (needs keras imported)

    pooled = AttentionPooling(name="attention_pool")(x)

    x = keras.layers.Dense(192, activation="relu")(pooled)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="vox_bilstm_attn")


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
        "--augment", type=int, default=3, metavar="N",
        help="augmented copies per training sample (0 disables; default 3)",
    )
    parser.add_argument(
        "--mirror", action=argparse.BooleanOptionalAction, default=True,
        help="also train on left-right mirrored copies (default on)",
    )
    parser.add_argument(
        "--label-smoothing", type=float, default=0.05,
        help="softens the target distribution; helps when classes are confusable",
    )
    args = parser.parse_args()

    X_train, y_train, X_val, y_val, labels = load_processed(args.processed_dir)
    num_classes = len(labels)

    if args.augment > 0 or args.mirror:
        rng = np.random.default_rng(args.seed)
        base = [X_train]
        base_y = [y_train]

        if args.mirror:
            # A mirrored copy of every sample, then augment BOTH — so the
            # augmentations apply to left- and right-dominant signing alike.
            base.append(np.stack([mirror_sequence(s) for s in X_train]))
            base_y.append(y_train)
        originals = np.concatenate(base)
        original_y = np.concatenate(base_y)

        copies = [originals]
        for _ in range(args.augment):
            copies.append(np.stack([augment_sequence(s, rng) for s in originals]))
        X_train = np.concatenate(copies)
        y_train = np.tile(original_y, args.augment + 1)
        # Validation and test are never augmented — they must stay real.
        print(
            f"augmentation: mirror={'on' if args.mirror else 'off'} x{args.augment} "
            f"-> train grows to {len(X_train)} samples"
        )

    import keras  # imported after arg parsing so --help stays fast

    keras.utils.set_random_seed(args.seed)

    args.models_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.models_dir / "vox_lstm.keras"

    print(f"train {X_train.shape}   val {X_val.shape}   {num_classes} classes")
    print(f"labels: {', '.join(f'{i}={l}' for i, l in enumerate(labels))}\n")

    model = build_model(num_classes, keras)
    # Label smoothing needs a distribution, not an index, so the targets are
    # one-hot here. With a 200-word vocabulary many classes are genuinely
    # confusable — several ISL signs differ only in a movement the landmarks
    # barely resolve — and a smoothed target stops the model being punished into
    # overconfidence about distinctions it cannot actually see.
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing)
    y_train_1h = keras.utils.to_categorical(y_train, num_classes)
    y_val_1h = keras.utils.to_categorical(y_val, num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss,
        metrics=["accuracy"],
    )
    model.summary()

    # Everything here watches val_ACCURACY, not val_loss.
    #
    # With label smoothing the two disagree, and they disagree in a way that
    # silently ruins the run: smoothed cross-entropy punishes confidence, so as
    # the model gets better at ranking the right class first it also gets more
    # confident and its val_loss climbs. Monitoring val_loss picked the epoch-1
    # weights out of a 19-epoch run and shipped a 17% model when the same run
    # reached 42%. Accuracy is what the product is judged on; monitor that.
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=args.patience,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            model_path, monitor="val_accuracy", mode="max", save_best_only=True,
            verbose=0,
        ),
    ]

    callbacks.append(
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", mode="max", factor=0.5,
            patience=max(3, args.patience // 3), min_lr=1e-5, verbose=1,
        )
    )

    history = model.fit(
        X_train, y_train_1h,
        validation_data=(X_val, y_val_1h),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    # The label map travels with the model — see the module docstring.
    shutil.copyfile(args.processed_dir / "label_map.json", args.models_dir / "label_map.json")

    val_loss, val_accuracy = model.evaluate(X_val, y_val_1h, verbose=0)
    y_pred = model.predict(X_val, verbose=0).argmax(axis=1)

    epochs_run = len(history.history["loss"])
    best_epoch = int(np.argmax(history.history["val_accuracy"])) + 1

    print("\n" + "=" * 62)
    print("  RESULTS")
    print("=" * 62)
    print(f"  epochs run       : {epochs_run} (best val_loss at epoch {best_epoch})")
    print(f"  val loss         : {val_loss:.4f}")
    print(f"  VAL ACCURACY     : {val_accuracy:.4f}  ({val_accuracy:.1%})")
    print(f"  chance baseline  : {1 / num_classes:.1%}")
    print(f"  val samples      : {len(y_val)}")

    if num_classes <= 40:
        print("\n  Classification report")
        print(
            classification_report(
                y_val, y_pred,
                labels=list(range(num_classes)), target_names=labels,
                zero_division=0, digits=3,
            )
        )

    cm = confusion_matrix(y_val, y_pred, labels=list(range(num_classes)))
    if num_classes <= 40:
        print_confusion_matrix(cm, labels)

    # A 242x242 confusion matrix is a 4 MB image with unreadable labels. Past a
    # few dozen classes the per-class numbers in metrics.json are the useful
    # artefact and the picture is not.
    cm_path = args.models_dir / "confusion_matrix.png"
    if len(labels) <= 40:
        save_confusion_matrix(cm, labels, val_accuracy, cm_path)
    else:
        cm_path.unlink(missing_ok=True)
        cm_path = None
        print(f"\n  {len(labels)} classes — confusion matrix PNG skipped; see "
              "ml/models/metrics.json for per-class results")

    print(f"\n  model      -> {model_path}")
    print(f"  label map  -> {args.models_dir / 'label_map.json'}")
    if cm_path:
        print(f"  confusion  -> {cm_path}")
    # The validation set chose when to stop training and which checkpoint to
    # keep, so its accuracy is optimistic by construction. The test split was
    # never looked at, and it is the only number worth quoting.
    test_x = args.processed_dir / "X_test.npy"
    test_y = args.processed_dir / "y_test.npy"
    if test_x.exists() and test_y.exists():
        X_test = np.load(test_x)
        y_test = np.load(test_y)
        if len(y_test):
            test_pred = model.predict(X_test, verbose=0).argmax(axis=1)
            test_accuracy = float((test_pred == y_test).mean())
            tested = sorted(set(y_test.tolist()))
            print("\n" + "=" * 62)
            print("  HELD-OUT TEST — signers and recordings never trained on")
            print("=" * 62)
            print(f"  TEST ACCURACY    : {test_accuracy:.4f}  ({test_accuracy:.1%})")
            print(f"  test samples     : {len(y_test)}")
            print(f"  classes measured : {len(tested)} of {num_classes}")
            if len(tested) < num_classes:
                print(
                    f"  {num_classes - len(tested)} class(es) have only one source\n"
                    "  recording, so nothing can be held out for them and this number\n"
                    "  says nothing about those words. See per_class_support below."
                )
            per_class = {}
            for index in range(num_classes):
                rows = y_test == index
                per_class[labels[index]] = {
                    "support": int(rows.sum()),
                    "accuracy": round(float((test_pred[rows] == index).mean()), 4)
                    if rows.any() else None,
                }
            (args.models_dir / "metrics.json").write_text(json.dumps({
                "val_accuracy": round(float(val_accuracy), 4),
                "test_accuracy": round(test_accuracy, 4),
                "test_samples": int(len(y_test)),
                "classes": num_classes,
                "classes_measured": len(tested),
                "per_class_support": per_class,
            }, indent=1) + "\n")
            print(f"  metrics    -> {args.models_dir / 'metrics.json'}")
    else:
        print(
            "\n  No test split found. Re-run ml/preprocess.py to produce one; the\n"
            "  validation accuracy above chose the checkpoint and is optimistic."
        )


if __name__ == "__main__":
    main()

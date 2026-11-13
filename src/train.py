"""
src/train.py — train the CNN on YOUR dataset and save it.

Run from the project root:
    python -m src.train                      # default settings from config.py
    python -m src.train --epochs 15          # change max epochs
    python -m src.train --batch-size 32      # use less memory
    python -m src.train --resplit            # recreate the train/val/test split
    python -m src.train --rebuild-cache      # re-read all images from disk

Steps:
  1. Detect hardware (GPU if TensorFlow can use one, otherwise CPU).
  2. Load / create the leakage-free patient-level split.
  3. Load images (cached after the first run) and build tf.data pipelines.
  4. Build and compile the CNN.
  5. Train with callbacks: ModelCheckpoint (keep best), EarlyStopping,
     ReduceLROnPlateau, CSVLogger.
  6. Save: models/malaria_cnn.keras, models/model_config.json,
     models/shap_background.npy, training history + graphs.
The TEST set is never used here; it is only used later by src/evaluate.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # hide TensorFlow info spam

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src import preprocessing as prep  # noqa: E402
from src.model import build_cnn, compile_model  # noqa: E402


def detect_hardware() -> dict:
    """Report which devices TensorFlow can use. Falls back to CPU automatically."""
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)  # don't grab all GPU memory
        except Exception:
            pass
    info = {
        "tensorflow_version": tf.__version__,
        "gpus": [g.name for g in gpus],
        "device_used": "GPU" if gpus else "CPU",
        "cpu_count": os.cpu_count(),
    }
    print(f"TensorFlow {info['tensorflow_version']} | device: {info['device_used']} "
          f"{info['gpus'] if gpus else ''} | CPU cores: {info['cpu_count']}")
    if not gpus:
        print("  No TensorFlow-compatible GPU found -> training on CPU (this is normal on Windows).")
    return info


def compute_class_weights(y: np.ndarray) -> dict:
    """'Balanced' weights so a minority class counts more in the loss (≈1.0 for balanced data)."""
    counts = np.bincount(y.astype(int), minlength=len(config.CLASS_NAMES))
    total = counts.sum()
    return {i: float(total / (len(counts) * c)) for i, c in enumerate(counts) if c > 0}


def save_shap_background(x_train_uint8: np.ndarray) -> None:
    """Save a small random sample of TRAINING images as SHAP's background dataset."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    n = min(config.SHAP_BACKGROUND_SIZE, len(x_train_uint8))
    idx = rng.choice(len(x_train_uint8), size=n, replace=False)
    np.save(config.SHAP_BACKGROUND_PATH, x_train_uint8[idx])


def train(epochs: int, batch_size: int, learning_rate: float,
          resplit: bool = False, rebuild_cache: bool = False) -> None:
    import tensorflow as tf
    from src.evaluate import plot_training_history

    config.ensure_dirs()
    tf.keras.utils.set_random_seed(config.RANDOM_SEED)   # reproducibility (python, numpy, tf)
    hardware = detect_hardware()

    # ---- data ----
    split_df = prep.create_splits(force=resplit)
    prep.print_split_summary(split_df)
    images = prep.load_all_images(split_df, config.IMAGE_SIZE, rebuild_cache=rebuild_cache)
    arrays = prep.get_split_arrays(split_df, images)
    (x_tr, y_tr), (x_va, y_va) = arrays["train"], arrays["val"]
    train_ds = prep.make_tf_dataset(x_tr, y_tr, batch_size, training=True)
    val_ds = prep.make_tf_dataset(x_va, y_va, batch_size, training=False)
    class_weight = compute_class_weights(y_tr)
    print(f"Class weights: { {config.CLASS_NAMES[k]: round(v, 3) for k, v in class_weight.items()} }")

    # ---- model ----
    model = compile_model(build_cnn(), learning_rate)
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(config.MODEL_PATH), monitor="val_loss",
                                           save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.EARLY_STOPPING_PATIENCE,
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=config.REDUCE_LR_PATIENCE, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.CSVLogger(str(config.HISTORY_PATH)),
    ]

    print(f"\nTraining for up to {epochs} epochs (batch size {batch_size}) ...")
    t0 = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                        class_weight=class_weight, callbacks=callbacks, verbose=1)
    minutes = (time.time() - t0) / 60

    # restore_best_weights=True means `model` now holds the best epoch -> save it
    model.save(str(config.MODEL_PATH))
    save_shap_background(x_tr)

    hist = history.history
    best_epoch = int(np.argmin(hist["val_loss"])) + 1
    summary = {
        "epochs_run": len(hist["loss"]),
        "best_epoch": best_epoch,
        "best_val_loss": float(hist["val_loss"][best_epoch - 1]),
        "best_val_accuracy": float(hist["val_accuracy"][best_epoch - 1]),
        "train_accuracy_at_best_epoch": float(hist["accuracy"][best_epoch - 1]),
        "training_minutes": round(minutes, 2),
        "epochs_max": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
        "image_size": config.IMAGE_SIZE,
        "train_images": int(len(y_tr)), "val_images": int(len(y_va)),
        "parameters": int(model.count_params()),
        "hardware": hardware,
        "note": "train accuracy is measured on augmented images with dropout active; "
                "see outputs/reports/metrics.json for clean train/val/test evaluation.",
    }
    with open(config.TRAINING_SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    prep.save_model_config({
        "split_strategy": str(split_df["split_strategy"].iloc[0]),
        "tensorflow_version": tf.__version__,
        "best_epoch": best_epoch,
        "shap_background": config.SHAP_BACKGROUND_PATH.name,
    })
    graphs = plot_training_history()

    print("\n================ TRAINING COMPLETE ================")
    print(f"Epochs run: {summary['epochs_run']} (best epoch: {best_epoch})  "
          f"time: {minutes:.1f} min on {hardware['device_used']}")
    print(f"Best validation accuracy: {summary['best_val_accuracy']:.4f}  "
          f"val loss: {summary['best_val_loss']:.4f}")
    print(f"Model saved to : {config.MODEL_PATH.relative_to(config.BASE_DIR)}")
    print(f"Config saved to: {config.MODEL_CONFIG_PATH.relative_to(config.BASE_DIR)}")
    print(f"Graphs         : {[str(g.relative_to(config.BASE_DIR)) for g in graphs]}")
    print("Next step      : python -m src.evaluate")


def parse_args():
    p = argparse.ArgumentParser(description="Train the malaria CNN")
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE)
    p.add_argument("--resplit", action="store_true", help="recreate the data split")
    p.add_argument("--rebuild-cache", action="store_true", help="re-read all images from disk")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    train(a.epochs, a.batch_size, a.learning_rate, a.resplit, a.rebuild_cache)

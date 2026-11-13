"""
src/preprocessing.py — everything that turns image files into model-ready numbers.

Key idea: ONE function, `preprocess_image`, is used for training, evaluation,
the Streamlit app, LIME and SHAP. That guarantees the model always sees images
prepared in exactly the same way (no train/inference mismatch).

    file -> RGB -> resize to 64x64 (uint8) -> divide by 255 -> float32 array (64, 64, 3)

This module also:
  * creates the leakage-free train/validation/test split (grouped by patient),
  * caches the resized images in cache/ so training starts quickly next time,
  * builds TensorFlow input pipelines with data augmentation (training only),
  * saves/loads the preprocessing + class-label configuration (models/model_config.json).

TensorFlow is imported only inside the functions that need it, so the
inspection and splitting code works even before TensorFlow is installed.
"""

from __future__ import annotations

import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# 1. Single-image preprocessing (shared by training AND inference)
# ---------------------------------------------------------------------------
def load_image_rgb(source) -> Image.Image:
    """
    Open an image from many kinds of input and return a PIL RGB image.
    `source` can be: a file path (str/Path), raw bytes, a file-like object
    (e.g. Streamlit's UploadedFile), a PIL Image, or a numpy array.
    Raises ValueError for anything that is not a readable image.
    """
    try:
        if isinstance(source, Image.Image):
            img = source
        elif isinstance(source, np.ndarray):
            arr = source
            if arr.dtype != np.uint8:
                arr = np.clip(arr * (255.0 if arr.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
        elif isinstance(source, (bytes, bytearray)):
            img = Image.open(io.BytesIO(source))
        elif hasattr(source, "read"):
            data = source.read()
            if hasattr(source, "seek"):
                source.seek(0)          # so the caller can read the file again
            img = Image.open(io.BytesIO(data))
        else:
            img = Image.open(Path(source))
        img.load()
    except Exception as exc:
        raise ValueError(f"Could not read image: {type(exc).__name__}: {exc}") from exc
    return img.convert("RGB")           # handles RGBA, grayscale (L), palette (P) images


def resize_uint8(img: Image.Image, image_size: int = config.IMAGE_SIZE) -> np.ndarray:
    """Resize to image_size x image_size with bilinear filtering; returns uint8 (H, W, 3)."""
    img = img.resize((image_size, image_size), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def normalize(arr_uint8: np.ndarray) -> np.ndarray:
    """0..255 integers -> 0.0..1.0 floats (float32)."""
    return arr_uint8.astype(np.float32) / 255.0


def preprocess_image(source, image_size: int = config.IMAGE_SIZE) -> np.ndarray:
    """The full pipeline for one image -> float32 array of shape (H, W, 3) in [0, 1]."""
    return normalize(resize_uint8(load_image_rgb(source), image_size))


# ---------------------------------------------------------------------------
# 2. Leakage-free data split
# ---------------------------------------------------------------------------
def _merge_duplicate_groups(df: pd.DataFrame) -> pd.Series:
    """
    Union-find: if two files are byte-identical (same md5), their groups are
    merged so the duplicates always land in the same split.
    """
    parent: dict[str, str] = {g: g for g in df["group"].unique()}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, grp in df[df["md5"].duplicated(keep=False)].groupby("md5"):
        roots = [find(g) for g in grp["group"].unique()]
        for r in roots[1:]:
            parent[r] = roots[0]
    return df["group"].map(find)


def create_splits(force: bool = False, verbose: bool = True) -> pd.DataFrame:
    """
    Split the valid images into train / val / test and save to outputs/reports/data_split.csv.

    Strategy:
      * If >= 95% of images have a patient ID -> split BY PATIENT (StratifiedGroupKFold),
        so no patient's cells appear in more than one split.
      * Otherwise -> stratified random split by image.
      * In both cases, exact duplicate images are forced into the same split.
    The split is reproducible (fixed RANDOM_SEED) and is saved, so training,
    evaluation and the app all use the identical split.
    """
    if config.SPLIT_PATH.exists() and not force:
        return pd.read_csv(config.SPLIT_PATH)

    if not config.INVENTORY_PATH.exists():
        if verbose:
            print("No dataset inventory found - running dataset inspection first ...")
        from src.dataset_inspection import run_inspection
        run_inspection(verbose=verbose)

    inv = pd.read_csv(config.INVENTORY_PATH, keep_default_na=False)
    inv["valid"] = inv["valid"].astype(str).str.lower() == "true"
    df = inv[inv["valid"]].copy().reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No valid images found. Run: python -m src.dataset_inspection")

    coverage = (df["patient_id"] != "").mean()
    if coverage >= 0.95:
        strategy = "patient"
        df["group"] = np.where(df["patient_id"] != "", "pat:" + df["patient_id"],
                               "img:" + df["rel_path"])
    else:
        strategy = "random_image"
        df["group"] = "img:" + df["rel_path"]
    df["split_group"] = _merge_duplicate_groups(df)

    y = df["label"].to_numpy()
    groups = df["split_group"].to_numpy()

    # --- test split ---
    n_test = max(2, round(1 / config.TEST_FRACTION))
    sgkf = StratifiedGroupKFold(n_splits=n_test, shuffle=True, random_state=config.RANDOM_SEED)
    trainval_idx, test_idx = next(sgkf.split(np.zeros(len(df)), y, groups))

    # --- validation split from the remainder ---
    val_share = config.VAL_FRACTION / (1 - config.TEST_FRACTION)
    n_val = max(2, round(1 / val_share))
    sgkf2 = StratifiedGroupKFold(n_splits=n_val, shuffle=True, random_state=config.RANDOM_SEED)
    tr_rel, va_rel = next(sgkf2.split(np.zeros(len(trainval_idx)), y[trainval_idx], groups[trainval_idx]))

    df["split"] = ""
    df.loc[trainval_idx[tr_rel], "split"] = "train"
    df.loc[trainval_idx[va_rel], "split"] = "val"
    df.loc[test_idx, "split"] = "test"
    df["split_strategy"] = strategy

    # --- leakage checks (fail loudly if something is wrong) ---
    assert (df["split"] != "").all()
    assert df.groupby("split_group")["split"].nunique().max() == 1, "group leaked across splits"
    assert df.groupby("md5")["split"].nunique().max() == 1, "duplicate image leaked across splits"
    if strategy == "patient":
        pid = df[df.patient_id != ""]
        assert pid.groupby("patient_id")["split"].nunique().max() == 1, "patient leaked across splits"

    out = df[["rel_path", "class_name", "label", "patient_id", "split_group", "md5",
              "split", "split_strategy"]]
    config.ensure_dirs()
    out.to_csv(config.SPLIT_PATH, index=False)
    plot_split_distribution(out)
    if verbose:
        print_split_summary(out)
    return out


def split_summary(split_df: pd.DataFrame) -> dict:
    s = {}
    for sp in SPLITS:
        part = split_df[split_df.split == sp]
        s[sp] = {
            "images": int(len(part)),
            "fraction": round(len(part) / len(split_df), 4),
            "per_class": part["class_name"].value_counts().reindex(config.CLASS_NAMES, fill_value=0)
                           .astype(int).to_dict(),
            "patients": int(part.loc[part.patient_id.astype(str) != "", "patient_id"].nunique()),
        }
    s["strategy"] = str(split_df["split_strategy"].iloc[0])
    return s


def print_split_summary(split_df: pd.DataFrame) -> None:
    s = split_summary(split_df)
    print(f"\nData split strategy: {s['strategy']}  (seed={config.RANDOM_SEED})")
    for sp in SPLITS:
        d = s[sp]
        print(f"  {sp:5s}: {d['images']:6d} images ({d['fraction']:.1%})  "
              f"{d['per_class']}  patients={d['patients']}")
    print(f"Split saved to {config.SPLIT_PATH.relative_to(config.BASE_DIR)}")


def plot_split_distribution(split_df: pd.DataFrame) -> Path:
    table = split_df.groupby(["split", "class_name"]).size().unstack(fill_value=0) \
                    .reindex(index=list(SPLITS), columns=config.CLASS_NAMES, fill_value=0)
    fig = Figure(figsize=(7, 4.5))
    ax = fig.subplots()
    x = np.arange(len(SPLITS))
    w = 0.38
    colours = {"Uninfected": "#2a9d8f", "Parasitized": "#c0392b"}
    for i, c in enumerate(config.CLASS_NAMES):
        bars = ax.bar(x + (i - 0.5) * w, table[c].values, w, label=c, color=colours[c])
        ax.bar_label(bars, fontsize=9)
    ax.set_xticks(x, [s.capitalize() for s in SPLITS])
    ax.set_ylabel("Number of images")
    ax.set_title(f"Train / validation / test split ({split_df['split_strategy'].iloc[0]} grouping)")
    ax.legend()
    fig.tight_layout()
    out = config.GRAPHS_DIR / "split_distribution.png"
    fig.savefig(out, dpi=150)
    return out


# ---------------------------------------------------------------------------
# 3. Loading many images (with an on-disk cache)
# ---------------------------------------------------------------------------
def _load_one(rel_path: str, image_size: int) -> np.ndarray:
    return resize_uint8(load_image_rgb(config.BASE_DIR / rel_path), image_size)


def load_all_images(split_df: pd.DataFrame, image_size: int = config.IMAGE_SIZE,
                    rebuild_cache: bool = False, verbose: bool = True) -> np.ndarray:
    """
    Return uint8 array (N, H, W, 3) in the same row order as split_df.
    The resized images are cached in cache/images_<size>.npz; the cache is
    rebuilt automatically if the list of files changes.
    """
    config.ensure_dirs()
    cache = config.CACHE_DIR / f"images_{image_size}.npz"
    paths = split_df["rel_path"].to_numpy().astype(str)
    if cache.exists() and not rebuild_cache:
        data = np.load(cache, allow_pickle=False)
        if data["rel_paths"].shape == paths.shape and (data["rel_paths"] == paths).all():
            if verbose:
                print(f"Loaded {len(paths)} cached images from {cache.relative_to(config.BASE_DIR)}")
            return data["images"]
        if verbose:
            print("Cache does not match current split - rebuilding.")

    if verbose:
        print(f"Loading and resizing {len(paths)} images to {image_size}x{image_size} "
              f"(first time only, ~1-3 min) ...")
    with ThreadPoolExecutor(max_workers=8) as pool:
        images = np.stack(list(pool.map(lambda p: _load_one(p, image_size), paths)))
    np.savez(cache, images=images, rel_paths=paths)
    if verbose:
        print(f"Cached to {cache.relative_to(config.BASE_DIR)} ({images.nbytes / 1e6:.0f} MB)")
    return images


def get_split_arrays(split_df: pd.DataFrame, images: np.ndarray) -> dict:
    """{'train': (x_uint8, y), 'val': (...), 'test': (...)}"""
    out = {}
    for sp in SPLITS:
        mask = (split_df["split"] == sp).to_numpy()
        out[sp] = (images[mask], split_df.loc[mask, "label"].to_numpy().astype(np.float32))
    return out


# ---------------------------------------------------------------------------
# 4. TensorFlow input pipelines + augmentation
# ---------------------------------------------------------------------------
def make_tf_dataset(x_uint8: np.ndarray, y: np.ndarray, batch_size: int = config.BATCH_SIZE,
                    training: bool = False, seed: int = config.RANDOM_SEED):
    """
    Build a tf.data pipeline. Normalisation (/255) happens here exactly as in
    `normalize`. Augmentation is applied ONLY when training=True:
      random horizontal/vertical flips, random 90-degree rotations and a small
      brightness change. Cells have no natural "up" direction, so flips and
      rotations create realistic new examples without changing the label.
    """
    import tensorflow as tf

    def to_float(img, label):
        return tf.cast(img, tf.float32) / 255.0, label

    def augment(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.rot90(img, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
        img = tf.image.random_brightness(img, max_delta=0.05)
        return tf.clip_by_value(img, 0.0, 1.0), label

    ds = tf.data.Dataset.from_tensor_slices((x_uint8, y.reshape(-1, 1).astype(np.float32)))
    if training:
        ds = ds.shuffle(min(len(y), 10000), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(to_float, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# 5. Saved configuration needed for future predictions
# ---------------------------------------------------------------------------
def save_model_config(extra: dict | None = None) -> Path:
    cfg = {
        "class_names": config.CLASS_NAMES,
        "label_to_index": config.LABEL_TO_INDEX,
        "positive_class": config.POSITIVE_CLASS,
        "model_output": "sigmoid probability of the positive class (Parasitized)",
        "decision_threshold": config.DECISION_THRESHOLD,
        "image_size": config.IMAGE_SIZE,
        "channels": config.CHANNELS,
        "color_mode": "RGB",
        "resize_method": "PIL bilinear",
        "normalization": config.NORMALIZATION,
        "random_seed": config.RANDOM_SEED,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        cfg.update(extra)
    config.ensure_dirs()
    with open(config.MODEL_CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
    return config.MODEL_CONFIG_PATH


def load_model_config() -> dict:
    if not config.MODEL_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"{config.MODEL_CONFIG_PATH} not found. Train the model first: python -m src.train")
    with open(config.MODEL_CONFIG_PATH) as f:
        return json.load(f)


if __name__ == "__main__":
    # Create (or show) the split and verify that preprocessing works.
    sdf = create_splits(force="--resplit" in sys.argv, verbose=False)
    print_split_summary(sdf)
    sample = preprocess_image(config.BASE_DIR / sdf.iloc[0].rel_path)
    print(f"Sample preprocessed image: shape={sample.shape}, dtype={sample.dtype}, "
          f"range=[{sample.min():.3f}, {sample.max():.3f}]")

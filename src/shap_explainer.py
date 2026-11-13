"""
src/shap_explainer.py — SHAP explanations for the trained CNN.

SHAP (SHapley Additive exPlanations) gives every part of the image a number,
its Shapley value: how much that part pushed the prediction up or down compared
with a baseline. All the SHAP values add up to (prediction - baseline).

Two genuine SHAP methods are implemented:

1. "partition" (DEFAULT) — shap.PartitionExplainer with an Image masker.
   * Model-agnostic: it only calls the real CNN's predict function.
   * Hidden image regions are replaced by a BLURRED copy of the image
     (masker "blur(8,8)"), so blurring away fine detail such as a parasite
     shows how much that detail mattered.
   * Pixels are grouped hierarchically (a partition tree), so SHAP needs only
     SHAP_MAX_EVALS (default 500) model calls instead of an impossible 2^12288.
     On a 64x64 image with our small CNN this takes a few seconds on a CPU.

2. "gradient" (optional) — shap.GradientExplainer (expected gradients).
   * Uses the model's gradients and a BACKGROUND DATASET: 100 random training
     images saved during training (models/shap_background.npy).
   * Pixel-level detail and very fast, but depends on TensorFlow internals.

Command-line use:
    python -m src.shap_explainer path\\to\\cell.png [--method gradient]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src import preprocessing as prep  # noqa: E402


# ---------------------------------------------------------------------------
# Method 1: Partition explainer (default)
# ---------------------------------------------------------------------------
def shap_partition(predictor, image: np.ndarray,
                   max_evals: int = config.SHAP_MAX_EVALS,
                   batch_size: int = config.SHAP_BATCH_SIZE,
                   masker_spec: str = config.SHAP_MASKER) -> dict:
    import shap

    image = np.asarray(image, dtype=np.float32)
    masker = shap.maskers.Image(masker_spec, image.shape)
    explainer = shap.Explainer(predictor.predict_proba_batch, masker,
                               algorithm="partition", output_names=predictor.class_names)
    sv = explainer(image[np.newaxis], max_evals=max_evals, batch_size=batch_size, silent=True)

    values = np.asarray(sv.values)[0]            # (H, W, 3, n_classes)
    base = np.asarray(sv.base_values)[0]         # (n_classes,) model output on fully blurred image
    if values.ndim != 4 or values.shape[-1] != len(predictor.class_names):
        raise RuntimeError(f"Unexpected SHAP value shape {values.shape}")
    per_class_maps = values.sum(axis=2)          # sum colour channels -> (H, W, n_classes)
    return {"maps": per_class_maps, "base_values": base,
            "method": "partition",
            "details": f"PartitionExplainer, masker={masker_spec}, max_evals={max_evals}"}


# ---------------------------------------------------------------------------
# Method 2: Gradient explainer (optional)
# ---------------------------------------------------------------------------
def load_background(path: Path = config.SHAP_BACKGROUND_PATH) -> np.ndarray:
    if not Path(path).exists():
        raise FileNotFoundError(f"SHAP background set not found at {path}. "
                                "It is created by training: python -m src.train")
    return prep.normalize(np.load(path))


def shap_gradient(predictor, image: np.ndarray,
                  nsamples: int = config.SHAP_GRADIENT_NSAMPLES) -> dict:
    import shap

    background = load_background()
    explainer = shap.GradientExplainer(predictor.model, background)
    raw = explainer.shap_values(np.asarray(image, np.float32)[np.newaxis], nsamples=nsamples)
    arr = np.asarray(raw[0] if isinstance(raw, list) else raw, dtype=np.float32)
    arr = arr.reshape(1, *image.shape)[0]         # (H, W, 3): contributions to P(Parasitized)
    pos_map = arr.sum(axis=2)
    p_bg = float(predictor.model.predict(background, verbose=0).mean())
    pos = predictor.positive_index
    maps = np.zeros((*pos_map.shape, 2), dtype=np.float32)
    maps[..., pos] = pos_map
    maps[..., 1 - pos] = -pos_map                 # P(Uninfected) = 1 - P(Parasitized)
    base = np.zeros(2, dtype=np.float32)
    base[pos], base[1 - pos] = p_bg, 1 - p_bg
    return {"maps": maps, "base_values": base, "method": "gradient",
            "details": f"GradientExplainer, background={len(background)} training images, "
                       f"nsamples={nsamples}"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def explain_with_shap(predictor, image: np.ndarray, method: str = config.SHAP_DEFAULT_METHOD) -> dict:
    """
    predictor: src.prediction.MalariaPredictor;  image: float32 (H, W, 3) in [0, 1].
    Returns maps of SHAP values per class + an additivity check.
    """
    image = np.asarray(image, dtype=np.float32)
    if method == "partition":
        res = shap_partition(predictor, image)
    elif method == "gradient":
        res = shap_gradient(predictor, image)
    else:
        raise ValueError(f"Unknown SHAP method '{method}' (use 'partition' or 'gradient')")

    probs = predictor.predict_proba_batch(image)[0]
    label = int(np.argmax(probs))
    total = float(res["maps"][..., label].sum())
    res.update({
        "image": image,
        "probabilities": probs,
        "label_index": label,
        "label": predictor.class_names[label],
        "map_for_label": res["maps"][..., label],
        # Additivity: base value + sum of SHAP values should be close to the prediction
        "sum_shap": total,
        "base_value": float(res["base_values"][label]),
        "prediction": float(probs[label]),
        "additivity_gap": float(abs(res["base_values"][label] + total - probs[label])),
    })
    return res


def plot_shap(result: dict, title: str = "") -> Figure:
    """3-panel figure: original | SHAP overlay for predicted class | SHAP values with colour bar."""
    img = result["image"]
    m = result["map_for_label"]
    label = result["label"]
    lim = float(np.abs(m).max()) or 1e-9
    gray = img.mean(axis=2)

    fig = Figure(figsize=(12, 4))
    axes = fig.subplots(1, 3)
    axes[0].imshow(img)
    axes[0].set_title("Original (64x64 model input)")

    axes[1].imshow(gray, cmap="gray")
    axes[1].imshow(np.where(m > 0, m, np.nan), cmap="Reds", vmin=0, vmax=lim, alpha=0.8)
    axes[1].set_title(f"Regions pushing towards\n'{label}' (red)")

    axes[2].imshow(gray, cmap="gray", alpha=0.35)
    im = axes[2].imshow(m, cmap="RdBu_r", vmin=-lim, vmax=lim, alpha=0.8)
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label=f"SHAP value for '{label}'")
    axes[2].set_title("Red = increases, blue = decreases\nprobability of predicted class")

    for ax in axes:
        ax.axis("off")
    fig.suptitle(title or (f"SHAP ({result['method']}) - predicted {label}: "
                           f"base {result['base_value']:.3f} + sum(SHAP) {result['sum_shap']:+.3f} "
                           f"≈ prediction {result['prediction']:.3f}"))
    fig.tight_layout()
    return fig


def save_figure(fig: Figure, name: str) -> Path:
    config.ensure_dirs()
    out = config.EXPLANATIONS_DIR / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    return out


if __name__ == "__main__":
    import argparse
    from src.prediction import MalariaPredictor

    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--method", default=config.SHAP_DEFAULT_METHOD, choices=["partition", "gradient"])
    args = ap.parse_args()
    pred = MalariaPredictor()
    for path in args.images:
        res = explain_with_shap(pred, pred.preprocess(path), method=args.method)
        out = save_figure(plot_shap(res), f"shap_{args.method}_{Path(path).stem}.png")
        print(f"{path}: predicted {res['label']} | additivity gap {res['additivity_gap']:.4f} "
              f"-> SHAP figure saved to {out}")

"""
src/lime_explainer.py — LIME explanations for the trained CNN.

How LIME works here (Local Interpretable Model-agnostic Explanations):
  1. The cell image is cut into ~40 "superpixels" (small regions of similar
     colour) with the SLIC algorithm.
  2. LIME creates LIME_NUM_SAMPLES (default 1000) perturbed copies of the image,
     each with a random subset of superpixels "switched off" (replaced by
     their mean colour).
  3. The REAL trained CNN predicts every perturbed copy.
  4. LIME fits a simple weighted linear model: "which superpixels, when
     present, push the prediction towards Parasitized / Uninfected?"
  5. The linear model's weights are the explanation.

Command-line use:
    python -m src.lime_explainer path\\to\\cell.png
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from matplotlib.figure import Figure
from skimage.segmentation import mark_boundaries, slic

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def segment_image(image: np.ndarray, n_segments: int = config.LIME_N_SEGMENTS,
                  compactness: float = config.LIME_COMPACTNESS) -> np.ndarray:
    """SLIC superpixels. image: float (H, W, 3) in [0, 1]. Returns int labels (H, W) starting at 0."""
    return slic(image, n_segments=n_segments, compactness=compactness, sigma=1,
                start_label=0, channel_axis=-1)


def explain_with_lime(predictor, image: np.ndarray,
                      num_samples: int = config.LIME_NUM_SAMPLES,
                      num_features: int = config.LIME_NUM_FEATURES,
                      seed: int = config.RANDOM_SEED) -> dict:
    """
    predictor: a src.prediction.MalariaPredictor (holds the trained CNN)
    image:     preprocessed float32 (H, W, 3) in [0, 1]
    Returns a dict with the LIME explanation object, figure-ready arrays and numbers.
    """
    from lime import lime_image

    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"LIME expects an (H, W, 3) image, got shape {image.shape}")

    # Sanity check of the function LIME will call: (N,H,W,3) -> (N,2) probabilities
    probe = predictor.predict_proba_batch(image[np.newaxis])
    assert probe.shape == (1, 2) and np.isclose(probe.sum(), 1.0, atol=1e-4), probe

    explainer = lime_image.LimeImageExplainer(random_state=seed)
    explanation = explainer.explain_instance(
        image,
        classifier_fn=predictor.predict_proba_batch,   # the REAL CNN
        top_labels=2,
        hide_color=config.LIME_HIDE_COLOR,
        num_samples=num_samples,
        batch_size=100,
        segmentation_fn=segment_image,
        random_seed=seed,
     
    )

    probs = probe[0]
    label = int(np.argmax(probs))                       # explain the predicted class
    weights = dict(explanation.local_exp[label])        # {superpixel_id: weight}
    segments = explanation.segments
    heatmap = np.vectorize(lambda s: weights.get(s, 0.0))(segments).astype(np.float32)

    # Top-k superpixels that SUPPORT the predicted class
    _, mask_pos = explanation.get_image_and_mask(label, positive_only=True,
                                                 num_features=num_features, hide_rest=False)
    # Top-k superpixels by |weight|: supporting (+1) and opposing (-1)
    _, mask_both = explanation.get_image_and_mask(label, positive_only=False,
                                                  num_features=num_features, hide_rest=False)

    score = getattr(explanation, "score", None)
    score = score.get(label) if isinstance(score, dict) else score
    return {
        "explanation": explanation,
        "label_index": label,
        "label": predictor.class_names[label],
        "probabilities": probs,
        "image": image,
        "segments": segments,
        "n_segments": int(len(np.unique(segments))),
        "weights": weights,
        "heatmap": heatmap,
        "mask_positive": mask_pos,
        "mask_both": mask_both,
        "surrogate_r2": float(score) if score is not None else None,
        "num_samples": num_samples,
    }


def plot_lime(result: dict, title: str = "") -> Figure:
    """4-panel figure: original | supporting regions | support vs oppose | weight heatmap."""
    img = result["image"]
    label = result["label"]
    fig = Figure(figsize=(14, 4))
    axes = fig.subplots(1, 4)

    axes[0].imshow(img)
    axes[0].set_title("Original (64x64 model input)")

    axes[1].imshow(mark_boundaries(img, result["mask_positive"], color=(1, 1, 0), mode="thick"))
    axes[1].set_title(f"Top regions supporting\n'{label}' (yellow outline)")

    overlay = img.copy()
    both = result["mask_both"]
    overlay[both == 1] = 0.55 * overlay[both == 1] + 0.45 * np.array([0.0, 0.8, 0.0])
    overlay[both == -1] = 0.55 * overlay[both == -1] + 0.45 * np.array([0.9, 0.0, 0.0])
    axes[2].imshow(mark_boundaries(overlay, result["segments"], color=(1, 1, 1), mode="inner"))
    axes[2].set_title("Green = supports, red = opposes\n(white lines = superpixels)")

    lim = float(np.abs(result["heatmap"]).max()) or 1e-9
    axes[3].imshow(img, alpha=0.35)
    im = axes[3].imshow(result["heatmap"], cmap="RdBu_r", vmin=-lim, vmax=lim, alpha=0.75)
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04, label=f"LIME weight for '{label}'")
    axes[3].set_title("LIME superpixel weights")

    for ax in axes:
        ax.axis("off")
    p = result["probabilities"]
    r2 = result["surrogate_r2"]
    fig.suptitle(title or f"LIME explanation - predicted {label} "
                          f"(P(Parasitized)={p[1]:.3f}; {result['num_samples']} samples"
                          f"{'' if r2 is None else f'; surrogate R²={r2:.2f}'})")
    fig.tight_layout()
    return fig


def save_figure(fig: Figure, name: str) -> Path:
    config.ensure_dirs()
    out = config.EXPLANATIONS_DIR / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    return out


if __name__ == "__main__":
    from src.prediction import MalariaPredictor

    if len(sys.argv) < 2:
        print("Usage: python -m src.lime_explainer <image_path>")
        sys.exit(1)
    pred = MalariaPredictor()
    for path in sys.argv[1:]:
        res = explain_with_lime(pred, pred.preprocess(path))
        out = save_figure(plot_lime(res), f"lime_{Path(path).stem}.png")
        print(f"{path}: predicted {res['label']} -> LIME figure saved to {out}")

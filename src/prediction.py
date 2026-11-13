"""
src/prediction.py — load the trained model once and make predictions.

Used by the Streamlit app, LIME and SHAP. Command-line use:
    python -m src.prediction path\\to\\cell.png
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src import preprocessing as prep  # noqa: E402


class MalariaPredictor:
    """Wraps the saved CNN + its saved preprocessing configuration."""

    def __init__(self, model_path: Path = config.MODEL_PATH):
        from src.model import load_trained_model

        self.config = prep.load_model_config()
        self.class_names: list[str] = self.config["class_names"]
        self.image_size: int = int(self.config["image_size"])
        self.threshold: float = float(self.config["decision_threshold"])
        self.positive_index = self.class_names.index(self.config["positive_class"])
        self.model = load_trained_model(model_path)

    # -- preprocessing ----------------------------------------------------
    def preprocess(self, source) -> np.ndarray:
        """Any image source -> float32 (H, W, 3) in [0, 1], exactly as in training."""
        return prep.preprocess_image(source, self.image_size)

    # -- core prediction function (also given to LIME and SHAP) ------------
    def predict_proba_batch(self, images: np.ndarray) -> np.ndarray:
        """
        images: (N, H, W, 3) floats in [0, 1]  (a single (H, W, 3) image is also accepted)
        returns: (N, 2) probabilities, columns ordered as class_names
                 -> [:, 0] = P(Uninfected), [:, 1] = P(Parasitized)
        """
        images = np.asarray(images, dtype=np.float32)
        if images.ndim == 3:
            images = images[np.newaxis]
        if images.shape[1:3] != (self.image_size, self.image_size):
            raise ValueError(f"Expected images of size {self.image_size}x{self.image_size}, "
                             f"got {images.shape[1:3]}")
        p_pos = self.model.predict(images, batch_size=128, verbose=0).reshape(-1)
        probs = np.zeros((len(p_pos), 2), dtype=np.float32)
        probs[:, self.positive_index] = p_pos
        probs[:, 1 - self.positive_index] = 1.0 - p_pos
        return probs

    def predict(self, source) -> dict:
        """Full prediction for one image (path, bytes, uploaded file, PIL image or array)."""
        image = self.preprocess(source)
        probs = self.predict_proba_batch(image)[0]
        p_pos = float(probs[self.positive_index])
        label_idx = self.positive_index if p_pos >= self.threshold else 1 - self.positive_index
        return {
            "label": self.class_names[label_idx],
            "label_index": int(label_idx),
            "confidence": float(probs[label_idx]),
            "probabilities": {name: float(probs[i]) for i, name in enumerate(self.class_names)},
            "image": image,
        }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.prediction <image_path> [<image_path> ...]")
        sys.exit(1)
    predictor = MalariaPredictor()
    for path in sys.argv[1:]:
        r = predictor.predict(path)
        probs = ", ".join(f"{k}: {v:.4f}" for k, v in r["probabilities"].items())
        print(f"{path}\n  -> {r['label']} (confidence {r['confidence']:.2%})  [{probs}]")

"""
src/evaluate.py — measure how good the trained model really is.

Run from the project root (after training):
    python -m src.evaluate

Everything is computed from REAL predictions of the saved model:
  * Test set (never seen during training): accuracy, precision, recall
    (sensitivity), specificity, F1-score, ROC-AUC, confusion matrix,
    classification report, ROC curve, precision-recall curve.
  * Train and validation sets are also evaluated (without augmentation) so you
    can compare all three and spot overfitting.
  * Training/validation accuracy and loss curves from the training history.
Outputs go to outputs/reports/ and outputs/graphs/.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import (accuracy_score, auc, classification_report, confusion_matrix,
                             f1_score, log_loss, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, p_parasitized: np.ndarray,
                    threshold: float = config.DECISION_THRESHOLD) -> dict:
    """All classification metrics, Parasitized (label 1) = positive class."""
    y_true = y_true.astype(int)
    y_pred = (p_parasitized >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    m = {
        "n_images": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall_sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "log_loss": log_loss(y_true, np.clip(p_parasitized, 1e-7, 1 - 1e-7), labels=[0, 1]),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    }
    if len(np.unique(y_true)) == 2:
        m["roc_auc"] = roc_auc_score(y_true, p_parasitized)
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in m.items()}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, out: Path) -> Path:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig = Figure(figsize=(5.5, 4.8))
    ax = fig.subplots()
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    names = config.CLASS_NAMES
    ax.set_xticks([0, 1], [f"Predicted\n{n}" for n in names])
    ax.set_yticks([0, 1], [f"Actual\n{n}" for n in names])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({cm[i, j] / cm[i].sum():.1%})", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    return out


def plot_roc(y_true, p, out: Path) -> Path:
    fpr, tpr, _ = roc_curve(y_true, p)
    fig = Figure(figsize=(5.5, 5))
    ax = fig.subplots()
    ax.plot(fpr, tpr, color="#c0392b", lw=2, label=f"CNN (AUC = {auc(fpr, tpr):.4f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Random guess (AUC = 0.5)")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity / recall)")
    ax.set_title("ROC curve (test set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    return out


def plot_pr(y_true, p, out: Path) -> Path:
    prec, rec, _ = precision_recall_curve(y_true, p)
    fig = Figure(figsize=(5.5, 5))
    ax = fig.subplots()
    ax.plot(rec, prec, color="#2a9d8f", lw=2, label=f"CNN (AUC = {auc(rec, prec):.4f})")
    ax.set_xlabel("Recall (sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision-recall curve (test set)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    return out


def plot_misclassified(x_uint8, y_true, p, out: Path, n: int = 12) -> Path | None:
    y_pred = (p >= config.DECISION_THRESHOLD).astype(int)
    wrong = np.where(y_pred != y_true.astype(int))[0]
    if len(wrong) == 0:
        return None
    wrong = wrong[np.argsort(-np.abs(p[wrong] - 0.5))][:n]   # most confident mistakes first
    cols = min(6, len(wrong))
    rows = int(np.ceil(len(wrong) / cols))
    fig = Figure(figsize=(cols * 2, rows * 2.4))
    axes = np.atleast_1d(fig.subplots(rows, cols)).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, i in zip(axes, wrong):
        ax.imshow(x_uint8[i])
        ax.set_title(f"true: {config.CLASS_NAMES[int(y_true[i])]}\n"
                     f"P(Parasitized)={p[i]:.2f}", fontsize=8)
    fig.suptitle("Misclassified test images (most confident errors)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    return out


def plot_training_history() -> list[Path]:
    """Accuracy and loss curves (train vs validation) from training_history.csv."""
    if not config.HISTORY_PATH.exists():
        return []
    h = pd.read_csv(config.HISTORY_PATH)
    epochs = h["epoch"] + 1
    outs = []
    for metric, title in [("accuracy", "Accuracy"), ("loss", "Loss")]:
        fig = Figure(figsize=(6.5, 4.5))
        ax = fig.subplots()
        ax.plot(epochs, h[metric], "o-", label=f"Training {title.lower()}", color="#264653")
        ax.plot(epochs, h[f"val_{metric}"], "s-", label=f"Validation {title.lower()}", color="#e76f51")
        best = int(h["val_loss"].idxmin())
        ax.axvline(epochs[best], ls="--", color="grey", label=f"Best epoch ({epochs[best]})")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Epoch"); ax.set_ylabel(title)
        ax.set_title(f"Training vs validation {title.lower()}")
        ax.legend()
        fig.tight_layout()
        out = config.GRAPHS_DIR / f"training_{metric}.png"
        fig.savefig(out, dpi=150)
        outs.append(out)
    return outs


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def evaluate() -> dict:
    from src import preprocessing as prep
    from src.model import load_trained_model

    config.ensure_dirs()
    model = load_trained_model()
    split_df = prep.create_splits()
    images = prep.load_all_images(split_df, config.IMAGE_SIZE)
    arrays = prep.get_split_arrays(split_df, images)

    results = {"threshold": config.DECISION_THRESHOLD,
               "split_strategy": str(split_df["split_strategy"].iloc[0]),
               "positive_class": config.POSITIVE_CLASS}
    probs = {}
    for sp in prep.SPLITS:
        x, y = arrays[sp]
        p = model.predict(prep.normalize(x), batch_size=256, verbose=0).ravel()
        probs[sp] = p
        results[sp] = compute_metrics(y, p)
        print(f"{sp:5s}: accuracy={results[sp]['accuracy']:.4f}  "
              f"F1={results[sp]['f1_score']:.4f}  AUC={results[sp].get('roc_auc', float('nan')):.4f}")

    x_te, y_te = arrays["test"]
    p_te = probs["test"]
    y_pred = (p_te >= config.DECISION_THRESHOLD).astype(int)

    report_txt = classification_report(y_te.astype(int), y_pred, target_names=config.CLASS_NAMES,
                                       digits=4)
    results["classification_report"] = classification_report(
        y_te.astype(int), y_pred, target_names=config.CLASS_NAMES, output_dict=True)

    test_rows = split_df[split_df.split == "test"].reset_index(drop=True)
    pd.DataFrame({
        "rel_path": test_rows.rel_path, "true_class": test_rows.class_name,
        "p_parasitized": np.round(p_te, 6),
        "predicted_class": [config.CLASS_NAMES[i] for i in y_pred],
        "correct": y_pred == y_te.astype(int),
    }).to_csv(config.TEST_PREDICTIONS_PATH, index=False)

    graphs = [plot_confusion_matrix(y_te.astype(int), y_pred, config.GRAPHS_DIR / "confusion_matrix.png"),
              plot_roc(y_te.astype(int), p_te, config.GRAPHS_DIR / "roc_curve.png"),
              plot_pr(y_te.astype(int), p_te, config.GRAPHS_DIR / "precision_recall_curve.png"),
              plot_misclassified(x_te, y_te, p_te, config.GRAPHS_DIR / "misclassified_examples.png")]
    graphs += plot_training_history()
    results["graphs"] = [str(g.relative_to(config.BASE_DIR)) for g in graphs if g is not None]

    with open(config.METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    with open(config.CLASSIFICATION_REPORT_PATH, "w") as f:
        f.write("Classification report - TEST set (held-out patients)\n")
        f.write(f"Decision threshold: P(Parasitized) >= {config.DECISION_THRESHOLD}\n\n")
        f.write(report_txt)
        t = results["test"]
        f.write(f"\nSpecificity: {t['specificity']:.4f}\nROC-AUC: {t.get('roc_auc', float('nan')):.4f}\n")
        f.write(f"Confusion matrix: {t['confusion_matrix']}\n")

    print("\n================ TEST SET RESULTS ================")
    print(report_txt)
    t = results["test"]
    print(f"Specificity: {t['specificity']:.4f}   ROC-AUC: {t.get('roc_auc', float('nan')):.4f}")
    print(f"Confusion matrix: {t['confusion_matrix']}")
    print(f"Saved: {config.METRICS_PATH.relative_to(config.BASE_DIR)}, "
          f"{config.CLASSIFICATION_REPORT_PATH.relative_to(config.BASE_DIR)}, graphs in outputs/graphs/")
    return results


if __name__ == "__main__":
    evaluate()

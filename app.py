"""
app.py — Streamlit web application for AI-Based Malaria Detection.

Run from the project root (with the virtual environment activated):
    streamlit run app.py

Pages: Home | Image Classification | Explainable AI | Model Performance | About
The app only LOADS the trained model (models/malaria_cnn.keras); it never retrains.
All ML logic lives in the src/ modules; this file is only the user interface.
"""

from __future__ import annotations

import hashlib
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import streamlit as st

import config
from src import preprocessing as prep

st.set_page_config(page_title="Malaria Detection | CNN + LIME + SHAP", page_icon="🔬", layout="wide")

DISCLAIMER = ("⚠️ **Educational research prototype.** This tool is NOT a medical device and must not "
              "be used for diagnosis. Malaria diagnosis must be made by trained professionals using "
              "approved methods (microscopy, rapid diagnostic tests, PCR).")
PAGES = ["🏠 Home", "🔬 Image Classification", "🧠 Explainable AI", "📊 Model Performance", "ℹ️ About"]


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained CNN ...")
def get_predictor():
    from src.prediction import MalariaPredictor
    return MalariaPredictor()


def load_predictor_or_explain():
    """Return the predictor, or show a helpful error and return None."""
    try:
        return get_predictor()
    except FileNotFoundError as e:
        st.error(f"**Trained model not found.**\n\n{e}\n\nOpen a terminal in the `malaria` folder and run:"
                 "\n```\npython -m src.dataset_inspection\npython -m src.train\npython -m src.evaluate\n```")
    except Exception as e:  # e.g. TensorFlow not installed or incompatible model file
        st.error(f"Could not load the model: {type(e).__name__}: {e}")
    return None


@st.cache_data
def test_split_images() -> list[str]:
    """Paths of held-out TEST images (never seen in training), if the dataset is present."""
    if not config.SPLIT_PATH.exists():
        return []
    df = pd.read_csv(config.SPLIT_PATH)
    df = df[df.split == "test"]
    return [p for p in df.rel_path.tolist() if (config.BASE_DIR / p).exists()]


def image_key(image: np.ndarray) -> str:
    return hashlib.md5(image.tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Image input (shared by Classification and XAI pages)
# ---------------------------------------------------------------------------
def image_input_widget(predictor) -> bool:
    """Upload widget + 'random test image' button. Stores the image in session_state."""
    col_up, col_sample = st.columns([3, 1])
    with col_up:
        uploaded = st.file_uploader("Upload a microscopic image of a SINGLE red blood cell",
                                    type=sorted(e.strip(".") for e in config.VALID_EXTENSIONS))
    # A new upload replaces the current image (detected by a unique id per upload)
    if uploaded is not None:
        uid = str(getattr(uploaded, "file_id", None) or (uploaded.name, uploaded.size))
        if st.session_state.get("uploaded_id") != uid:
            st.session_state["img_bytes"] = uploaded.getvalue()
            st.session_state["img_name"] = uploaded.name
            st.session_state["uploaded_id"] = uid

    with col_sample:
        st.write("")
        st.write("")
        samples = test_split_images()
        if st.button("🎲 Use a random test image", disabled=not samples,
                     help="Picks a random image from the local test dataset."):
            rel = samples[np.random.randint(len(samples))]
            st.session_state["img_bytes"] = (config.BASE_DIR / rel).read_bytes()
            st.session_state["img_name"] = rel

    if "img_bytes" not in st.session_state:
        return False
    try:
        original = prep.load_image_rgb(st.session_state["img_bytes"])
        model_input = predictor.preprocess(original)
    except ValueError as e:
        st.error(f"❌ This file could not be read as an image. {e}")
        st.session_state.pop("img_bytes", None)
        return False
    st.session_state["original"] = original
    st.session_state["model_input"] = model_input
    w, h = original.size
    if max(w, h) > 400:
        st.warning(f"This image is {w}x{h} px. The model was trained on single-cell crops "
                   f"(about 100-200 px). Whole blood-smear photos will give unreliable results.")
    return True


def show_prediction(predictor) -> dict:
    result = predictor.predict(st.session_state["model_input"])
    p_par = result["probabilities"]["Parasitized"]
    p_uni = result["probabilities"]["Uninfected"]
    if result["label"] == "Parasitized":
        st.error(f"### 🦠 Prediction: **Parasitized**  \nConfidence: **{result['confidence']:.2%}**")
    else:
        st.success(f"### ✅ Prediction: **Uninfected**  \nConfidence: **{result['confidence']:.2%}**")
    c1, c2 = st.columns(2)
    c1.metric("P(Parasitized)", f"{p_par:.2%}")
    c2.metric("P(Uninfected)", f"{p_uni:.2%}")
    st.progress(float(p_par), text=f"Probability of Parasitized: {p_par:.2%}")
    if 0.35 < p_par < 0.65:
        st.info("The model is uncertain about this cell (probability close to 50%).")
    return result


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_home():
    st.title("🔬 AI-Based Malaria Detection")
    st.subheader("Convolutional Neural Network with LIME and SHAP Explainability")
    st.markdown("""
Malaria is caused by *Plasmodium* parasites that infect red blood cells. A standard diagnosis is
made by a trained microscopist examining a stained blood smear, which is slow and needs expertise.
This project trains a **Convolutional Neural Network (CNN)** to classify microscopic images of single
red blood cells as **Parasitized** or **Uninfected**, and uses **Explainable AI** to show *why* the
model made each prediction.

#### Objectives
1. Inspect and prepare the NIH Malaria Cell Images dataset (27,558 cell images).
2. Train a CNN that classifies cells as Parasitized or Uninfected.
3. Evaluate it on an image-level random test split using accuracy, precision, recall, F1, ROC-AUC and a confusion matrix.
4. Explain individual predictions with **LIME** and **SHAP**.
5. Provide this interactive application for upload, prediction and explanation.

#### How the system works
`Cell image → preprocessing (RGB, 64×64, ÷255) → CNN → P(Parasitized) → prediction`
`                                      ↘ LIME & SHAP → highlighted image regions`
""")
    arch = config.BASE_DIR / "docs" / "architecture_diagram.png"
    if arch.exists():
        st.image(str(arch), caption="System architecture", use_column_width=True)
    st.markdown("Use the sidebar to classify an image, view explanations or see model performance.")


def page_classify():
    st.title("🔬 Image Classification")
    predictor = load_predictor_or_explain()
    if predictor is None or not image_input_widget(predictor):
        return
    c1, c2 = st.columns([1, 2])
    with c1:
        st.image(st.session_state["original"], caption=f"Uploaded: {st.session_state.get('img_name', '')}",
                 use_column_width=True)
        st.image((st.session_state["model_input"] * 255).astype(np.uint8),
                 caption=f"What the model sees ({predictor.image_size}×{predictor.image_size})", width=160)
    with c2:
        try:
            show_prediction(predictor)
        except Exception as e:
            st.error(f"Prediction failed: {type(e).__name__}: {e}")
            return
        st.markdown("➡️ Open **🧠 Explainable AI** in the sidebar to see *why* the model decided this.")


def page_xai():
    st.title("🧠 Explainable AI: LIME and SHAP")
    predictor = load_predictor_or_explain()
    if predictor is None or not image_input_widget(predictor):
        st.info("Upload an image (or pick a random test image) to explain its prediction.")
        return
    img = st.session_state["model_input"]
    key = image_key(img)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.image(st.session_state["original"], caption="Original image", use_column_width=True)
    with c2:
        show_prediction(predictor)

    with st.expander("⚙️ Explanation settings"):
        lime_samples = st.slider("LIME: number of perturbed samples", 200, 3000,
                                 config.LIME_NUM_SAMPLES, step=100)
        lime_features = st.slider("LIME: superpixels to highlight", 2, 15, config.LIME_NUM_FEATURES)
        shap_method = st.radio("SHAP method", ["partition"], horizontal=True,
                               help="partition = model-agnostic, blur masker (default). "
                                    )

    tab_lime, tab_shap = st.tabs(["🟩 LIME", "🟥 SHAP"])

    with tab_lime:
        lkey = f"lime_{key}_{lime_samples}_{lime_features}"
        if st.button("Generate LIME explanation", type="primary") or lkey in st.session_state:
            if lkey not in st.session_state:
                from src.lime_explainer import explain_with_lime, plot_lime, save_figure
                try:
                    with st.spinner(f"Running LIME ({lime_samples} model predictions) ..."):
                        res = explain_with_lime(predictor, img, num_samples=lime_samples,
                                                num_features=lime_features)
                        fig = plot_lime(res)
                        saved = save_figure(fig, f"lime_{key[:10]}.png")
                    st.session_state[lkey] = (fig, res["n_segments"], res["surrogate_r2"], saved)
                except Exception as e:
                    st.error(f"LIME failed: {type(e).__name__}: {e}")
                    st.exception(e)
            if lkey in st.session_state:
                fig, nseg, r2, saved = st.session_state[lkey]
                st.pyplot(fig)
                st.caption(f"{nseg} superpixels · surrogate model R² = "
                           f"{'n/a' if r2 is None else f'{r2:.2f}'} · saved to "
                           f"{saved.relative_to(config.BASE_DIR)}")
        st.markdown("""
**How to read LIME:** the cell is split into superpixels (small regions). LIME hides random
combinations of them, asks the CNN to predict each version, and fits a simple linear model.
* **Yellow outline / green**: regions that *support* the predicted class.
* **Red**: regions that *oppose* it.
* For a Parasitized prediction, a trustworthy explanation usually highlights the purple-stained
  parasite inside the cell, not the black background.
""")

    with tab_shap:
        skey = f"shap_{key}_{shap_method}"
        if st.button("Generate SHAP explanation", type="primary") or skey in st.session_state:
            if skey not in st.session_state:
                from src.shap_explainer import explain_with_shap, plot_shap, save_figure as save_shap
                try:
                    with st.spinner(f"Running SHAP ({shap_method}) ..."):
                        res = explain_with_shap(predictor, img, method=shap_method)
                        fig = plot_shap(res)
                        saved = save_shap(fig, f"shap_{shap_method}_{key[:10]}.png")
                    st.session_state[skey] = (fig, res, saved)
                except Exception as e:
                    st.error(f"SHAP ({shap_method}) failed: {type(e).__name__}: {e}")
                    st.exception(e)
            if skey in st.session_state:
                fig, res, saved = st.session_state[skey]
                st.pyplot(fig)
                st.caption(f"{res['details']} · base value {res['base_value']:.3f} + "
                           f"sum of SHAP values {res['sum_shap']:+.3f} vs prediction "
                           f"{res['prediction']:.3f} (gap {res['additivity_gap']:.3f}) · saved to "
                           f"{saved.relative_to(config.BASE_DIR)}")
        st.markdown("""
**How to read SHAP:** each pixel region gets a Shapley value, its fair share of the difference
between the model's prediction and a baseline (for PartitionExplainer: the prediction on a blurred copy of the image)
* **Red**: increases the probability of the predicted class. **Blue**: decreases it.
* Base value + all SHAP values ≈ the model's prediction (additivity).
""")
    st.warning("Explanations show what the **model** relied on. They do not prove the prediction is "
               "medically correct.")



def page_performance():
    st.title("📊 Model Performance")

    # Load the actual evaluation results
    metrics_path = config.REPORTS_DIR / "evaluation_results.json"

    if not metrics_path.exists():
        st.error("Evaluation results file not found.")
        return

    with open(metrics_path) as f:
        m = json.load(f)

    st.markdown("### Test Set Performance")
    st.caption("Results from the trained CNN model's test-set evaluation.")

    # Display actual metrics
    cols = st.columns(5)

    metrics = [
        ("Accuracy", "test_accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1-score", "f1_score"),
        ("ROC-AUC", "roc_auc")
    ]

    for col, (name, key) in zip(cols, metrics):
        col.metric(name, f"{m[key]:.4f}")

    st.divider()

    st.info(
        "These results were obtained using an image-level random split. "
        "They should not be interpreted as patient-independent evaluation."
    )

    # Display available performance graphs
    st.subheader("Performance Graphs")

    g = config.GRAPHS_DIR

    for name, caption in [
        ("confusion_matrix.png", "Confusion Matrix"),
        ("roc_curve.png", "ROC Curve"),
        ("training_accuracy.png", "Training vs Validation Accuracy"),
        ("training_loss.png", "Training vs Validation Loss"),
        ("precision_recall_curve.png", "Precision-Recall Curve"),
        ("misclassified_examples.png", "Misclassified Test Images")
    ]:
        if (g / name).exists():
            st.image(str(g / name), caption=caption, use_container_width=True)

    # Display classification report if available
    if config.CLASSIFICATION_REPORT_PATH.exists():
        st.subheader("Classification Report")
        st.code(
            config.CLASSIFICATION_REPORT_PATH.read_text(),
            language="text"
        )

    # Display available EDA graphs
    with st.expander("Dataset Graphs (EDA)"):
        for name in [
            "class_distribution.png",
            "split_distribution.png",
            "sample_images.png",
            "image_dimensions.png",
            "colour_intensity.png",
            "cells_per_patient.png"
        ]:
            if (g / name).exists():
                st.image(str(g / name), use_container_width=True)


def page_about():
    st.title("ℹ️ About the Project")
    st.markdown("""
### Dataset
**NIH Malaria Cell Images Dataset** (Lister Hill National Center for Biomedical Communications,
U.S. National Library of Medicine). 27,558 single red blood cell images, 13,779 Parasitized and
13,779 Uninfected, cropped from Giemsa-stained thin blood smear photos taken with a smartphone
on a microscope at Chittagong Medical College Hospital, Bangladesh (150 *P. falciparum*-infected
and 50 healthy patients). Reference: Rajaraman S. et al. (2018), *PeerJ* 6:e4568.

### Technologies
Python · TensorFlow/Keras (CNN) · scikit-learn (metrics, split) · LIME · SHAP · Pillow ·
scikit-image · Matplotlib · Streamlit

### CNN
A Convolutional Neural Network learns small filters that detect patterns such as edges, stained
dots and textures. Three convolution blocks (32→64→128 filters, each with batch normalisation, ReLU,
max-pooling and dropout), followed by global average pooling, a 128-unit dense layer and a sigmoid
output giving P(Parasitized). About 110,785 parameters, small enough to train on a laptop CPU.

### LIME (Local Interpretable Model-agnostic Explanations)
Splits the image into superpixels, hides random subsets, records how the CNN's prediction changes,
and fits a simple linear model whose weights show which regions mattered for **this** prediction.

### SHAP (SHapley Additive exPlanations)
Based on Shapley values from cooperative game theory: each image region receives its fair share of
the prediction. Values are additive: baseline + sum of SHAP values = prediction.

### Limitations
* Educational prototype, **not** a diagnostic device. High accuracy on this dataset does not mean clinical reliability.
* All images come from one hospital, one staining protocol and one imaging setup; performance on other labs, microscopes or stains is unknown.
* Only *P. falciparum* vs uninfected; other species and parasite stages are not distinguished.
* Works on pre-segmented single cells, not whole blood-smear images; no parasite counting (parasitaemia).
* Some labels in the public dataset are known to be noisy.
* LIME and SHAP explain the model's behaviour; they do not prove that a prediction is correct, and they can vary with settings (number of samples, segmentation, baseline).
""")


# ---------------------------------------------------------------------------
def main():
    st.sidebar.title("Malaria Detection")
    page = st.sidebar.radio("Navigate", PAGES)
    st.sidebar.markdown("---")
    st.sidebar.warning(DISCLAIMER)
    {PAGES[0]: page_home, PAGES[1]: page_classify, PAGES[2]: page_xai,
     PAGES[3]: page_performance, PAGES[4]: page_about}[page]()


if __name__ == "__main__":
    main()

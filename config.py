"""
config.py — central configuration for the Malaria Detection project.

Every other module imports its paths and settings from here, so if you want
to change something (image size, epochs, batch size, LIME/SHAP settings),
change it in ONE place: this file.

All paths are resolved relative to this file's folder (the project root),
so the project works no matter where the `malaria` folder lives on your PC.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# 1. PATHS
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent          # ...\malaria
DATASET_PATH = BASE_DIR / "cell_images"              # your existing dataset (not moved/renamed)

# Optional official NIH metadata (patient-ID -> cell filename mapping).
# If you place these two small CSV files in malaria\metadata\, they are used
# for patient-level splitting. If they are absent, the NIH-documented filename
# convention is used instead (see README, "Patient-level split").
METADATA_DIR = BASE_DIR / "metadata"
PATIENT_MAPPING_FILES = {
    "Parasitized": METADATA_DIR / "patientid_cellmapping_parasitized.csv",
    "Uninfected": METADATA_DIR / "patientid_cellmapping_uninfected.csv",
}

MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
GRAPHS_DIR = OUTPUTS_DIR / "graphs"
EXPLANATIONS_DIR = OUTPUTS_DIR / "explanations"
REPORTS_DIR = OUTPUTS_DIR / "reports"
CACHE_DIR = BASE_DIR / "cache"                       # preprocessed image arrays (safe to delete)

MODEL_PATH = MODELS_DIR / "malaria_cnn.keras"        # trained CNN
MODEL_CONFIG_PATH = MODELS_DIR / "model_config.json" # class mapping + preprocessing settings
SHAP_BACKGROUND_PATH = MODELS_DIR / "shap_background.npy"  # background images for SHAP GradientExplainer

INVENTORY_PATH = REPORTS_DIR / "dataset_inventory.csv"     # one row per image file
DATASET_REPORT_PATH = REPORTS_DIR / "dataset_report.json"
SPLIT_PATH = REPORTS_DIR / "data_split.csv"                # which image is in train/val/test
HISTORY_PATH = REPORTS_DIR / "training_history.csv"
TRAINING_SUMMARY_PATH = REPORTS_DIR / "training_summary.json"
METRICS_PATH = REPORTS_DIR / "metrics.json"
CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "classification_report.txt"
TEST_PREDICTIONS_PATH = REPORTS_DIR / "test_predictions.csv"

# ---------------------------------------------------------------------------
# 2. CLASSES
# ---------------------------------------------------------------------------
# Index in this list = numeric label. Parasitized = 1 is the "positive" class
# (the condition we want to detect), which is the usual medical convention.
CLASS_NAMES = ["Uninfected", "Parasitized"]
POSITIVE_CLASS = "Parasitized"
LABEL_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}
DECISION_THRESHOLD = 0.5   # P(Parasitized) >= 0.5  ->  "Parasitized"

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# ---------------------------------------------------------------------------
# 3. PREPROCESSING
# ---------------------------------------------------------------------------
IMAGE_SIZE = 64            # images are resized to IMAGE_SIZE x IMAGE_SIZE pixels
CHANNELS = 3               # RGB
NORMALIZATION = "divide_by_255"   # pixel values 0..255 -> 0.0..1.0

# Data split (fractions of the whole dataset). Splits are made by patient groups.
TEST_FRACTION = 0.15
VAL_FRACTION = 0.15
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# 4. TRAINING
# ---------------------------------------------------------------------------
EPOCHS = 30                # maximum; early stopping usually stops sooner
BATCH_SIZE = 64            # lower to 32 if you run out of memory
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 6
REDUCE_LR_PATIENCE = 3
L2_WEIGHT = 1e-4           # L2 regularization strength
DROPOUT_CONV = 0.2
DROPOUT_DENSE = 0.5
SHAP_BACKGROUND_SIZE = 100 # training images saved as SHAP background set

# ---------------------------------------------------------------------------
# 5. EXPLAINABILITY
# ---------------------------------------------------------------------------
# LIME
LIME_NUM_SAMPLES = 1000    # perturbed images LIME generates (more = slower, more stable)
LIME_NUM_FEATURES = 6      # top superpixels to highlight
LIME_N_SEGMENTS = 40       # target number of SLIC superpixels
LIME_COMPACTNESS = 10
LIME_HIDE_COLOR = None     # None = hidden superpixel replaced by its mean colour

# SHAP
SHAP_DEFAULT_METHOD = "partition"   # "partition" (default) or "gradient"
SHAP_MAX_EVALS = 500       # model evaluations for the Partition explainer
SHAP_BATCH_SIZE = 50
SHAP_MASKER = "blur(8,8)"  # hidden regions are replaced by a blurred copy of the image
SHAP_GRADIENT_NSAMPLES = 200

ALL_OUTPUT_DIRS = [MODELS_DIR, GRAPHS_DIR, EXPLANATIONS_DIR, REPORTS_DIR, CACHE_DIR]


def ensure_dirs() -> None:
    """Create the output folders if they don't exist yet."""
    for d in ALL_OUTPUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)

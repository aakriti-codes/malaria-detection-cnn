"""
src/dataset_inspection.py — inspect, validate and visualise the dataset (EDA).

What it does
------------
1. Checks that cell_images/Parasitized and cell_images/Uninfected exist.
2. Opens EVERY file, records its size, colour mode, format and average colour,
   and flags corrupted/unreadable files and non-image files (e.g. Thumbs.db).
3. Finds exact duplicate images (same file content, via an MD5 hash).
4. Works out which PATIENT each cell came from (needed for a leakage-free split):
      a) official NIH patient-to-cell CSV files, if present in malaria/metadata/
      b) otherwise the NIH-documented filename convention
         ("C33P1thinF_IMG_..._cell_179.png" -> patient "P1").
5. Checks class balance and that images load correctly through the same
   preprocessing function used for training and prediction.
6. Saves graphs to outputs/graphs/ and reports to outputs/reports/.

Run from the project root (the malaria folder):
    python -m src.dataset_inspection
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from PIL import Image

# Allow "python src/dataset_inspection.py" as well as "python -m src.dataset_inspection"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# ---------------------------------------------------------------------------
# Patient identification
# ---------------------------------------------------------------------------
# NIH (LHNCBC) states that the patient ID is encoded in the cell name, e.g.
# "P1" in "C33P1thinF_IMG_20150619_114756a_cell_179.png".
_PATIENT_RE = re.compile(r"^C\d+P(\d+)", re.IGNORECASE)


def load_official_patient_mapping() -> dict[str, str]:
    """
    Read NIH's patient-to-cell CSV files if the user placed them in metadata/.
    Each line looks like:  C100P61ThinF,['C100P61ThinF_IMG_..._cell_128.png', ...]
    Returns {cell_filename: patient_identifier}. Empty dict if files are absent.
    """
    mapping: dict[str, str] = {}
    for class_name, path in config.PATIENT_MAPPING_FILES.items():
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or "," not in line:
                    continue
                identifier = line.split(",", 1)[0].strip()
                start, end = line.find("["), line.rfind("]")
                files: list[str] = []
                if start != -1 and end > start:
                    try:
                        files = list(ast.literal_eval(line[start:end + 1]))
                    except (ValueError, SyntaxError):
                        files = re.findall(r"'([^']+?\.png)'", line)
                patient = normalise_patient_id(identifier)
                for fname in files:
                    mapping[Path(fname).name] = patient
    return mapping


def normalise_patient_id(identifier: str) -> str:
    """'C100P61ThinF' -> 'P61'.  Identifiers without a P-number are kept (lower-cased)."""
    m = _PATIENT_RE.match(identifier)
    if m:
        return f"P{int(m.group(1))}"
    return identifier.lower()


def patient_from_filename(filename: str) -> str | None:
    """
    Apply the NIH-documented naming convention.
    'C100P61ThinF_IMG_20150918_144104_cell_162.png' -> 'P61'
    Files from slides without a P-number: the slide prefix before '_IMG_' is
    used (e.g. 'c1_thinf'). Returns None if the name doesn't follow the pattern.
    """
    m = _PATIENT_RE.match(filename)
    if m:
        return f"P{int(m.group(1))}"
    if "_IMG_" in filename:
        return filename.split("_IMG_")[0].lower()
    return None


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------
def check_class_folders() -> dict[str, bool]:
    """Return {class_name: folder_exists}."""
    return {c: (config.DATASET_PATH / c).is_dir() for c in config.CLASS_NAMES}


def scan_dataset(verbose: bool = True) -> pd.DataFrame:
    """
    Open every file in both class folders and build an inventory table.
    One row per file with: path, class, label, validity, size, mode, md5, colour stats.
    """
    folders = check_class_folders()
    missing = [c for c, ok in folders.items() if not ok]
    if missing:
        raise FileNotFoundError(
            f"Missing class folder(s): {missing}. Expected them inside {config.DATASET_PATH}"
        )

    rows = []
    t0 = time.time()
    for class_name in config.CLASS_NAMES:
        files = sorted(p for p in (config.DATASET_PATH / class_name).iterdir() if p.is_file())
        if verbose:
            print(f"  Scanning {class_name}: {len(files)} files ...")
        for i, path in enumerate(files, 1):
            row = {
                "rel_path": path.relative_to(config.BASE_DIR).as_posix(),
                "filename": path.name,
                "class_name": class_name,
                "label": config.LABEL_TO_INDEX[class_name],
                "extension": path.suffix.lower(),
                "file_bytes": path.stat().st_size,
                "is_image_ext": path.suffix.lower() in config.VALID_EXTENSIONS,
                "valid": False,
                "error": "",
                "width": np.nan, "height": np.nan, "mode": "", "format": "",
                "mean_r": np.nan, "mean_g": np.nan, "mean_b": np.nan,
                "md5": "",
            }
            if not row["is_image_ext"]:
                row["error"] = "not an image file"
            else:
                try:
                    data = path.read_bytes()
                    row["md5"] = hashlib.md5(data).hexdigest()
                    with Image.open(path) as img:
                        img.verify()                      # detects truncated/corrupt files
                    with Image.open(path) as img:          # re-open: verify() invalidates the object
                        row["format"] = img.format or ""
                        row["mode"] = img.mode
                        row["width"], row["height"] = img.size
                        arr = np.asarray(img.convert("RGB"), dtype=np.float32)
                    row["mean_r"], row["mean_g"], row["mean_b"] = arr.reshape(-1, 3).mean(axis=0)
                    row["valid"] = True
                except Exception as exc:  # unreadable / corrupted
                    row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            if verbose and i % 3000 == 0:
                print(f"    {i}/{len(files)} ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows)

    # ---- patient grouping ----
    official = load_official_patient_mapping()
    patients, sources = [], []
    for fname in df["filename"]:
        if fname in official:
            patients.append(official[fname]); sources.append("official_csv")
        else:
            pid = patient_from_filename(fname)
            patients.append(pid if pid else ""); sources.append("filename" if pid else "none")
    df["patient_id"] = patients
    df["patient_source"] = sources
    return df


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def summarise(df: pd.DataFrame) -> dict:
    valid = df[df["valid"]]
    counts = valid["class_name"].value_counts().reindex(config.CLASS_NAMES, fill_value=0)
    total = int(counts.sum())

    # duplicates (identical bytes)
    dup = valid[valid["md5"].duplicated(keep=False)]
    dup_groups = dup.groupby("md5")
    cross_class = int(sum(g["class_name"].nunique() > 1 for _, g in dup_groups))

    # patient grouping coverage
    with_pid = valid[valid["patient_id"] != ""]
    patients_per_class = with_pid.groupby("class_name")["patient_id"].nunique().to_dict()
    pids_both = set(with_pid[with_pid.class_name == "Parasitized"].patient_id) & set(
        with_pid[with_pid.class_name == "Uninfected"].patient_id)

    minority, majority = counts.min(), counts.max()
    report = {
        "dataset_path": str(config.DATASET_PATH),
        "class_folders_found": check_class_folders(),
        "total_files": int(len(df)),
        "non_image_files": df.loc[~df.is_image_ext, "rel_path"].tolist(),
        "corrupted_or_unreadable": df.loc[df.is_image_ext & ~df.valid, ["rel_path", "error"]]
                                     .to_dict("records"),
        "valid_images_per_class": {k: int(v) for k, v in counts.items()},
        "total_valid_images": total,
        "class_balance_ratio_min_over_max": round(float(minority / majority), 4) if majority else None,
        "is_imbalanced": bool(majority and minority / majority < 0.8),
        "extensions": valid["extension"].value_counts().to_dict(),
        "colour_modes": valid["mode"].value_counts().to_dict(),
        "width_stats": valid["width"].describe().round(2).to_dict(),
        "height_stats": valid["height"].describe().round(2).to_dict(),
        "duplicate_groups": int(dup_groups.ngroups),
        "images_in_duplicate_groups": int(len(dup)),
        "duplicate_groups_with_conflicting_labels": cross_class,
        "patient_id_sources": valid["patient_source"].value_counts().to_dict(),
        "patient_id_coverage": round(float(len(with_pid) / max(total, 1)), 4),
        "unique_patients_total": int(with_pid["patient_id"].nunique()),
        "unique_patients_per_class": {k: int(v) for k, v in patients_per_class.items()},
        "patients_with_cells_in_both_classes": len(pids_both),
    }
    return report


# ---------------------------------------------------------------------------
# Visualisations (saved to outputs/graphs)
# ---------------------------------------------------------------------------
COLOURS = {"Uninfected": "#2a9d8f", "Parasitized": "#c0392b"}


def plot_class_distribution(df: pd.DataFrame) -> Path:
    counts = df[df.valid]["class_name"].value_counts().reindex(config.CLASS_NAMES)
    fig = Figure(figsize=(6, 4.5))
    ax = fig.subplots()
    bars = ax.bar(counts.index, counts.values, color=[COLOURS[c] for c in counts.index])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({v / counts.sum():.1%})",
                ha="center", va="bottom", fontsize=10)
    ax.set_title("Class distribution (valid images)")
    ax.set_ylabel("Number of images")
    ax.set_ylim(0, counts.max() * 1.18)
    fig.tight_layout()
    out = config.GRAPHS_DIR / "class_distribution.png"
    fig.savefig(out, dpi=150)
    return out


def plot_sample_images(df: pd.DataFrame, n_per_class: int = 8) -> Path:
    fig = Figure(figsize=(n_per_class * 1.6, 4))
    axes = fig.subplots(2, n_per_class)
    rng = np.random.default_rng(config.RANDOM_SEED)
    for r, class_name in enumerate(config.CLASS_NAMES[::-1]):  # Parasitized on top
        subset = df[(df.valid) & (df.class_name == class_name)]
        picks = subset.iloc[rng.choice(len(subset), size=min(n_per_class, len(subset)), replace=False)]
        for c in range(n_per_class):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(picks):
                with Image.open(config.BASE_DIR / picks.iloc[c].rel_path) as img:
                    ax.imshow(img.convert("RGB"))
        axes[r, 0].set_title(class_name, loc="left", fontsize=11, color=COLOURS[class_name])
    fig.suptitle("Random sample cells from each class", fontsize=12)
    fig.tight_layout()
    out = config.GRAPHS_DIR / "sample_images.png"
    fig.savefig(out, dpi=150)
    return out


def plot_image_dimensions(df: pd.DataFrame) -> Path:
    v = df[df.valid]
    fig = Figure(figsize=(13, 4))
    ax1, ax2, ax3 = fig.subplots(1, 3)
    for c in config.CLASS_NAMES:
        s = v[v.class_name == c]
        ax1.hist(s.width, bins=50, alpha=0.6, label=c, color=COLOURS[c])
        ax2.hist(s.height, bins=50, alpha=0.6, label=c, color=COLOURS[c])
        ax3.scatter(s.width, s.height, s=3, alpha=0.3, label=c, color=COLOURS[c])
    ax1.set_title("Image width (pixels)"); ax2.set_title("Image height (pixels)")
    ax3.set_title(f"Width vs height (all resized to {config.IMAGE_SIZE}x{config.IMAGE_SIZE})")
    ax3.set_xlabel("width"); ax3.set_ylabel("height")
    for ax in (ax1, ax2, ax3):
        ax.legend()
    fig.tight_layout()
    out = config.GRAPHS_DIR / "image_dimensions.png"
    fig.savefig(out, dpi=150)
    return out


def plot_colour_intensity(df: pd.DataFrame) -> Path:
    v = df[df.valid]
    fig = Figure(figsize=(13, 4))
    axes = fig.subplots(1, 3)
    for ax, ch, name in zip(axes, ["mean_r", "mean_g", "mean_b"], ["Red", "Green", "Blue"]):
        for c in config.CLASS_NAMES:
            ax.hist(v[v.class_name == c][ch], bins=50, alpha=0.6, label=c, color=COLOURS[c])
        ax.set_title(f"Mean {name} intensity per image")
        ax.set_xlabel("0 = dark, 255 = bright")
        ax.legend()
    fig.suptitle("Colour characteristics (includes black background around each cell)")
    fig.tight_layout()
    out = config.GRAPHS_DIR / "colour_intensity.png"
    fig.savefig(out, dpi=150)
    return out


def plot_cells_per_patient(df: pd.DataFrame) -> Path | None:
    v = df[(df.valid) & (df.patient_id != "")]
    if v.empty:
        return None
    per = v.groupby(["patient_id", "class_name"]).size().unstack(fill_value=0)
    fig = Figure(figsize=(7, 4))
    ax = fig.subplots()
    for c in config.CLASS_NAMES:
        if c in per:
            vals = per[c][per[c] > 0]
            ax.hist(vals, bins=40, alpha=0.6, label=f"{c} ({len(vals)} patients)", color=COLOURS[c])
    ax.set_title("Number of cell images contributed per patient")
    ax.set_xlabel("cells per patient"); ax.set_ylabel("number of patients")
    ax.legend()
    fig.tight_layout()
    out = config.GRAPHS_DIR / "cells_per_patient.png"
    fig.savefig(out, dpi=150)
    return out


# ---------------------------------------------------------------------------
# Loading validation (uses the SAME preprocessing as training/prediction)
# ---------------------------------------------------------------------------
def validate_loading(df: pd.DataFrame, n_per_class: int = 20) -> dict:
    from src.preprocessing import preprocess_image  # local import avoids circular import

    results = {}
    for c in config.CLASS_NAMES:
        subset = df[(df.valid) & (df.class_name == c)].head(n_per_class)
        arrays = [preprocess_image(config.BASE_DIR / p) for p in subset.rel_path]
        stack = np.stack(arrays)
        assert stack.shape[1:] == (config.IMAGE_SIZE, config.IMAGE_SIZE, config.CHANNELS), stack.shape
        assert stack.dtype == np.float32
        assert 0.0 <= stack.min() and stack.max() <= 1.0
        assert (subset.label == config.LABEL_TO_INDEX[c]).all()
        results[c] = {"checked": len(arrays), "shape": list(stack.shape[1:]),
                      "min": float(stack.min()), "max": float(stack.max()),
                      "label": config.LABEL_TO_INDEX[c]}
    return results


def run_inspection(verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    config.ensure_dirs()
    if verbose:
        print(f"Dataset location: {config.DATASET_PATH}")
    df = scan_dataset(verbose=verbose)
    df.to_csv(config.INVENTORY_PATH, index=False)
    report = summarise(df)
    report["loading_validation"] = validate_loading(df)
    report["graphs"] = [str(p.relative_to(config.BASE_DIR)) for p in [
        plot_class_distribution(df), plot_sample_images(df), plot_image_dimensions(df),
        plot_colour_intensity(df), plot_cells_per_patient(df)] if p is not None]
    with open(config.DATASET_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    return df, report


def print_report(r: dict) -> None:
    print("\n================ DATASET INSPECTION REPORT ================")
    print(f"Class folders found      : {r['class_folders_found']}")
    print(f"Total files              : {r['total_files']}")
    print(f"Valid images per class   : {r['valid_images_per_class']}")
    print(f"Total valid images       : {r['total_valid_images']}")
    print(f"Non-image files skipped  : {len(r['non_image_files'])} {r['non_image_files'][:5]}")
    print(f"Corrupted/unreadable     : {len(r['corrupted_or_unreadable'])}")
    print(f"Extensions               : {r['extensions']}")
    print(f"Colour modes             : {r['colour_modes']}")
    print(f"Width  (min/mean/max)    : {r['width_stats'].get('min')} / "
          f"{r['width_stats'].get('mean')} / {r['width_stats'].get('max')}")
    print(f"Height (min/mean/max)    : {r['height_stats'].get('min')} / "
          f"{r['height_stats'].get('mean')} / {r['height_stats'].get('max')}")
    print(f"Balance ratio (min/max)  : {r['class_balance_ratio_min_over_max']} "
          f"-> {'IMBALANCED' if r['is_imbalanced'] else 'balanced'}")
    print(f"Duplicate groups         : {r['duplicate_groups']} "
          f"({r['images_in_duplicate_groups']} images, "
          f"{r['duplicate_groups_with_conflicting_labels']} with conflicting labels)")
    print(f"Patient ID sources       : {r['patient_id_sources']}")
    print(f"Patient ID coverage      : {r['patient_id_coverage']:.1%}")
    print(f"Unique patients          : {r['unique_patients_total']} "
          f"(per class: {r['unique_patients_per_class']}, "
          f"in both classes: {r['patients_with_cells_in_both_classes']})")
    print(f"Loading validation       : OK for {list(r['loading_validation'])}")
    print(f"Graphs saved             : {r['graphs']}")
    print(f"Inventory saved          : {config.INVENTORY_PATH.relative_to(config.BASE_DIR)}")
    print(f"Report saved             : {config.DATASET_REPORT_PATH.relative_to(config.BASE_DIR)}")
    print("============================================================")


if __name__ == "__main__":
    _, rep = run_inspection()
    print_report(rep)

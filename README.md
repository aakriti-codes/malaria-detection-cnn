# AI-Based Malaria Detection Using CNN with LIME and SHAP Explainability

A Convolutional Neural Network (CNN) classifies microscopic images of single red blood cells
as **Parasitized** or **Uninfected**. **LIME** and **SHAP** show which image regions drove each
prediction, and a **Streamlit** web app lets you upload an image to see the prediction and both
explanations.

> ⚠️ **Disclaimer:** this is an educational research prototype, not a medical device. Do not use it
> for diagnosis.

---

## 1. Project structure

```text
malaria/
├── cell_images/                 # YOUR dataset (unchanged)
│   ├── Parasitized/
│   └── Uninfected/
├── metadata/                    # (optional) official NIH patient-ID CSV files
├── models/                      # created by training
│   ├── malaria_cnn.keras        #   trained CNN
│   ├── model_config.json        #   class labels + preprocessing settings
│   └── shap_background.npy      #   100 training images used by SHAP GradientExplainer
├── outputs/
│   ├── graphs/                  # EDA, training curves, confusion matrix, ROC ...
│   ├── explanations/            # saved LIME / SHAP figures
│   └── reports/                 # inventory, split, metrics.json, classification report ...
├── cache/                       # resized images for fast reloading (safe to delete)
├── notebooks/
│   └── malaria_project_walkthrough.ipynb
├── docs/                        # report text, viva Q&A, architecture & workflow diagrams
├── src/
│   ├── __init__.py
│   ├── dataset_inspection.py    # validation + EDA + patient IDs
│   ├── preprocessing.py         # image pipeline, patient-level split, tf.data, augmentation
│   ├── model.py                 # CNN architecture
│   ├── train.py                 # training with callbacks
│   ├── evaluate.py              # metrics + graphs on the test set
│   ├── prediction.py            # load model once, predict any image
│   ├── lime_explainer.py        # LIME (LimeImageExplainer)
│   └── shap_explainer.py        # SHAP (PartitionExplainer / GradientExplainer)
├── app.py                       # Streamlit application
├── config.py                    # ALL paths and settings in one place
├── check_setup.py               # checks Python, packages and dataset before training
├── requirements.txt
├── README.md
├── PROJECT_WORKFLOW.md
└── .gitignore
```

Files added beyond the brief, and why:
* `check_setup.py` finds installation problems before a long training run.
* `docs/` holds the abstract, conclusion, future scope, viva Q&A and diagrams.
* `metadata/` is optional and holds NIH's patient-mapping files (see section 6).

---

## 2. One-time setup (Windows)

### 2.1 Install Python 3.11

TensorFlow 2.15 works **only with Python 3.9 to 3.11**. Python 3.12 and 3.13 will fail.
Your existing Python does not need to be removed. Choose **one** option.

**Option A: python.org (recommended)**
1. Download *Python 3.11.9, Windows installer (64-bit)* from <https://www.python.org/downloads/windows/>.
2. In the installer, tick **"Add python.exe to PATH"**, then click *Install Now*.
3. Open a new *Command Prompt* and check that it is available: `py -3.11 --version`

**Option B: Anaconda (already on your laptop)**
```bat
conda create -n malaria python=3.11 -y
conda activate malaria
```
Then skip step 2.2 and go straight to 2.3.

### 2.2 Create and activate a virtual environment (Option A)

Open VS Code, then **File → Open Folder… → `malaria`**, then **Terminal → New Terminal**.
In the terminal dropdown (˅ next to +), choose **Command Prompt** and run:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
```

Your prompt should now start with `(.venv)`. Run the `activate` line every time you open a new terminal.

> If you use **PowerShell** instead and see *"running scripts is disabled"*, run this once:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then use `.venv\Scripts\Activate.ps1`.

### 2.3 Install the packages
```bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```
This downloads about 1 GB (TensorFlow is the large one), so it can take 5 to 15 minutes.

### 2.4 Tell VS Code to use this environment
Press **Ctrl+Shift+P**, choose **Python: Select Interpreter**, then pick `.venv` (or `malaria` for conda).

### 2.5 Check everything
```bat
python check_setup.py
```
You should see `✅ Ready.` If you don't, see **Troubleshooting** (section 9).

---

## 3. Run the project (in this order)

All commands are run from the `malaria` folder with the environment activated.

| Step | Command | What it does | Produces |
|---|---|---|---|
| 1 | `python -m src.dataset_inspection` | Validates every image, finds duplicates and patient IDs, draws EDA graphs | `outputs/reports/dataset_*.{csv,json}`, `outputs/graphs/*.png` |
| 2 | `python -m src.preprocessing` | Creates the patient-level train/val/test split | `outputs/reports/data_split.csv`, `split_distribution.png` |
| 3 | `python -m src.train` | Trains the CNN (early stopping) | `models/malaria_cnn.keras`, `model_config.json`, training curves |
| 4 | `python -m src.evaluate` | Tests on unseen patients | `metrics.json`, `classification_report.txt`, confusion matrix, ROC |
| 5 | `streamlit run app.py` | Starts the web app | opens <http://localhost:8501> |

Step 2 is optional because step 3 creates the split automatically. Steps 1 to 4 only need to be run
once. After that, `streamlit run app.py` is enough (the app loads the saved model and never retrains).

Optional command-line tools (replace the path with any cell image):
```bat
python -m src.prediction cell_images\Parasitized\<file>.png
python -m src.lime_explainer cell_images\Parasitized\<file>.png
python -m src.shap_explainer cell_images\Parasitized\<file>.png
python -m src.shap_explainer cell_images\Parasitized\<file>.png --method gradient
```
LIME and SHAP figures are saved in `outputs/explanations/`.

To stop Streamlit, press **Ctrl+C** in the terminal.

### Rough timings on a laptop CPU (estimates, not measurements)
* Dataset inspection: 2 to 4 minutes (opens all 27,558 files).
* First training run: 1 to 3 minutes to load and cache the images, then roughly 1 minute per
  epoch. Early stopping usually ends training after 15 to 30 epochs.
* LIME takes a few seconds per image. SHAP (partition) takes a few seconds; SHAP (gradient) takes about a second.

---

## 4. Changing settings

Edit **`config.py`**. Common changes:

| Setting | Default | When to change it |
|---|---|---|
| `EPOCHS` | 30 | Lower it (e.g. 5) for a quick trial run |
| `BATCH_SIZE` | 64 | Use 32 if you get memory errors |
| `IMAGE_SIZE` | 64 | 96 or 128 may add a little accuracy but is slower. **Retrain afterwards.** |
| `LIME_NUM_SAMPLES` | 1000 | More gives more stable explanations but is slower |
| `SHAP_MAX_EVALS` | 500 | More gives finer SHAP maps but is slower |

Some can be set from the command line instead, e.g. `python -m src.train --epochs 10 --batch-size 32`.

---

## 5. GPU or CPU?

The code detects hardware automatically and prints `device: GPU` or `device: CPU`.
**Native Windows TensorFlow (2.11 and later) cannot use NVIDIA GPUs**, so on Windows the model
trains on the CPU even on a gaming laptop. This is expected, and the small CNN is designed for it.
(GPU training on Windows needs WSL2, which is not required for this project.)

---

## 6. Patient-level split (data-leakage prevention)

Several cell images come from the **same patient**. If one patient's cells were in both the
training and test sets, the test score would be over-optimistic, because the model would partly
recognise the patient's slide (stain colour, camera) rather than the parasite.

* NIH states that the patient ID is encoded in each filename. For example, `P1` in
  `C33P1thinF_IMG_20150619_114756a_cell_179.png`. NIH also publishes official
  patient-to-cell CSV files.
* `dataset_inspection.py` assigns each image a patient using:
  1. the official CSV files, **if** you place them in `malaria/metadata/` (optional):
     `patientid_cellmapping_parasitized.csv` and `patientid_cellmapping_uninfected.csv`
     from <https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-datasheet.html>;
  2. otherwise, the NIH-documented filename convention. Filenames without a `P` number use
     their slide prefix (the text before `_IMG_`) as the group.
* `preprocessing.py` then uses `StratifiedGroupKFold`, so **every patient is in exactly one split**
  (about 72% train, 14% validation, 14% test). Byte-identical duplicate images are also forced into
  the same split. The code asserts these conditions and stops with an error if any is violated.
* If fewer than 95% of the files had a recognisable patient ID, it would fall back to a
  stratified random split by image and record this in `data_split.csv`.

---

## 7. What each output means

* `outputs/reports/dataset_report.json`: counts, corrupted files, duplicates, sizes and patient statistics.
* `outputs/reports/metrics.json`: accuracy, precision, recall, specificity, F1, ROC-AUC and the
  confusion matrix, reported separately for train, validation and test.
* `outputs/reports/classification_report.txt`: per-class precision, recall and F1 on the test set.
* `outputs/reports/test_predictions.csv`: every test image with its predicted probability.
* `outputs/graphs/`: all figures, ready for your report or presentation.

Metric definitions are in `PROJECT_WORKFLOW.md` (stage 6).

---

## 8. Dataset source

NIH Malaria Cell Images Dataset: Lister Hill National Center for Biomedical Communications (LHNCBC),
U.S. National Library of Medicine. 27,558 cell images (13,779 per class) from Giemsa-stained thin
blood smears of 150 *P. falciparum*-infected and 50 uninfected patients, collected at Chittagong
Medical College Hospital, Bangladesh.

Rajaraman S, Antani SK, Poostchi M, Silamut K, Hossain MA, Maude RJ, Jaeger S, Thoma GR (2018).
*Pre-trained convolutional neural networks as feature extractors toward improved malaria parasite
detection in thin blood smear images.* PeerJ 6:e4568.

---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `No matching distribution found for tensorflow==2.15.1` | You are on Python 3.12 or 3.13. Recreate the venv with `py -3.11 -m venv .venv`. |
| `OSError ... No such file or directory` while installing TensorFlow | Windows long-path limit. Either enable long paths (run PowerShell as admin: `New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force`, then restart), or create the venv in a short path: `py -3.11 -m venv C:\venvs\malaria` and `C:\venvs\malaria\Scripts\activate`. |
| `ModuleNotFoundError: No module named 'config'` or `'src'` | Run commands from the `malaria` folder and use `python -m src.train` (with `-m`), not `python src\train.py` from another folder. |
| `DLL load failed` when importing TensorFlow | Install the *Microsoft Visual C++ Redistributable (x64)* from Microsoft, then restart. |
| `Trained model not found` in the app | Run steps 1 to 4 in section 3 first. |
| Out of memory during training | `python -m src.train --batch-size 32` |
| Training is slow | Lower `EPOCHS` for a test run. Close other heavy programs. |
| Streamlit doesn't open a browser | Open <http://localhost:8501> manually. |

---

## 10. Testing status

This project was written and partly tested in a cloud workspace **without** TensorFlow, LIME, SHAP or
Streamlit, and **without** your full dataset. See `PROJECT_WORKFLOW.md`, section "Testing status",
for exactly what was executed and what must be run on your computer.

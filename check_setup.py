"""
check_setup.py — verify your environment BEFORE training.

(Extra file, added because package-installation problems on Windows are common:
it tells you exactly what is wrong instead of a long error in the middle of training.)

Run from the project root:
    python check_setup.py
"""

import importlib
import platform
import sys

import config

EXPECTED = {
    "tensorflow": "2.15.1", "numpy": "1.26.4", "pandas": "2.2.2", "sklearn": "1.4.2",
    "skimage": "0.22.0", "matplotlib": "3.8.4", "PIL": "10.3.0", "lime": None,
    "shap": "0.44.1", "cv2": "4.9.0", "streamlit": "1.32.2",
}

ok = True
print(f"Python {platform.python_version()} ({platform.architecture()[0]}) at {sys.executable}")
if sys.version_info[:2] != (3, 11):
    print("  ⚠ Recommended Python is 3.11. TensorFlow 2.15 does not support 3.12+.")
    ok = sys.version_info[:2] in [(3, 9), (3, 10), (3, 11)] and ok
if sys.prefix == sys.base_prefix and "conda" not in sys.version.lower():
    print("  ⚠ You are NOT inside a virtual environment. Activate .venv first.")

print("\nPackages:")
for mod, want in EXPECTED.items():
    try:
        m = importlib.import_module(mod)
        have = getattr(m, "__version__", "installed")
        flag = "OK" if (want is None or str(have).startswith(want)) else f"expected {want}"
        print(f"  {mod:12s} {have:12s} {flag}")
    except Exception as e:
        ok = False
        print(f"  {mod:12s} MISSING / BROKEN -> {type(e).__name__}: {e}")

try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    print(f"\nTensorFlow devices: {'GPU ' + str(gpus) if gpus else 'CPU only (normal on native Windows)'}")
except Exception:
    pass

print(f"\nDataset folder: {config.DATASET_PATH}")
for c in config.CLASS_NAMES:
    d = config.DATASET_PATH / c
    if d.is_dir():
        n = sum(1 for p in d.iterdir() if p.suffix.lower() in config.VALID_EXTENSIONS)
        print(f"  {c:12s} found, {n} image files")
    else:
        ok = False
        print(f"  {c:12s} NOT FOUND at {d}")

print(f"\nTrained model: {'found' if config.MODEL_PATH.exists() else 'not trained yet'} ({config.MODEL_PATH.name})")
print("\nRESULT:", "✅ Ready." if ok else "❌ Fix the problems above (see README > Troubleshooting).")

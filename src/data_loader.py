"""
data_loader.py
---------------
Loads raw WESAD subject files (S2.pkl, S3.pkl, ... S17.pkl) once you have
downloaded and extracted the dataset from:
https://www.eti.uni-siegen.de/ubicomp/home/datasets/icmi18/

Expected folder layout after extraction:

    data/raw/WESAD/S2/S2.pkl
    data/raw/WESAD/S3/S3.pkl
    ...

Each .pkl file is a Python dict, pickled with protocol requiring
encoding='latin1' to unpickle correctly (it was originally saved
under Python 2 / early Python 3 numpy versions).

WESAD label codes (from the dataset README):
    0 = not defined / transient
    1 = baseline
    2 = stress
    3 = amusement
    4 = meditation
    5, 6, 7 = ignore (should be dropped)

This module focuses on the WRIST device (Empatica E4) signals, since
those match the sensors realistically available on a low-power wearable:
    BVP  (blood volume pulse, proxy for heart rate) - 64 Hz
    EDA  (electrodermal activity)                    - 4 Hz
    TEMP (skin temperature)                          - 4 Hz
    ACC  (3-axis accelerometer)                      - 32 Hz
"""

import pickle
import numpy as np
from pathlib import Path

# Sampling rates for each wrist signal, in Hz (fixed by the Empatica E4 device)
WRIST_SAMPLING_RATES = {
    "BVP": 64,
    "EDA": 4,
    "TEMP": 4,
    "ACC": 32,
}

# The chest device's label stream is sampled at 700 Hz and is the
# reference clock for the whole recording.
LABEL_SAMPLING_RATE = 700

VALID_LABELS = {1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}


def load_subject_raw(subject_dir: Path) -> dict:
    """
    Load a single subject's raw .pkl file.

    Parameters
    ----------
    subject_dir : Path
        Path to the subject's folder, e.g. data/raw/WESAD/S2

    Returns
    -------
    dict with keys:
        'wrist'  -> dict of {signal_name: np.ndarray}
        'label'  -> np.ndarray of per-sample labels at 700 Hz
        'subject'-> subject id string, e.g. "S2"
    """
    subject_id = subject_dir.name
    pkl_path = subject_dir / f"{subject_id}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Could not find {pkl_path}. Check that you extracted WESAD "
            f"into data/raw/WESAD/ with one folder per subject."
        )

    with open(pkl_path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")

    wrist_signals = {
        "BVP": np.asarray(raw["signal"]["wrist"]["BVP"]).flatten(),
        "EDA": np.asarray(raw["signal"]["wrist"]["EDA"]).flatten(),
        "TEMP": np.asarray(raw["signal"]["wrist"]["TEMP"]).flatten(),
        "ACC": np.asarray(raw["signal"]["wrist"]["ACC"]),  # shape (N, 3)
    }
    labels = np.asarray(raw["label"]).flatten()

    return {"wrist": wrist_signals, "label": labels, "subject": subject_id}


def load_all_subjects(wesad_root: str = "data/raw/WESAD") -> list:
    """
    Load every subject folder found under wesad_root.
    Returns a list of the dicts produced by load_subject_raw().
    Skips any folder that doesn't contain a matching .pkl file.
    """
    root = Path(wesad_root)
    if not root.exists():
        raise FileNotFoundError(
            f"{root} does not exist yet. Download WESAD and extract it "
            f"there first (see the docstring at the top of this file)."
        )

    subjects = []
    for subject_dir in sorted(root.glob("S*")):
        if not subject_dir.is_dir():
            continue
        try:
            subjects.append(load_subject_raw(subject_dir))
            print(f"Loaded {subject_dir.name}")
        except FileNotFoundError as e:
            print(f"Skipping {subject_dir.name}: {e}")
    return subjects


if __name__ == "__main__":
    # Quick manual test once you have real WESAD data downloaded.
    subjects = load_all_subjects()
    print(f"\nLoaded {len(subjects)} subjects.")
    if subjects:
        s = subjects[0]
        print(f"Example subject: {s['subject']}")
        for name, arr in s["wrist"].items():
            print(f"  {name}: shape={arr.shape}")
        print(f"  label: shape={s['label'].shape}, unique values={np.unique(s['label'])}")

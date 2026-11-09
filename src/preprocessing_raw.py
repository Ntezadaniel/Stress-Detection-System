"""
preprocessing_raw.py
---------------------
Extends Step 2's windowing to produce RAW resampled sequences instead
of summary statistics, for the deep learning models coming in Step 4+.

Why a separate script rather than editing preprocessing.py: your
classical baseline (Step 3) already works off the feature-based
windows.npz. Keeping that file and script untouched means you can
always re-run your baseline exactly as before -- this file adds a new
capability rather than replacing a working one.

Design decisions (write these into your Methodology chapter):
  - All four signal types are resampled to a common rate (TARGET_HZ)
    so they can be stacked into one tensor per window.
  - The 3-axis accelerometer is collapsed to a single magnitude value
    per timestep (sqrt(x^2 + y^2 + z^2)), representing overall movement
    intensity regardless of orientation. This is standard practice and
    keeps the channel count small.
  - Resulting channel order per window: [BVP, EDA, TEMP, ACC_magnitude]
"""

import argparse
import numpy as np
from pathlib import Path
from scipy.signal import resample

from data_loader import load_all_subjects, LABEL_SAMPLING_RATE, WRIST_SAMPLING_RATES

KEEP_LABELS = {1, 2, 3}
WINDOW_SECONDS = 60
TARGET_HZ = 32  # matches the accelerometer's native rate


def majority_label(label_window: np.ndarray) -> int:
    values, counts = np.unique(label_window, return_counts=True)
    return int(values[np.argmax(counts)])


def resample_to_length(signal: np.ndarray, target_length: int) -> np.ndarray:
    """Resample a 1D signal to exactly target_length samples."""
    return resample(signal, target_length)


def windowize_subject_raw(subject: dict, window_seconds: int = WINDOW_SECONDS,
                           target_hz: int = TARGET_HZ):
    labels = subject["label"]
    label_window_size = window_seconds * LABEL_SAMPLING_RATE
    n_windows = len(labels) // label_window_size
    target_length = window_seconds * target_hz

    X, y = [], []
    for w in range(n_windows):
        label_start = w * label_window_size
        label_end = label_start + label_window_size
        window_label = majority_label(labels[label_start:label_end])
        if window_label not in KEEP_LABELS:
            continue

        channels = []
        for name in ["BVP", "EDA", "TEMP"]:
            hz = WRIST_SAMPLING_RATES[name]
            sig_window_size = window_seconds * hz
            sig_start = w * sig_window_size
            sig_end = sig_start + sig_window_size
            raw_signal = subject["wrist"][name][sig_start:sig_end]
            channels.append(resample_to_length(raw_signal, target_length))

        # ACC: collapse 3 axes to a single magnitude value, then resample
        acc_hz = WRIST_SAMPLING_RATES["ACC"]
        acc_window_size = window_seconds * acc_hz
        acc_start = w * acc_window_size
        acc_end = acc_start + acc_window_size
        acc_raw = subject["wrist"]["ACC"][acc_start:acc_end]  # shape (N, 3)
        acc_magnitude = np.linalg.norm(acc_raw, axis=1)
        channels.append(resample_to_length(acc_magnitude, target_length))

        window_array = np.stack(channels, axis=1)  # shape (time_steps, 4 channels)
        X.append(window_array)
        y.append(window_label)

    return np.array(X), np.array(y)


def build_dataset_raw(subjects: list, window_seconds: int = WINDOW_SECONDS,
                       target_hz: int = TARGET_HZ):
    all_X, all_y, all_subject_ids = [], [], []
    for subject in subjects:
        X, y = windowize_subject_raw(subject, window_seconds, target_hz)
        all_X.append(X)
        all_y.append(y)
        all_subject_ids.extend([subject["subject"]] * len(y))
        print(f"{subject['subject']}: {len(y)} windows kept")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    subject_ids = np.array(all_subject_ids)
    return X, y, subject_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--wesad-root", type=str, default="data/raw/WESAD")
    parser.add_argument("--out", type=str, default="data/processed/raw_windows.npz")
    args = parser.parse_args()

    if args.synthetic:
        from synthetic_data import generate_synthetic_dataset
        subjects = generate_synthetic_dataset(n_subjects=3, duration_sec=600)
    else:
        subjects = load_all_subjects(args.wesad_root)

    X, y, subject_ids = build_dataset_raw(subjects)

    print(f"\nTotal windows: {len(y)}")
    print(f"Window shape: {X.shape[1:]} (time_steps, channels)")
    unique, counts = np.unique(y, return_counts=True)
    label_names = {1: "baseline", 2: "stress", 3: "amusement"}
    for u, c in zip(unique, counts):
        print(f"  label {u} ({label_names[u]}): {c} windows")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, X=X, y=y, subject_ids=subject_ids)
    print(f"\nSaved to {args.out}")
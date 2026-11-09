"""
preprocessing.py
-----------------
STEP 2: turn one subject's raw, multi-rate signals into fixed-length
windows, each summarized as a small feature vector with one label.

Why features instead of raw signal, for now: it keeps this step simple
and testable, and gives you exactly what Step 3's classical baseline
model needs. Raw-sequence windows (for the deep learning teacher/student
models) come later, once you reach Step 4.
"""

import argparse
import numpy as np
from pathlib import Path

from data_loader import load_all_subjects, LABEL_SAMPLING_RATE, WRIST_SAMPLING_RATES

# Only these labels correspond to a condition we care about.
# Everything else (0, 4=meditation, 5, 6, 7) gets dropped for this project.
KEEP_LABELS = {1, 2, 3}
WINDOW_SECONDS = 60


def majority_label(label_window: np.ndarray) -> int:
    """Return the most common label value inside this window."""
    values, counts = np.unique(label_window, return_counts=True)
    return int(values[np.argmax(counts)])


def extract_window_features(signal_window: np.ndarray) -> list:
    """Turn one channel's raw samples in a window into 4 summary numbers."""
    if signal_window.ndim == 2:  # ACC has 3 axes -- collapse to magnitude first
        signal_window = np.linalg.norm(signal_window, axis=1)
    return [
        float(np.mean(signal_window)),
        float(np.std(signal_window)),
        float(np.min(signal_window)),
        float(np.max(signal_window)),
    ]


def windowize_subject(subject: dict, window_seconds: int = WINDOW_SECONDS):
    """
    Split one subject's recording into non-overlapping windows.
    Returns (X, y): X has shape (n_windows, n_features), y has shape
    (n_windows,) with values in {1, 2, 3}.
    """
    labels = subject["label"]
    label_window_size = window_seconds * LABEL_SAMPLING_RATE
    n_windows = len(labels) // label_window_size

    X, y = [], []
    for w in range(n_windows):
        label_start = w * label_window_size
        label_end = label_start + label_window_size
        window_label = majority_label(labels[label_start:label_end])

        if window_label not in KEEP_LABELS:
            continue  # skip transient/meditation/ignore windows

        feature_row = []
        for name, hz in WRIST_SAMPLING_RATES.items():
            sig_window_size = window_seconds * hz
            sig_start = w * sig_window_size
            sig_end = sig_start + sig_window_size
            signal = subject["wrist"][name][sig_start:sig_end]
            feature_row.extend(extract_window_features(signal))

        X.append(feature_row)
        y.append(window_label)

    return np.array(X), np.array(y)


def build_dataset(subjects: list, window_seconds: int = WINDOW_SECONDS):
    """
    Run windowize_subject() on every subject, tracking which subject
    each window came from -- Step 3/4 will need this to split
    train/test BY SUBJECT, not by window.
    """
    all_X, all_y, all_subject_ids = [], [], []
    for subject in subjects:
        X, y = windowize_subject(subject, window_seconds)
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
    parser.add_argument("--out", type=str, default="data/processed/windows.npz")
    args = parser.parse_args()

    if args.synthetic:
        from synthetic_data import generate_synthetic_dataset
        subjects = generate_synthetic_dataset(n_subjects=3, duration_sec=600)
    else:
        subjects = load_all_subjects(args.wesad_root)

    X, y, subject_ids = build_dataset(subjects)

    print(f"\nTotal windows: {len(y)}")
    print(f"Feature vector size per window: {X.shape[1]}")
    unique, counts = np.unique(y, return_counts=True)
    label_names = {1: "baseline", 2: "stress", 3: "amusement"}
    for u, c in zip(unique, counts):
        print(f"  label {u} ({label_names[u]}): {c} windows")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, X=X, y=y, subject_ids=subject_ids)
    print(f"\nSaved to {args.out}")
"""
compare_normalization.py
-------------------------
One command, fair comparisons on the SAME 5 subject-grouped folds.

Question being tested: does standardising each subject's signals against
THEIR OWN overall statistics (no labels used) help models generalise to
people they have never seen?

  Row 1: Random Forest on summary features            (your Step 3 result, recomputed)
  Row 2: Random Forest + per-subject normalization
  Row 3: CNN-LSTM teacher + per-subject normalization

Your previous teacher run (global normalization) is printed for reference.

Design note for your Methodology chapter: per-subject normalization means
each wearer needs their own running mean/std, estimated from their own
data (for example over a calibration period). That is cheap to compute on
a wearable, but it IS a deployment requirement, so state it as one.
"""

import argparse
import time
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

from baseline_model import load_windows
from train_teacher import (load_raw_windows, downsample, train_one_model,
                           predict_in_batches, LABEL_TO_INDEX)


def subject_normalize(X: np.ndarray, subject_ids: np.ndarray) -> np.ndarray:
    """
    Z-score every subject's data using that subject's own mean/std per
    feature/channel. Works for (N, F) feature arrays and (N, T, C) sequences.
    Uses no labels.
    """
    Xn = np.empty(X.shape, dtype=np.float64)
    for s in np.unique(subject_ids):
        mask = subject_ids == s
        Xs = X[mask]
        axes = tuple(range(Xs.ndim - 1))  # everything except the last axis
        mean = Xs.mean(axis=axes, keepdims=True)
        std = Xs.std(axis=axes, keepdims=True) + 1e-8
        Xn[mask] = (Xs - mean) / std
    return Xn


def cv_random_forest(X, y, subject_ids):
    accs, f1s = [], []
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups=subject_ids):
        model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
        model.fit(X[tr], y[tr])
        preds = model.predict(X[te])
        accs.append(float((preds == y[te]).mean()))
        f1s.append(f1_score(y[te], preds, labels=[0, 1, 2], average=None, zero_division=0))
    return np.array(accs), np.array(f1s)


def cv_teacher(X, y, subject_ids, epochs, batch_size, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    accs, f1s = [], []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, groups=subject_ids), start=1):
        t0 = time.time()
        model, _ = train_one_model(X[tr], y[tr], X[te], y[te], epochs=epochs,
                                   batch_size=batch_size, select_best=False, verbose=False)
        preds = predict_in_batches(model, torch.tensor(X[te], dtype=torch.float32))
        accs.append(float((preds == y[te]).mean()))
        f1s.append(f1_score(y[te], preds, labels=[0, 1, 2], average=None, zero_division=0))
        print(f"    teacher fold {fold}/5 done ({time.time() - t0:.0f}s)")
    return np.array(accs), np.array(f1s)


def row(name, accs, f1s):
    m = f1s.mean(axis=0)
    return (f"{name:42s} {accs.mean():.2f} +/- {accs.std():.2f}    "
            f"{m[0]:.2f}        {m[1]:.2f}       {m[2]:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/processed/windows.npz")
    parser.add_argument("--raw", default="data/processed/raw_windows.npz")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--downsample", type=int, default=8)
    parser.add_argument("--rf-only", action="store_true", help="skip the slow teacher run")
    args = parser.parse_args()

    to_index = np.vectorize(LABEL_TO_INDEX.get)
    rows = []

    # ---- Random Forest rows (fast) ----
    Xf, yf_raw, sid_f = load_windows(args.features)
    yf = to_index(yf_raw)
    print("Random Forest, raw summary features...")
    rows.append(row("Random Forest (Step 3 features)", *cv_random_forest(Xf, yf, sid_f)))
    print("Random Forest, per-subject normalization...")
    rows.append(row("Random Forest + per-subject norm", *cv_random_forest(subject_normalize(Xf, sid_f), yf, sid_f)))

    # ---- Teacher row (slow, about 4-5 minutes) ----
    if not args.rf_only:
        Xr, yr_raw, sid_r = load_raw_windows(args.raw)
        yr = to_index(yr_raw)
        Xr = subject_normalize(downsample(Xr, args.downsample), sid_r)
        print("CNN-LSTM teacher, per-subject normalization (about 5 minutes)...")
        rows.append(row("CNN-LSTM teacher + per-subject norm",
                        *cv_teacher(Xr, yr, sid_r, args.epochs, args.batch_size)))

    print("\n" + "=" * 92)
    print(f"{'Model (5 folds, grouped by subject)':42s} {'accuracy':15s} {'baseline F1':12s}{'stress F1':11s}{'amusement F1'}")
    print("=" * 92)
    for r in rows:
        print(r)
    print("-" * 92)
    print(f"{'(previous run) CNN-LSTM teacher, global norm':42s} 0.53 +/- 0.10    0.60        0.55       0.18")


if __name__ == "__main__":
    main()
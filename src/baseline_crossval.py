"""
baseline_crossval.py
---------------------
Runs the classical baseline across MULTIPLE subject-based train/test
splits (grouped cross-validation), instead of trusting just one.

Why: with only 15 subjects, one lucky or unlucky choice of 3 held-out
subjects can make a class look much better or worse than it really is.
Averaging over several different splits gives a far more trustworthy
picture -- especially for a small class like amusement.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

from baseline_model import load_windows, LABEL_NAMES


def main(n_splits: int = 5):
    X, y, subject_ids = load_windows()
    gkf = GroupKFold(n_splits=n_splits)

    fold_f1s = {label: [] for label in LABEL_NAMES}
    fold_accuracy = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=subject_ids), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = (y_pred == y_test).mean()
        fold_accuracy.append(acc)

        f1s = f1_score(y_test, y_pred, labels=sorted(LABEL_NAMES), average=None, zero_division=0)
        for label, f1 in zip(sorted(LABEL_NAMES), f1s):
            fold_f1s[label].append(f1)

        print(f"Fold {fold}: accuracy={acc:.2f}, test subjects={sorted(set(subject_ids[test_idx]))}")

    print(f"\n=== Averaged over {n_splits} folds ===")
    print(f"Mean accuracy: {np.mean(fold_accuracy):.2f} (+/- {np.std(fold_accuracy):.2f})")
    for label in sorted(LABEL_NAMES):
        f1s = fold_f1s[label]
        print(f"  {LABEL_NAMES[label]:10s} mean F1: {np.mean(f1s):.2f} (+/- {np.std(f1s):.2f})")


if __name__ == "__main__":
    main()
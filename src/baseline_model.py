"""
baseline_model.py
------------------
STEP 3: a fast, simple classical model on the windowed features from
Step 2. This is your sanity check and your floor: if this can't beat
random guessing, the bug is almost certainly in the data pipeline, not
in whatever deep learning model you build later.

Critical detail: we split train/test BY SUBJECT, not by window. Windows
from the same subject are correlated (similar resting heart rate, skin
conductance baseline, etc.), so mixing a subject's windows across both
train and test leaks information and makes accuracy look better than it
really is. GroupShuffleSplit below enforces this automatically.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix


LABEL_NAMES = {1: "baseline", 2: "stress", 3: "amusement"}


def load_windows(path: str = "data/processed/windows.npz"):
    data = np.load(path, allow_pickle=True)
    return data["X"], data["y"], data["subject_ids"]


def subject_based_split(X, y, subject_ids, test_size: float = 0.2, seed: int = 42):
    """
    Split so that ALL windows from a given subject go entirely into
    either train or test -- never split across both.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups=subject_ids))
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx], subject_ids[train_idx], subject_ids[test_idx]


def main():
    X, y, subject_ids = load_windows()
    print(f"Loaded {len(y)} windows from {len(np.unique(subject_ids))} subjects.\n")

    X_train, X_test, y_train, y_test, subj_train, subj_test = subject_based_split(X, y, subject_ids)

    print(f"Train: {len(y_train)} windows from subjects {sorted(set(subj_train))}")
    print(f"Test:  {len(y_test)} windows from subjects {sorted(set(subj_test))}\n")

    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("=== Classification report (test subjects, never seen during training) ===")
    target_names = [LABEL_NAMES[l] for l in sorted(LABEL_NAMES)]
    print(classification_report(y_test, y_pred, labels=sorted(LABEL_NAMES), target_names=target_names, zero_division=0))

    print("=== Confusion matrix ===")
    print("Rows = true label, columns = predicted label, order:", target_names)
    print(confusion_matrix(y_test, y_pred, labels=sorted(LABEL_NAMES)))


if __name__ == "__main__":
    main()
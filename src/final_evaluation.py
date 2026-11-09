"""
final_evaluation.py
---------------------
STEP 7: the definitive comparison for your Results chapter. Trains a
fresh teacher per fold (as in Steps 4-6), distills student_target with
the reinforced penalty you've now chosen based on the sweep (default
5.0, matching your proposal's original alpha), and reports:

  - full per-class precision/recall/F1 for both teacher and student
  - parameter count and compression ratio
  - CPU inference latency per window, compared against your proposal's
    target of under 50ms

Latency note: measured with torch.set_num_threads(1), to roughly
approximate a single-core constrained device (a Raspberry Pi Zero or
ESP32-class chip). This is a laptop CPU, not the actual target hardware,
so report this as an ESTIMATE, not a measured result on real deployment
hardware -- that distinction matters for your Methodology chapter.
"""

import argparse
import time
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, precision_recall_fscore_support

from train_teacher import (StressClassifier, load_raw_windows, downsample,
                           subject_normalize, predict_in_batches, LABEL_TO_INDEX,
                           train_one_model as train_teacher_model)
from train_student import STUDENT_CONFIGS, train_student, freeze, LABEL_TO_INDEX

LABEL_NAMES_ORDERED = ["baseline", "stress", "amusement"]
STRESS_INDEX = LABEL_TO_INDEX[2]  # 1


def measure_latency(model, X_sample: np.ndarray, n_repeats: int = 50) -> float:
    """
    Average time, in milliseconds, for ONE forward pass on ONE window.
    Runs single-threaded to roughly simulate a constrained device.
    """
    torch.set_num_threads(1)
    model.eval()
    x = torch.tensor(X_sample[:1], dtype=torch.float32)  # one window, shape (1, T, C)

    with torch.no_grad():
        for _ in range(5):  # warm-up, not timed
            model(x)
        start = time.perf_counter()
        for _ in range(n_repeats):
            model(x)
        elapsed = time.perf_counter() - start

    return (elapsed / n_repeats) * 1000  # ms per window


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/processed/raw_windows.npz")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--penalty", type=float, default=5.0)
    parser.add_argument("--downsample", type=int, default=8)
    args = parser.parse_args()

    X, y_raw, subject_ids = load_raw_windows(args.data)
    y = np.array([LABEL_TO_INDEX[label] for label in y_raw])
    X = downsample(X, args.downsample)
    X = subject_normalize(X, subject_ids)

    cfg = STUDENT_CONFIGS["student_target"]
    gkf = GroupKFold(n_splits=5)

    teacher_preds_all, teacher_true_all = [], []
    student_preds_all, student_true_all = [], []
    teacher_params = student_params = None
    teacher_latency, student_latency = [], []

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=subject_ids), start=1):
        X_train, X_val = X[tr], X[te]
        y_train, y_val = y[tr], y[te]

        teacher, _ = train_teacher_model(X_train, y_train, X_val, y_val,
                                         hidden_size=128, num_layers=2,
                                         epochs=args.epochs, batch_size=args.batch_size,
                                         select_best=False, verbose=False)
        freeze(teacher)
        teacher_params = teacher.count_parameters()

        student = StressClassifier(X.shape[2], cfg["hidden_size"], cfg["num_layers"],
                                    conv_channels=cfg["conv_channels"])
        student, _ = train_student(student, teacher, X_train, y_train, X_val, y_val,
                                   epochs=args.epochs, batch_size=args.batch_size,
                                   temperature=args.temperature, alpha=args.alpha,
                                   penalty_class=STRESS_INDEX, penalty=args.penalty)
        student_params = student.count_parameters()

        t_preds = predict_in_batches(teacher, torch.tensor(X_val, dtype=torch.float32))
        s_preds = predict_in_batches(student, torch.tensor(X_val, dtype=torch.float32))

        teacher_preds_all.extend(t_preds)
        teacher_true_all.extend(y_val)
        student_preds_all.extend(s_preds)
        student_true_all.extend(y_val)

        teacher_latency.append(measure_latency(teacher, X_val))
        student_latency.append(measure_latency(student, X_val))

        print(f"Fold {fold} done: teacher_latency={teacher_latency[-1]:.2f}ms  "
              f"student_latency={student_latency[-1]:.2f}ms")

    print("\n" + "=" * 70)
    print("TEACHER -- pooled predictions across all 5 folds")
    print("=" * 70)
    print(classification_report(teacher_true_all, teacher_preds_all,
                                target_names=LABEL_NAMES_ORDERED, zero_division=0))

    print("=" * 70)
    print(f"STUDENT_TARGET (reinforced, penalty={args.penalty}) -- pooled predictions")
    print("=" * 70)
    print(classification_report(student_true_all, student_preds_all,
                                target_names=LABEL_NAMES_ORDERED, zero_division=0))

    reduction = 100 * (1 - student_params / teacher_params)
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'':20s} {'teacher':>12s} {'student_target':>16s}")
    print(f"{'Parameters':20s} {teacher_params:12d} {student_params:16d}")
    print(f"{'Compression':20s} {'--':>12s} {reduction:15.1f}%")
    print(f"{'Latency (ms/window)':20s} {np.mean(teacher_latency):12.2f} {np.mean(student_latency):16.2f}")
    print(f"{'Target from proposal':20s} {'--':>12s} {'< 50.0 ms':>16s}")

    status = "MEETS" if np.mean(student_latency) < 50.0 else "DOES NOT MEET"
    print(f"\nStudent {status} the <50ms latency target (on this CPU, single-threaded estimate).")


if __name__ == "__main__":
    main()
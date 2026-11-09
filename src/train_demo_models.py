"""
train_demo_models.py
---------------------
Trains the two models used in the presentation demo, in a way that keeps
the demo honest:

  - The 12 training subjects and 3 held-out subjects are the same split
    you used earlier (GroupShuffleSplit, seed 42), so the held-out
    people are S10, S4 and S6.
  - A fresh teacher is trained on the 12 training subjects only.
  - The student (the ~73% smaller "student_target" size) is distilled
    from that teacher with the reinforced loss (stress penalty 5.0).
  - Both models are trained for a FIXED number of epochs and the FINAL
    epoch is kept. Nothing is chosen by looking at the held-out
    subjects, so the demo numbers are not flattered by epoch selection.

Run:   python src\\train_demo_models.py
Takes a few minutes. Saves outputs/demo_teacher.pt and outputs/demo_student.pt
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import recall_score

from wesad_core import (StressClassifier, load_raw_windows, downsample,
                           subject_normalize, predict_in_batches, LABEL_TO_INDEX,
                           train_one_model)

STUDENT_CFG = {"conv_channels": 24, "hidden_size": 64, "num_layers": 2}  # student_target
STRESS_INDEX = LABEL_TO_INDEX[2]  # 1
CLASS_NAMES = ["baseline", "stress", "amusement"]


def reinforced_distillation_loss(student_logits, teacher_logits, y, penalty, temperature, alpha):
    """Distillation loss where true-stress samples count `penalty` times more."""
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    per_sample_kl = F.kl_div(soft_student, soft_teacher, reduction="none").sum(dim=1)
    per_sample_ce = F.cross_entropy(student_logits, y, reduction="none")
    weights = torch.ones_like(y, dtype=torch.float32)
    weights[y == STRESS_INDEX] = penalty
    distill = (per_sample_kl * weights).mean() * (temperature ** 2)
    hard = (per_sample_ce * weights).mean()
    return alpha * distill + (1 - alpha) * hard


def distill_student(student, teacher, X_train, y_train, epochs, batch_size,
                    penalty, temperature, alpha, lr=1e-3):
    teacher.eval()
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    with torch.no_grad():
        teacher_logits = teacher(X_t)
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    n = len(y_train)
    for _ in range(epochs):
        student.train()
        order = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            optimizer.zero_grad()
            loss = reinforced_distillation_loss(student(X_t[idx]), teacher_logits[idx], y_t[idx],
                                                penalty, temperature, alpha)
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
    return student


def report(name, model, X_val, y_val, subj_val):
    preds = predict_in_batches(model, torch.tensor(X_val, dtype=torch.float32))
    acc = float((preds == y_val).mean())
    rec = recall_score(y_val, preds, labels=[0, 1, 2], average=None, zero_division=0)
    print(f"\n{name}: accuracy={acc:.2f} | recall baseline/stress/amusement = "
          f"{rec[0]:.2f}/{rec[1]:.2f}/{rec[2]:.2f}")
    for s in sorted(set(subj_val)):
        m = subj_val == s
        a = float((preds[m] == y_val[m]).mean())
        stress_m = m & (y_val == STRESS_INDEX)
        sr = float((preds[stress_m] == STRESS_INDEX).mean()) if stress_m.any() else float("nan")
        print(f"    held-out {s}: accuracy={a:.2f}  stress recall={sr:.2f}  ({int(m.sum())} windows)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/raw_windows.npz")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--penalty", type=float, default=5.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--downsample", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    t0 = time.time()

    X, y_raw, subject_ids = load_raw_windows(args.data)
    y = np.array([LABEL_TO_INDEX[v] for v in y_raw])
    X = downsample(X, args.downsample)
    X = subject_normalize(X, subject_ids)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(X, y, groups=subject_ids))
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    heldout = sorted(set(str(s) for s in subject_ids[val_idx]))
    print(f"Training subjects: {len(set(subject_ids[train_idx]))} | held-out subjects: {heldout}")

    print("\n[1/2] Training teacher (fixed epochs, no selection on held-out subjects)...")
    teacher, _ = train_one_model(X_train, y_train, X_val, y_val, hidden_size=128, num_layers=2,
                                 epochs=args.epochs, batch_size=args.batch_size,
                                 select_best=False, verbose=False)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print("[2/2] Distilling student with the reinforced loss...")
    student = StressClassifier(X.shape[2], STUDENT_CFG["hidden_size"], STUDENT_CFG["num_layers"],
                               conv_channels=STUDENT_CFG["conv_channels"])
    student = distill_student(student, teacher, X_train, y_train, args.epochs, args.batch_size,
                              args.penalty, args.temperature, args.alpha)
    student.eval()

    print(f"\nParameters: teacher={teacher.count_parameters()}  student={student.count_parameters()}  "
          f"({100 * (1 - student.count_parameters() / teacher.count_parameters()):.1f}% smaller)")
    if (teacher.count_parameters(), student.count_parameters()) != (221251, 59923):
        print("\n*** WARNING: parameter counts differ from the thesis (221,251 / 59,923). "
              "The Results and How-it-works tabs hard-code those numbers. ***")
    subj_val = subject_ids[val_idx]
    report("TEACHER", teacher, X_val, y_val, subj_val)
    report("STUDENT (reinforced)", student, X_val, y_val, subj_val)

    Path("outputs").mkdir(exist_ok=True)
    common = {"input_channels": X.shape[2], "downsample": args.downsample,
              "heldout_subjects": heldout, "epochs": args.epochs}
    torch.save({"model_state": teacher.state_dict(), "hidden_size": 128, "num_layers": 2,
                "conv_channels": 32, **common}, "outputs/demo_teacher.pt")
    torch.save({"model_state": student.state_dict(), **STUDENT_CFG, "penalty": args.penalty,
                **common}, "outputs/demo_student.pt")
    print(f"\nSaved outputs/demo_teacher.pt and outputs/demo_student.pt  ({time.time() - t0:.0f}s total)")
    print("Next: python src\\build_web_app.py")


if __name__ == "__main__":
    main()

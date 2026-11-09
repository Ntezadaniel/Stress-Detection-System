"""
train_student.py
------------------
STEP 5: distill several student model sizes from a trained teacher.

Three modes:
  (default)   Uses the already-saved outputs/teacher_model.pt and the
              single train/val split, for a quick first look.
  --cv        Runs 5-fold cross-validation, training a FRESH teacher
              INSIDE each fold (only on that fold's training subjects),
              then distilling every student from that fold's own
              teacher, and evaluating on that fold's held-out subjects.
  --reinforced Compares PLAIN vs cost-sensitive distillation for
              student_target only, tracking stress recall specifically.

Why --cv retrains the teacher per fold rather than reusing one teacher:
outputs/teacher_model.pt was trained on one particular set of subjects.
Reusing it across all 5 folds would mean some folds' "held out" subjects
were actually in the teacher's own training set -- the student would be
distilling from a teacher that already knows those "unseen" people. That
is leakage, not a fair test.

Distillation loss = weighted sum of:
  - KL-divergence between temperature-softened student and teacher logits
  - standard cross-entropy against the true label
alpha controls the balance (alpha=1.0 means "only imitate the teacher").

Reinforced distillation loss (--reinforced) additionally weights every
sample whose TRUE label is stress by `penalty` times more, so missing a
real stress event costs more in training than a false alarm elsewhere --
the cost-sensitive "Reinforced" piece from your proposal's Phase 3.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import f1_score

from train_teacher import (StressClassifier, load_raw_windows, downsample,
                           subject_normalize, predict_in_batches, LABEL_TO_INDEX,
                           train_one_model as train_teacher_model)

STUDENT_CONFIGS = {
    "student_tiny":   {"conv_channels": 4,  "hidden_size": 8,  "num_layers": 1},
    "student_small":  {"conv_channels": 8,  "hidden_size": 16, "num_layers": 1},
    "student_medium": {"conv_channels": 16, "hidden_size": 32, "num_layers": 1},
    "student_target": {"conv_channels": 24, "hidden_size": 64, "num_layers": 2},  # ~73% smaller
}


def distillation_loss(student_logits, teacher_logits, true_labels,
                       temperature: float = 2.0, alpha: float = 0.7):
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    distill_term = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temperature ** 2)
    hard_term = F.cross_entropy(student_logits, true_labels)
    return alpha * distill_term + (1 - alpha) * hard_term


def reinforced_distillation_loss(student_logits, teacher_logits, true_labels,
                                  penalty_class: int, penalty: float = 5.0,
                                  temperature: float = 2.0, alpha: float = 0.7):
    """
    Same idea as distillation_loss, but every sample whose TRUE label is
    penalty_class (stress, in this project) counts `penalty` times more
    in the loss. This is the cost-sensitive "Reinforced" piece from your
    proposal's Phase 3: missing a real stress event should cost more than
    a false alarm on another class.

    Implementation note: we use reduction="none" to get one loss value
    PER SAMPLE, multiply by that sample's weight, and only then average --
    doing it the other way around (averaging first) would erase the
    per-sample weighting entirely.
    """
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    per_sample_kl = F.kl_div(soft_student, soft_teacher, reduction="none").sum(dim=1)
    per_sample_ce = F.cross_entropy(student_logits, true_labels, reduction="none")

    weights = torch.ones_like(true_labels, dtype=torch.float32)
    weights[true_labels == penalty_class] = penalty

    distill_term = (per_sample_kl * weights).mean() * (temperature ** 2)
    hard_term = (per_sample_ce * weights).mean()
    return alpha * distill_term + (1 - alpha) * hard_term


def load_teacher(path: str = "outputs/teacher_model.pt"):
    ckpt = torch.load(path, weights_only=False)
    teacher = StressClassifier(ckpt["input_channels"], ckpt["hidden_size"],
                                ckpt["num_layers"], conv_channels=ckpt["conv_channels"])
    teacher.load_state_dict(ckpt["model_state"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher, ckpt["downsample"]


def freeze(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def train_student(student, teacher, X_train, y_train, X_val, y_val,
                   epochs=40, lr=1e-3, batch_size=16, temperature=2.0, alpha=0.7,
                   penalty_class=None, penalty=1.0):
    """
    penalty_class=None (default): plain distillation, every sample weighted equally.
    penalty_class=<index>, penalty>1: reinforced distillation -- samples whose
        TRUE label is penalty_class count `penalty` times more in the loss.
    """
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)

    with torch.no_grad():
        teacher_train_logits = teacher(X_train_t)

    best_f1, best_state = -1.0, None
    n_train = len(y_train)

    for epoch in range(1, epochs + 1):
        student.train()
        order = torch.randperm(n_train)
        for start in range(0, n_train, batch_size):
            idx = order[start:start + batch_size]
            optimizer.zero_grad()
            student_logits = student(X_train_t[idx])
            if penalty_class is None:
                loss = distillation_loss(student_logits, teacher_train_logits[idx],
                                          y_train_t[idx], temperature, alpha)
            else:
                loss = reinforced_distillation_loss(student_logits, teacher_train_logits[idx],
                                                     y_train_t[idx], penalty_class, penalty,
                                                     temperature, alpha)
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

        val_preds = predict_in_batches(student, X_val_t)
        val_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.clone() for k, v in student.state_dict().items()}

    student.load_state_dict(best_state)
    return student, best_f1


def run_single_split(X, y, subject_ids, args):
    teacher, ds_factor = load_teacher(args.teacher)
    teacher_params = teacher.count_parameters()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(X, y, groups=subject_ids))
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    print(f"Teacher parameters: {teacher_params}\n")

    results = []
    for name, cfg in STUDENT_CONFIGS.items():
        student = StressClassifier(X.shape[2], cfg["hidden_size"], cfg["num_layers"],
                                    conv_channels=cfg["conv_channels"])
        n_params = student.count_parameters()
        reduction = 100 * (1 - n_params / teacher_params)
        print(f"Training {name} ({n_params} params, {reduction:.1f}% smaller than teacher)...")
        student, _ = train_student(student, teacher, X_train, y_train, X_val, y_val,
                                    epochs=args.epochs, batch_size=args.batch_size,
                                    temperature=args.temperature, alpha=args.alpha)
        val_preds = predict_in_batches(student, torch.tensor(X_val, dtype=torch.float32))
        macro_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
        per_class = f1_score(y_val, val_preds, labels=[0, 1, 2], average=None, zero_division=0)
        results.append((name, n_params, reduction, macro_f1, per_class))
        Path("outputs").mkdir(exist_ok=True)
        torch.save({"model_state": student.state_dict(), "hidden_size": cfg["hidden_size"],
                    "num_layers": cfg["num_layers"], "conv_channels": cfg["conv_channels"],
                    "input_channels": X.shape[2]}, f"outputs/{name}.pt")

    print("\n" + "=" * 95)
    print(f"{'Model':16s} {'params':>8s} {'vs teacher':>11s} {'macro F1':>9s} {'baseline':>9s} {'stress':>8s} {'amusement':>10s}")
    print("=" * 95)
    print(f"{'teacher':16s} {teacher_params:8d} {'--':>11s}")
    for name, n_params, reduction, macro_f1, per_class in results:
        print(f"{name:16s} {n_params:8d} {reduction:10.1f}% {macro_f1:9.2f} "
              f"{per_class[0]:9.2f} {per_class[1]:8.2f} {per_class[2]:10.2f}")


def run_cv(X, y, subject_ids, args):
    gkf = GroupKFold(n_splits=5)
    teacher_accs, teacher_f1s = [], []
    student_accs = {name: [] for name in STUDENT_CONFIGS}
    student_f1s = {name: [] for name in STUDENT_CONFIGS}

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=subject_ids), start=1):
        X_train, X_val = X[tr], X[te]
        y_train, y_val = y[tr], y[te]

        # Fresh teacher, trained ONLY on this fold's training subjects.
        teacher, _ = train_teacher_model(X_train, y_train, X_val, y_val,
                                         hidden_size=128, num_layers=2,
                                         epochs=args.epochs, batch_size=args.batch_size,
                                         select_best=False, verbose=False)
        freeze(teacher)

        t_preds = predict_in_batches(teacher, torch.tensor(X_val, dtype=torch.float32))
        t_acc = float((t_preds == y_val).mean())
        t_f1 = f1_score(y_val, t_preds, labels=[0, 1, 2], average=None, zero_division=0)
        teacher_accs.append(t_acc)
        teacher_f1s.append(t_f1)

        fold_line = f"Fold {fold}: teacher acc={t_acc:.2f}"
        for name, cfg in STUDENT_CONFIGS.items():
            student = StressClassifier(X.shape[2], cfg["hidden_size"], cfg["num_layers"],
                                        conv_channels=cfg["conv_channels"])
            student, _ = train_student(student, teacher, X_train, y_train, X_val, y_val,
                                       epochs=args.epochs, batch_size=args.batch_size,
                                       temperature=args.temperature, alpha=args.alpha)
            s_preds = predict_in_batches(student, torch.tensor(X_val, dtype=torch.float32))
            s_acc = float((s_preds == y_val).mean())
            s_f1 = f1_score(y_val, s_preds, labels=[0, 1, 2], average=None, zero_division=0)
            student_accs[name].append(s_acc)
            student_f1s[name].append(s_f1)
            fold_line += f" | {name} acc={s_acc:.2f}"
        print(fold_line)

    print("\n" + "=" * 90)
    print(f"{'Model':22s} {'mean acc':>12s} {'baseline F1':>13s} {'stress F1':>11s} {'amusement F1':>13s}")
    print("=" * 90)
    tf = np.array(teacher_f1s)
    print(f"{'teacher (fresh/fold)':22s} {np.mean(teacher_accs):.2f} +/- {np.std(teacher_accs):.2f}   "
          f"{tf[:,0].mean():5.2f}        {tf[:,1].mean():5.2f}       {tf[:,2].mean():5.2f}")
    for name in STUDENT_CONFIGS:
        sf = np.array(student_f1s[name])
        accs = np.array(student_accs[name])
        print(f"{name:22s} {accs.mean():.2f} +/- {accs.std():.2f}   "
              f"{sf[:,0].mean():5.2f}        {sf[:,1].mean():5.2f}       {sf[:,2].mean():5.2f}")
    print("-" * 90)
    print("Reference, same folds: RF 0.70 (0.79/0.71/0.29) | earlier teacher+norm CV 0.69 (0.73/0.81/0.32)")


def run_cv_reinforced(X, y, subject_ids, args):
    """
    Compares PLAIN vs REINFORCED distillation for student_target only,
    using the same fresh-teacher-per-fold structure as run_cv(). Tracks
    recall per class specifically (not just F1), since the penalty is
    meant to move recall on stress -- catching more real stress cases --
    possibly at the cost of precision (more false alarms on other classes).
    """
    from sklearn.metrics import recall_score
    STRESS_INDEX = LABEL_TO_INDEX[2]  # 1

    cfg = STUDENT_CONFIGS["student_target"]
    gkf = GroupKFold(n_splits=5)

    plain_recalls, reinforced_recalls = [], []
    plain_f1s, reinforced_f1s = [], []

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=subject_ids), start=1):
        X_train, X_val = X[tr], X[te]
        y_train, y_val = y[tr], y[te]

        teacher, _ = train_teacher_model(X_train, y_train, X_val, y_val,
                                         hidden_size=128, num_layers=2,
                                         epochs=args.epochs, batch_size=args.batch_size,
                                         select_best=False, verbose=False)
        freeze(teacher)

        plain = StressClassifier(X.shape[2], cfg["hidden_size"], cfg["num_layers"],
                                  conv_channels=cfg["conv_channels"])
        plain, _ = train_student(plain, teacher, X_train, y_train, X_val, y_val,
                                 epochs=args.epochs, batch_size=args.batch_size,
                                 temperature=args.temperature, alpha=args.alpha)

        reinforced = StressClassifier(X.shape[2], cfg["hidden_size"], cfg["num_layers"],
                                       conv_channels=cfg["conv_channels"])
        reinforced, _ = train_student(reinforced, teacher, X_train, y_train, X_val, y_val,
                                      epochs=args.epochs, batch_size=args.batch_size,
                                      temperature=args.temperature, alpha=args.alpha,
                                      penalty_class=STRESS_INDEX, penalty=args.penalty)

        p_preds = predict_in_batches(plain, torch.tensor(X_val, dtype=torch.float32))
        r_preds = predict_in_batches(reinforced, torch.tensor(X_val, dtype=torch.float32))

        p_recall = recall_score(y_val, p_preds, labels=[0, 1, 2], average=None, zero_division=0)
        r_recall = recall_score(y_val, r_preds, labels=[0, 1, 2], average=None, zero_division=0)
        p_f1 = f1_score(y_val, p_preds, average="macro", zero_division=0)
        r_f1 = f1_score(y_val, r_preds, average="macro", zero_division=0)

        plain_recalls.append(p_recall)
        reinforced_recalls.append(r_recall)
        plain_f1s.append(p_f1)
        reinforced_f1s.append(r_f1)

        print(f"Fold {fold}: plain stress recall={p_recall[1]:.2f}  "
              f"reinforced stress recall={r_recall[1]:.2f}  "
              f"(penalty={args.penalty})")

    pr = np.array(plain_recalls)
    rr = np.array(reinforced_recalls)
    names = ["baseline", "stress", "amusement"]

    print("\n" + "=" * 78)
    print(f"student_target: PLAIN vs REINFORCED distillation (penalty={args.penalty})")
    print("=" * 78)
    print(f"{'':12s} {'plain recall':>14s} {'reinforced recall':>19s} {'change':>10s}")
    for i, name in enumerate(names):
        print(f"{name:12s} {pr[:, i].mean():14.2f} {rr[:, i].mean():19.2f} "
              f"{rr[:, i].mean() - pr[:, i].mean():+10.2f}")
    print("-" * 78)
    print(f"{'macro F1':12s} {np.mean(plain_f1s):14.2f} {np.mean(reinforced_f1s):19.2f} "
          f"{np.mean(reinforced_f1s) - np.mean(plain_f1s):+10.2f}")

def run_penalty_sweep(X, y, subject_ids, args):
    """
    Sweeps several penalty values for student_target, sharing each fold's
    teacher across every penalty tested in that fold (the teacher is
    retrained once per fold, not once per penalty -- it's the same
    fold's own held-out subjects either way, so reusing it across
    penalties within a fold is not leakage).

    penalty=1.0 is mathematically identical to plain distillation (every
    sample weighted equally), so it doubles as the sweep's baseline point.
    """
    from sklearn.metrics import recall_score
    STRESS_INDEX = LABEL_TO_INDEX[2]  # 1
    cfg = STUDENT_CONFIGS["student_target"]
    gkf = GroupKFold(n_splits=5)

    penalties = [float(p) for p in args.penalties.split(",")]
    recalls_by_penalty = {p: [] for p in penalties}
    f1_by_penalty = {p: [] for p in penalties}

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=subject_ids), start=1):
        X_train, X_val = X[tr], X[te]
        y_train, y_val = y[tr], y[te]

        teacher, _ = train_teacher_model(X_train, y_train, X_val, y_val,
                                         hidden_size=128, num_layers=2,
                                         epochs=args.epochs, batch_size=args.batch_size,
                                         select_best=False, verbose=False)
        freeze(teacher)

        fold_line = f"Fold {fold}:"
        for penalty in penalties:
            student = StressClassifier(X.shape[2], cfg["hidden_size"], cfg["num_layers"],
                                        conv_channels=cfg["conv_channels"])
            student, _ = train_student(student, teacher, X_train, y_train, X_val, y_val,
                                       epochs=args.epochs, batch_size=args.batch_size,
                                       temperature=args.temperature, alpha=args.alpha,
                                       penalty_class=STRESS_INDEX, penalty=penalty)
            preds = predict_in_batches(student, torch.tensor(X_val, dtype=torch.float32))
            recall = recall_score(y_val, preds, labels=[0, 1, 2], average=None, zero_division=0)
            macro_f1 = f1_score(y_val, preds, average="macro", zero_division=0)
            recalls_by_penalty[penalty].append(recall)
            f1_by_penalty[penalty].append(macro_f1)
            fold_line += f" | p={penalty:g} stress_recall={recall[1]:.2f}"
        print(fold_line)

    print("\n" + "=" * 84)
    print("student_target: penalty sweep, averaged over 5 folds (fresh teacher per fold)")
    print("=" * 84)
    print(f"{'penalty':>8s} {'baseline recall':>16s} {'stress recall':>15s} {'amusement recall':>18s} {'macro F1':>10s}")
    for penalty in penalties:
        r = np.array(recalls_by_penalty[penalty])
        f1s = np.array(f1_by_penalty[penalty])
        print(f"{penalty:8.1f} {r[:,0].mean():16.2f} {r[:,1].mean():15.2f} {r[:,2].mean():18.2f} {f1s.mean():10.2f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/processed/raw_windows.npz")
    parser.add_argument("--teacher", type=str, default="outputs/teacher_model.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--downsample", type=int, default=8)
    parser.add_argument("--cv", action="store_true")
    parser.add_argument("--reinforced", action="store_true",
                        help="compare plain vs cost-sensitive distillation for student_target")
    parser.add_argument("--penalty", type=float, default=5.0,
                        help="how many times more a missed stress sample counts, when --reinforced")
    parser.add_argument("--penalty-sweep", action="store_true",
                        help="sweep several penalty values for student_target and report the trade-off")
    parser.add_argument("--penalties", type=str, default="1,2,5,10",
                        help="comma-separated penalty values to test with --penalty-sweep")
    args = parser.parse_args()

    X, y_raw, subject_ids = load_raw_windows(args.data)
    y = np.array([LABEL_TO_INDEX[label] for label in y_raw])

    if args.penalty_sweep:
        X = downsample(X, args.downsample)
        X = subject_normalize(X, subject_ids)
        run_penalty_sweep(X, y, subject_ids, args)
    elif args.reinforced:
        X = downsample(X, args.downsample)
        X = subject_normalize(X, subject_ids)
        run_cv_reinforced(X, y, subject_ids, args)
    elif args.cv:
        X = downsample(X, args.downsample)
        X = subject_normalize(X, subject_ids)
        run_cv(X, y, subject_ids, args)
    else:
        _, ds_factor = load_teacher(args.teacher)
        X = downsample(X, ds_factor)
        X = subject_normalize(X, subject_ids)
        run_single_split(X, y, subject_ids, args)


if __name__ == "__main__":
    main()
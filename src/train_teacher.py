"""
train_teacher.py  (version 4)
------------------------------
STEP 4: train the larger "teacher" model on raw sensor sequences from
preprocessing_raw.py. Its job is to be as accurate as possible; later we
compress what it learned into small student models.

What changed from version 3, and why:
  - PER-SUBJECT NORMALIZATION replaces the earlier global (train-only)
    normalization. Testing showed this was the single biggest lever:
    teacher stress F1 went from 0.55 to 0.81 in 5-fold cross-validation
    once each subject's signals were standardized against their OWN
    mean/std, rather than one mean/std shared across all subjects.
    Physiologically this makes sense -- resting heart rate, skin
    conductance baseline, and skin temperature vary a lot between
    people, so a model trained on "raw" cross-subject values spends its
    capacity learning who a person is rather than what state they're in.

Kept from version 3: CNN front-end + LSTM, mini-batches, downsampling,
class weights, subject-based splits, --cv mode for fair comparison.
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import classification_report, f1_score

LABEL_NAMES = {1: "baseline", 2: "stress", 3: "amusement"}
LABEL_TO_INDEX = {1: 0, 2: 1, 3: 2}
INDEX_TO_LABEL = {v: k for k, v in LABEL_TO_INDEX.items()}


class StressClassifier(nn.Module):
    """CNN front-end + LSTM classifier. hidden_size / num_layers control size."""
    def __init__(self, input_channels: int, hidden_size: int, num_layers: int = 2,
                 num_classes: int = 3, conv_channels: int = 32, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, conv_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(conv_channels, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        z = self.conv(x.transpose(1, 2)).transpose(1, 2)
        _, (h_n, _) = self.lstm(z)
        return self.classifier(self.dropout(h_n[-1]))

    def count_parameters(self) -> int:
        # Counts ALL parameters, not just trainable ones -- this matters
        # because a frozen teacher (requires_grad=False, used during
        # distillation) still has the same number of weights in memory.
        # Model SIZE and TRAINABILITY are different questions.
        return sum(p.numel() for p in self.parameters())


def load_raw_windows(path: str = "data/processed/raw_windows.npz"):
    data = np.load(path)
    return data["X"], data["y"], data["subject_ids"]


def downsample(X: np.ndarray, factor: int) -> np.ndarray:
    """Average every `factor` consecutive timesteps: (N, T, C) -> (N, T//factor, C)."""
    if factor <= 1:
        return X
    n, t, c = X.shape
    t_trim = (t // factor) * factor
    return X[:, :t_trim, :].reshape(n, t_trim // factor, factor, c).mean(axis=2)


def normalize_fit(X_train: np.ndarray):
    """Compute per-channel mean/std from training data only (global -- kept
    for reference; the model actually used below is subject_normalize)."""
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    return mean, std


def subject_normalize(X: np.ndarray, subject_ids: np.ndarray) -> np.ndarray:
    """
    Z-score every subject's data using THAT subject's own mean/std,
    computed per feature/channel from their own windows only. No labels
    involved, so this is safe to apply before a train/val split -- each
    subject's normalization never sees another subject's data.

    On a real device this corresponds to a short, label-free calibration
    period per wearer, whose mean/std is then used to normalize their
    later readings. That is a deployment requirement worth stating
    explicitly in your Methodology chapter, not an implementation detail.

    Works for (N, F) feature arrays and (N, T, C) sequence arrays.
    """
    Xn = np.empty(X.shape, dtype=np.float64)
    for s in np.unique(subject_ids):
        mask = subject_ids == s
        Xs = X[mask]
        axes = tuple(range(Xs.ndim - 1))
        mean = Xs.mean(axis=axes, keepdims=True)
        std = Xs.std(axis=axes, keepdims=True) + 1e-8
        Xn[mask] = (Xs - mean) / std
    return Xn


def predict_in_batches(model, X_t, batch_size: int = 32):
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(X_t), batch_size):
            preds.append(model(X_t[start:start + batch_size]).argmax(dim=1))
    return torch.cat(preds).cpu().numpy()


def train_one_model(X_train, y_train, X_val, y_val, hidden_size=128, num_layers=2,
                    epochs=40, lr=1e-3, batch_size=16, select_best=True, verbose=True):
    model = StressClassifier(X_train.shape[2], hidden_size, num_layers)

    class_counts = np.maximum(np.bincount(y_train, minlength=3), 1)
    class_weights = torch.tensor(len(y_train) / (3 * class_counts), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)

    do_eval = select_best or verbose
    best_f1, best_state, history = -1.0, None, []
    n_train = len(y_train)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        order = torch.randperm(n_train)
        epoch_loss = 0.0
        for start in range(0, n_train, batch_size):
            idx = order[start:start + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(X_train_t[idx]), y_train_t[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n_train

        if do_eval:
            val_f1 = f1_score(y_val, predict_in_batches(model, X_val_t), average="macro", zero_division=0)
            history.append(val_f1)
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == epochs):
                print(f"  epoch {epoch:3d}: train_loss={epoch_loss:.4f}  val_macro_f1={val_f1:.3f}  ({time.time() - t0:.1f}s)")

    if select_best:
        model.load_state_dict(best_state)
    return model, history


def run_cv(X, y, subject_ids, args):
    """5-fold cross-validation grouped by subject; same folds as baseline_crossval.py."""
    gkf = GroupKFold(n_splits=5)
    accs, f1s = [], []
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=subject_ids), start=1):
        t0 = time.time()
        X_tr = subject_normalize(X[tr], subject_ids[tr])
        X_te = subject_normalize(X[te], subject_ids[te])
        model, _ = train_one_model(X_tr, y[tr], X_te, y[te], epochs=args.epochs,
                                   batch_size=args.batch_size, select_best=False, verbose=False)
        preds = predict_in_batches(model, torch.tensor(X_te, dtype=torch.float32))
        acc = float((preds == y[te]).mean())
        f = f1_score(y[te], preds, labels=[0, 1, 2], average=None, zero_division=0)
        accs.append(acc)
        f1s.append(f)
        print(f"Fold {fold}: accuracy={acc:.2f}  F1 baseline/stress/amusement = "
              f"{f[0]:.2f}/{f[1]:.2f}/{f[2]:.2f}  test subjects={sorted(set(subject_ids[te]))}  ({time.time() - t0:.0f}s)")

    f1s = np.array(f1s)
    print("\n=== Teacher, averaged over 5 folds (final epoch, no peeking) ===")
    print(f"Mean accuracy: {np.mean(accs):.2f} (+/- {np.std(accs):.2f})")
    for i in range(3):
        name = LABEL_NAMES[INDEX_TO_LABEL[i]]
        print(f"  {name:10s} mean F1: {f1s[:, i].mean():.2f} (+/- {f1s[:, i].std():.2f})")
    print("\nYour Step 3 Random Forest, same folds: accuracy 0.70 | baseline 0.79 | stress 0.71 | amusement 0.29")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/processed/raw_windows.npz")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--downsample", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv", action="store_true", help="run 5-fold grouped cross-validation instead")
    parser.add_argument("--out", type=str, default="outputs/teacher_model.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X, y_raw, subject_ids = load_raw_windows(args.data)
    y = np.array([LABEL_TO_INDEX[label] for label in y_raw])
    X = downsample(X, args.downsample)
    print(f"Sequence shape after downsampling: {X.shape[1:]} (time_steps, channels)")

    X = subject_normalize(X, subject_ids)

    if args.cv:
        run_cv(X, y, subject_ids, args)
        return

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(X, y, groups=subject_ids))
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    print(f"Train: {len(y_train)} windows from {len(set(subject_ids[train_idx]))} subjects")
    print(f"Val:   {len(y_val)} windows from {len(set(subject_ids[val_idx]))} subjects\n")

    print("Training teacher model (CNN front-end + 2-layer LSTM, hidden_size=128)...")
    model, history = train_one_model(X_train, y_train, X_val, y_val, epochs=args.epochs,
                                     batch_size=args.batch_size, select_best=True, verbose=True)

    print(f"\nFinal-epoch val macro F1: {history[-1]:.3f}")
    print(f"Best-epoch  val macro F1: {max(history):.3f}  (picked using these same windows, so slightly optimistic)")
    print(f"Teacher parameter count: {model.count_parameters()}")

    val_preds = predict_in_batches(model, torch.tensor(X_val, dtype=torch.float32))
    target_names = [LABEL_NAMES[INDEX_TO_LABEL[i]] for i in range(3)]
    print("\n=== Classification report (validation subjects, best-epoch model) ===")
    print(classification_report(y_val, val_preds, target_names=target_names, zero_division=0))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "architecture": "cnn_lstm",
        "hidden_size": 128,
        "num_layers": 2,
        "conv_channels": 32,
        "input_channels": X.shape[2],
        "downsample": args.downsample,
        "normalization": "per_subject",
    }, args.out)
    print(f"\nSaved teacher model to {args.out}")


if __name__ == "__main__":
    main()
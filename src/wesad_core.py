"""
wesad_core.py
-------------
Everything the demo scripts share: the CNN-LSTM model, data helpers and a plain trainer.
(Named wesad_core so it never collides with your own train_teacher.py.)

Architecture (Chapter 3.5): 2 x [Conv1d(k=5) -> ReLU -> MaxPool(2)] -> LSTM -> Dropout(0.3) -> Linear(3)
  teacher: conv 32, hidden 128, 2 layers  = 221,251 parameters
  student: conv 24, hidden  64, 2 layers  =  59,923 parameters
Layer names (conv.0, conv.3, lstm.*, classifier.*) are what the web page reads, so do not rename them.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LABEL_TO_INDEX = {1: 0, 2: 1, 3: 2}          # WESAD: 1 baseline, 2 stress, 3 amusement


class StressClassifier(nn.Module):
    def __init__(self, input_channels, hidden_size, num_layers, conv_channels=32, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, conv_channels, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(conv_channels, conv_channels, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2))
        self.lstm = nn.LSTM(conv_channels, hidden_size, num_layers, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 3)

    def forward(self, x):                      # x: (batch, time, channels)
        z = self.conv(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.lstm(z)
        return self.classifier(self.drop(out[:, -1]))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


def load_raw_windows(path):
    """Returns X (n, 1920, 4), y (raw labels 1/2/3), subject_ids. Accepts a few common key names."""
    d = np.load(path, allow_pickle=True)
    def pick(*names):
        for n in names:
            if n in d.files:
                return d[n]
        raise KeyError(f"None of {names} found in {path}; it contains {d.files}")
    return pick("X", "X_raw", "windows"), pick("y", "labels", "y_raw"), pick("subject_ids", "subjects", "groups")


def downsample(X, factor):
    """Average every `factor` consecutive timesteps (32 Hz -> 4 Hz with factor 8)."""
    n, T, C = X.shape
    T2 = T // factor
    return X[:, :T2 * factor].reshape(n, T2, factor, C).mean(axis=2)


def subject_normalize(X, subject_ids):
    """Per-subject standardisation, per channel, using no labels (Chapter 3.3.3)."""
    X = X.astype(np.float64).copy()
    for s in np.unique(subject_ids):
        m = subject_ids == s
        X[m] = (X[m] - X[m].mean(axis=(0, 1), keepdims=True)) / (X[m].std(axis=(0, 1), keepdims=True) + 1e-8)
    return X


def predict_in_batches(model, X, batch=64):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(model(X[i:i + batch]).argmax(1).numpy())
    return np.concatenate(out)


def train_one_model(X_train, y_train, X_val, y_val, hidden_size, num_layers, epochs, batch_size,
                    select_best=False, verbose=False, lr=1e-3, conv_channels=32):
    """Plain cross-entropy training for a FIXED number of epochs; the final epoch is kept."""
    model = StressClassifier(X_train.shape[2], hidden_size, num_layers, conv_channels=conv_channels)
    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        order = torch.randperm(len(yt))
        for i in range(0, len(yt), batch_size):
            idx = order[i:i + batch_size]
            opt.zero_grad()
            F.cross_entropy(model(Xt[idx]), yt[idx]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if verbose and (ep + 1) % 10 == 0:
            print(f"   epoch {ep + 1}/{epochs}")
    model.eval()
    return model, None

"""
student_models.py
------------------
Defines several student model SIZES so you can compare accuracy vs.
compression trade-offs in Step 5, rather than committing to one
architecture on your first attempt.

Your proposal targets ~74% parameter reduction while retaining 80-85%
F1-score -- you won't know which student size actually hits that trade-off
until you train a few and compare them side by side. That comparison
IS a result for your thesis, not just an implementation detail.

Each model takes a window of multimodal sensor data (shape: batch x
time_steps x channels) and outputs logits over your stress classes.
"""

import torch
import torch.nn as nn


class StressClassifier(nn.Module):
    """
    A small LSTM-based classifier. hidden_size and num_layers control
    how large (and how accurate, and how slow) the model is.
    """
    def __init__(self, input_channels: int, hidden_size: int,
                 num_layers: int = 1, num_classes: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(input_channels, hidden_size, num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch, time_steps, input_channels)
        _, (h_n, _) = self.lstm(x)
        return self.classifier(h_n[-1])  # use the final hidden state

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Three candidate student sizes. Train and evaluate ALL of them in Step 5,
# then report the compression-vs-accuracy trade-off for each -- your
# thesis should show this comparison, not just the one you picked.
STUDENT_CONFIGS = {
    "student_tiny":   {"hidden_size": 8,  "num_layers": 1},
    "student_small":  {"hidden_size": 16, "num_layers": 1},
    "student_medium": {"hidden_size": 32, "num_layers": 2},
    "student_target": {"hidden_size": 64, "num_layers": 3},  # ~74.6% smaller than the
                                                              # teacher below -- matches
                                                              # the compression ratio your
                                                              # proposal already commits to
}


def build_student(name: str, input_channels: int, num_classes: int = 3) -> StressClassifier:
    if name not in STUDENT_CONFIGS:
        raise ValueError(f"Unknown student config '{name}'. Choose from {list(STUDENT_CONFIGS)}")
    cfg = STUDENT_CONFIGS[name]
    return StressClassifier(input_channels, cfg["hidden_size"], cfg["num_layers"], num_classes)


def build_teacher(input_channels: int, num_classes: int = 3) -> StressClassifier:
    """
    A deliberately larger version of the same architecture, to act as
    your Step 4 teacher. Using the same class keeps the comparison in
    Step 6 apples-to-apples: same architecture family, different size.
    """
    return StressClassifier(input_channels, hidden_size=128, num_layers=3, num_classes=num_classes)


if __name__ == "__main__":
    input_channels = 4  # e.g. BVP, EDA, TEMP, ACC-magnitude, per timestep
    teacher = build_teacher(input_channels)
    teacher_params = teacher.count_parameters()
    print(f"teacher: {teacher_params} parameters\n")

    print("Student candidates (compare these in Step 6):")
    for name in STUDENT_CONFIGS:
        model = build_student(name, input_channels)
        params = model.count_parameters()
        reduction = 100 * (1 - params / teacher_params)
        print(f"  {name:15s} {params:6d} parameters   ({reduction:.1f}% smaller than teacher)")

    # Confirm a forward pass actually works before you build a training loop around it.
    dummy_batch = torch.randn(5, 60, input_channels)  # 5 windows, 60 timesteps, 4 channels
    out = build_student("student_small", input_channels)(dummy_batch)
    print(f"\nForward pass output shape: {out.shape}  (should be [5, 3] -- 5 windows, 3 classes)")

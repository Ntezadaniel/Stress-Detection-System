"""
explore.py
----------
STEP 1 of the project: load one subject's data and actually LOOK at it
before writing any model code. This is the step most people skip, and
it's the step that saves you the most debugging time later.

Run this with synthetic data first (works immediately, no download needed):
    python src/explore.py --synthetic

Once you've downloaded and extracted real WESAD data into data/raw/WESAD/,
run it again on real data:
    python src/explore.py --subject S2

What to look for when you run this:
    1. Do the signal shapes match the expected sampling rates?
       (BVP should have ~64 samples per second of recording, etc.)
    2. Does the label array show clear blocks of 1 (baseline), 2 (stress),
       3 (amusement)? If everything is one label, something's wrong.
    3. Do the EDA and BVP plots visibly change between baseline and stress
       segments? This is the signal your model will have to learn to detect.
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # so this runs headlessly and just saves a file
import matplotlib.pyplot as plt

from data_loader import load_subject_raw, LABEL_SAMPLING_RATE, VALID_LABELS
from synthetic_data import generate_synthetic_subject


def summarize(subject: dict):
    print(f"\n=== Subject: {subject['subject']} ===")
    for name, arr in subject["wrist"].items():
        print(f"  {name:5s} shape={arr.shape}")
    labels = subject["label"]
    print(f"  label shape={labels.shape}")
    unique, counts = np.unique(labels, return_counts=True)
    print("  label distribution:")
    for u, c in zip(unique, counts):
        name = VALID_LABELS.get(u, "other/ignore")
        seconds = c / LABEL_SAMPLING_RATE
        print(f"    label {u} ({name}): {c} samples (~{seconds:.0f}s)")


def plot_signals(subject: dict, out_path: str = "outputs/signal_overview.png"):
    fig, axes = plt.subplots(len(subject["wrist"]) + 1, 1, figsize=(10, 9), sharex=False)

    for ax, (name, signal) in zip(axes, subject["wrist"].items()):
        if signal.ndim == 2:  # ACC has 3 axes
            for i, axis_name in enumerate(["x", "y", "z"]):
                ax.plot(signal[:, i], label=axis_name, linewidth=0.5)
            ax.legend(fontsize=7)
        else:
            ax.plot(signal, linewidth=0.5)
        ax.set_title(name)
        ax.set_xlabel("sample index")

    axes[-1].plot(subject["label"], color="black", linewidth=0.8)
    axes[-1].set_title("label (1=baseline, 2=stress, 3=amusement)")
    axes[-1].set_xlabel("sample index (700 Hz)")

    fig.suptitle(f"Signal overview: {subject['subject']}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true",
                         help="Use fake data instead of real WESAD")
    parser.add_argument("--subject", type=str, default="S2",
                         help="Real subject id to load, e.g. S2 (ignored if --synthetic)")
    parser.add_argument("--wesad-root", type=str, default="data/raw/WESAD")
    args = parser.parse_args()

    if args.synthetic:
        subject = generate_synthetic_subject(duration_sec=600, seed=0)
    else:
        from pathlib import Path
        subject = load_subject_raw(Path(args.wesad_root) / args.subject)

    summarize(subject)
    plot_signals(subject)

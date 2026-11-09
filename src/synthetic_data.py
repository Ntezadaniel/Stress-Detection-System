"""
synthetic_data.py
------------------
Generates fake data with the SAME shape and sampling rates as real WESAD
wrist-device data. This lets you write, run, and debug your entire
pipeline today -- before your 2.5GB WESAD download even finishes.

IMPORTANT: this data is not real physiology. It exists only so you can
confirm your code runs correctly. Every result you produce with this
data is meaningless for your actual research -- swap in data_loader.py
(real WESAD) before you record any results for your thesis.
"""

import numpy as np

from data_loader import WRIST_SAMPLING_RATES, VALID_LABELS


def generate_synthetic_subject(subject_id: str = "S_fake", duration_sec: int = 600,
                                seed: int | None = None) -> dict:
    """
    Generate one fake subject's wrist signals + labels.

    duration_sec: length of the fake recording, in seconds (600s = 10 min,
    short enough to generate and process quickly while you're testing).
    """
    rng = np.random.default_rng(seed)

    wrist_signals = {}
    for name, hz in WRIST_SAMPLING_RATES.items():
        n_samples = duration_sec * hz
        if name == "ACC":
            # 3-axis accelerometer: small random jitter around gravity on Z
            signal = rng.normal(loc=[0, 0, 1.0], scale=0.05, size=(n_samples, 3))
        elif name == "BVP":
            # Rough sinusoid to mimic a pulse waveform, plus noise
            t = np.arange(n_samples) / hz
            signal = np.sin(2 * np.pi * 1.2 * t) + rng.normal(0, 0.1, n_samples)
        elif name == "EDA":
            # Slow-varying signal with occasional "arousal" spikes
            signal = np.cumsum(rng.normal(0, 0.01, n_samples)) + 2.0
        else:  # TEMP
            signal = rng.normal(loc=33.0, scale=0.2, size=n_samples)
        wrist_signals[name] = signal

    # Labels at 700 Hz (the WESAD reference clock), randomly split into
    # contiguous blocks of baseline / stress / amusement, like a real protocol.
    label_hz = 700
    n_label_samples = duration_sec * label_hz
    labels = np.ones(n_label_samples, dtype=int)  # start as baseline
    block_size = n_label_samples // 6
    pattern = [1, 1, 2, 1, 3, 1]  # baseline-heavy with stress + amusement blocks
    for i, lab in enumerate(pattern):
        start, end = i * block_size, (i + 1) * block_size
        labels[start:end] = lab

    return {"wrist": wrist_signals, "label": labels, "subject": subject_id}


def generate_synthetic_dataset(n_subjects: int = 5, duration_sec: int = 600) -> list:
    """Generate a small fake multi-subject dataset for pipeline testing."""
    return [
        generate_synthetic_subject(subject_id=f"S_fake{i}", duration_sec=duration_sec, seed=i)
        for i in range(n_subjects)
    ]


if __name__ == "__main__":
    data = generate_synthetic_dataset(n_subjects=3, duration_sec=300)
    for s in data:
        print(f"\nSubject: {s['subject']}")
        for name, arr in s["wrist"].items():
            print(f"  {name}: shape={arr.shape}")
        print(f"  label: shape={s['label'].shape}, unique values={np.unique(s['label'])}")
